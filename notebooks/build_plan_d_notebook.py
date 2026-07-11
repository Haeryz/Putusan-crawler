#!/usr/bin/env python3
"""Generate the authoritative Plan D Colab notebook.

The generated notebook deliberately contains no outputs or execution counts.  Keep
this source beside the notebook so large notebook rewrites remain reviewable.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Qwen3_5_(4B)(1).ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"plan-d-{len(cells):02d}",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"plan-d-{len(cells):02d}",
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells: list[dict] = []

cells.append(markdown(r"""
# Plan D — 32K partial-field conversational SFT

Authoritative implementation of `Plan D.md` for `Qwen/Qwen3.5-9B`. This
notebook executes the data audit/build, hard gates, budgeted one-epoch QLoRA
run, and merged raw-window evaluation from top to bottom.

There are no smoke, dry, rehearsal, debug, or tiny-subset training paths. The
only GPU training call is the real resumable run. Builder checks are CPU-only;
no example is truncated; raw windows are sliced directly from `input_text`;
and `sections_json` is used only to construct supervision targets/audit
metadata, never to select or assemble source text. Rotate any previously
exposed W&B credential before running this notebook.
"""))

cells.append(markdown(r"""
## 1. Install runtime dependencies

Run in a fresh Colab runtime. Dependency installation is not a training run.
"""))

cells.append(code(r"""
%%capture
import os, sys
!pip install -U "unsloth[colab-new]" datasets transformers trl accelerate bitsandbytes wandb pyarrow pandas rapidfuzz
"""))

cells.append(markdown(r"""
## 2. Fixed contract and compute budget

Set the available wall-clock before the build. The epoch token budget is fixed
from that value and the conservative planning throughput; measured throughput
comes from the first logged steps of the real run.
"""))

cells.append(code(r"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoProcessor, set_seed

SEED = 3407
MODEL_ID = "Qwen/Qwen3.5-9B"
DATASET_REPO = os.getenv("PLAN_D_DATASET_REPO", "Haeryz/putusan-structured-extraction")
ARTIFACT_DIR = Path(os.getenv("PLAN_D_ARTIFACT_DIR", "/content/plan_d_dataset"))
MAX_SEQ_LENGTH = 32_768
DEFAULT_BUILD_TOKEN_CEILING = 32_256
BUILD_TOKEN_CEILING = int(os.getenv("PLAN_D_BUILD_CEILING", str(DEFAULT_BUILD_TOKEN_CEILING)))
SAFETY_MARGIN = MAX_SEQ_LENGTH - BUILD_TOKEN_CEILING
WINDOW_CONTEXT_TOKENS = min(15_360, max(1_024, (BUILD_TOKEN_CEILING - 512) // 2))
WINDOW_OVERLAP_FRACTION = 0.25

# Change this one value to match the booked session before starting the build.
SESSION_WALLCLOCK_HOURS = float(os.getenv("PLAN_D_SESSION_HOURS", "28"))
NON_TRAINING_RESERVE_HOURS = float(os.getenv("PLAN_D_RESERVE_HOURS", "2"))
PLANNING_TOKENS_PER_SECOND = 1_500
MAX_PLANNED_EPOCH_TOKENS = 150_000_000
usable_seconds = max(0.0, SESSION_WALLCLOCK_HOURS - NON_TRAINING_RESERVE_HOURS) * 3600
EPOCH_TOKEN_BUDGET = min(
    MAX_PLANNED_EPOCH_TOKENS,
    int(usable_seconds * PLANNING_TOKENS_PER_SECOND),
)
if EPOCH_TOKEN_BUDGET <= 0:
    raise ValueError("The configured session leaves no positive training-token budget.")

CANONICAL_FIELDS = [
    "judul", "nomor_putusan", "irah_irah", "nama_pengadilan_negeri",
    "keterangan_perkara", "nama_lengkap", "tempat_lahir", "umur_tanggal_lahir",
    "jenis_kelamin", "kebangsaan", "tempat_tinggal", "agama", "pekerjaan",
    "penangkapan", "penahanan", "tuntutan", "dakwaan", "saksi", "ahli",
    "terdakwa", "surat", "petunjuk_barang_bukti", "fakta_hukum",
    "pertimbangan_hukum", "amar_putusan", "hari", "tanggal", "tahun",
    "siapa_yang_memutus", "panitera_pengganti", "tanda_tangan_majelis",
]
FIELD_GROUPS = collections.OrderedDict([
    ("header", ["judul", "nomor_putusan", "irah_irah"]),
    ("court_case", ["nama_pengadilan_negeri", "keterangan_perkara"]),
    ("identity_a", ["nama_lengkap", "tempat_lahir", "umur_tanggal_lahir"]),
    ("identity_b", ["jenis_kelamin", "kebangsaan", "tempat_tinggal"]),
    ("identity_c", ["agama", "pekerjaan"]),
    ("custody", ["penangkapan", "penahanan"]),
    ("prosecution", ["tuntutan", "dakwaan"]),
    ("testimony", ["saksi", "ahli"]),
    ("evidence", ["terdakwa", "surat", "petunjuk_barang_bukti"]),
    ("analysis", ["fakta_hukum", "pertimbangan_hukum"]),
    ("ruling", ["amar_putusan"]),
    ("date", ["hari", "tanggal", "tahun"]),
    ("bench", ["siapa_yang_memutus", "panitera_pengganti", "tanda_tangan_majelis"]),
])

SYSTEM_PROMPT = (
    "Anda adalah pengekstrak teks putusan pengadilan Indonesia. Berdasarkan hanya "
    "konteks putusan yang diberikan, yang mungkin merupakan kutipan terbatas dari "
    "putusan yang lebih panjang, ekstrak hanya bagian yang diminta sebagaimana "
    "bagian tersebut muncul dalam kutipan ini. Setiap nilai harus berupa daftar "
    "kutipan verbatim yang disalin persis dari konteks; jangan memparafrasekan, "
    "meringkas, menambah, atau mengarang. Jika bagian yang diminta tidak terdapat "
    "dalam konteks, gunakan daftar kosong dan cantumkan namanya pada empty_sections. "
    "Keluarkan hanya satu objek JSON tanpa markdown atau penjelasan."
)

flat_fields = [field for fields in FIELD_GROUPS.values() for field in fields]
assert len(FIELD_GROUPS) == 13
assert len(CANONICAL_FIELDS) == 31
assert flat_fields == CANONICAL_FIELDS
assert len(set(flat_fields)) == 31
assert BUILD_TOKEN_CEILING <= DEFAULT_BUILD_TOKEN_CEILING
assert (DEFAULT_BUILD_TOKEN_CEILING - BUILD_TOKEN_CEILING) % 1_024 == 0
assert SAFETY_MARGIN >= 512
assert WINDOW_CONTEXT_TOKENS < BUILD_TOKEN_CEILING
set_seed(SEED)

RUN_STARTED_AT = time.perf_counter()
print(json.dumps({
    "model": MODEL_ID,
    "build_ceiling": BUILD_TOKEN_CEILING,
    "trainer_limit": MAX_SEQ_LENGTH,
    "safety_margin": SAFETY_MARGIN,
    "session_hours": SESSION_WALLCLOCK_HOURS,
    "epoch_token_budget": EPOCH_TOKEN_BUDGET,
    "seed": SEED,
}, indent=2))
"""))

cells.append(markdown(r"""
## 3. CPU tokenizer, source partitions, and leakage gate

The text tokenizer is obtained through the checkpoint processor. Train,
validation, and test are transformed identically; no split is regenerated.
"""))

cells.append(code(r"""
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
text_tokenizer = getattr(processor, "tokenizer", processor)
if not getattr(text_tokenizer, "is_fast", False):
    raise RuntimeError("Plan D raw-window slicing requires a fast tokenizer with offset mappings.")

raw = load_dataset(DATASET_REPO, "sft")
if "validation" not in raw and "val" in raw:
    raw["validation"] = raw.pop("val")
required_splits = ("train", "validation", "test")
if any(split not in raw for split in required_splits):
    raise KeyError(f"Expected train/validation/test, found {list(raw)}")

required_columns = {"id", "corpus", "annotator_model", "source_sha256", "input_text", "sections_json"}
for split in required_splits:
    missing = required_columns - set(raw[split].column_names)
    if missing:
        raise KeyError(f"{split} is missing required columns: {sorted(missing)}")

source_sets = {split: set(raw[split]["source_sha256"]) for split in required_splits}
for i, left in enumerate(required_splits):
    for right in required_splits[i + 1:]:
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            raise AssertionError(f"Cross-split leakage: {left}/{right} share {len(overlap)} sources")

print({split: len(raw[split]) for split in required_splits})
print("Cross-split source leakage: 0")
"""))

cells.append(markdown(r"""
## 4. Verbatim audit and lossless target relocation

Whitespace canonicalization locates annotations only. A repaired target is
always re-extracted from the original `input_text`; unresolved values are
excluded and counted by corpus and field. Coverage is analysis-only.
"""))

cells.append(code(r"""
@dataclasses.dataclass(frozen=True)
class LocatedSpan:
    text: str
    start: int
    end: int


def normalize_sections(value: Any) -> dict[str, list[str]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("sections_json must decode to an object")
    result: dict[str, list[str]] = {}
    for field in CANONICAL_FIELDS:
        spans = value.get(field, [])
        if spans is None:
            spans = []
        if isinstance(spans, str):
            spans = [spans]
        if not isinstance(spans, list):
            raise TypeError(f"{field} must be a list or string")
        result[field] = [str(span) for span in spans if str(span).strip()]
    return result


def canonicalize_with_map(text: str) -> tuple[str, list[int], list[int]]:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for match in re.finditer(r"\s+|\S", text):
        token = match.group(0)
        if token.isspace():
            chars.append(" ")
            starts.append(match.start())
            ends.append(match.end())
        else:
            chars.append(token)
            starts.append(match.start())
            ends.append(match.end())
    left = 0
    right = len(chars)
    while left < right and chars[left] == " ":
        left += 1
    while right > left and chars[right - 1] == " ":
        right -= 1
    return "".join(chars[left:right]), starts[left:right], ends[left:right]


def locate_verbatim(source: str, annotation: str, search_from: int = 0) -> tuple[LocatedSpan | None, str]:
    exact_at = source.find(annotation, search_from)
    if exact_at < 0 and search_from:
        exact_at = source.find(annotation)
    if exact_at >= 0:
        return LocatedSpan(annotation, exact_at, exact_at + len(annotation)), "exact"

    source_view = source[search_from:]
    source_canon, starts, ends = canonicalize_with_map(source_view)
    target_canon, _, _ = canonicalize_with_map(annotation)
    at = source_canon.find(target_canon) if target_canon else -1
    base = search_from
    if at < 0 and search_from:
        source_canon, starts, ends = canonicalize_with_map(source)
        at = source_canon.find(target_canon) if target_canon else -1
        base = 0
    if at < 0:
        return None, "excluded"
    start = base + starts[at]
    end = base + ends[at + len(target_canon) - 1]
    return LocatedSpan(source[start:end], start, end), "relocated"


def merged_interval_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((a, b) for a, b in intervals if b > a)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def audit_and_relocate(row: dict[str, Any], counters: collections.Counter) -> dict[str, list[LocatedSpan]]:
    source = str(row["input_text"])
    corpus = str(row["corpus"])
    sections = normalize_sections(row["sections_json"])
    repaired: dict[str, list[LocatedSpan]] = {field: [] for field in CANONICAL_FIELDS}
    all_intervals: list[tuple[int, int]] = []
    for field in CANONICAL_FIELDS:
        cursor = 0
        for annotation in sections[field]:
            counters[(corpus, field, "nonempty")] += 1
            located, status = locate_verbatim(source, annotation, cursor)
            counters[(corpus, field, status)] += 1
            if located is not None:
                repaired[field].append(located)
                all_intervals.append((located.start, located.end))
                cursor = located.end
    counters[(corpus, "__coverage__", "covered_chars")] += merged_interval_length(all_intervals)
    counters[(corpus, "__coverage__", "input_chars")] += len(source)
    return repaired


def audit_report(counters: collections.Counter) -> dict[str, Any]:
    report: dict[str, Any] = {"by_corpus_field": {}, "coverage_by_corpus": {}}
    corpora = sorted({key[0] for key in counters})
    for corpus in corpora:
        report["by_corpus_field"][corpus] = {}
        for field in CANONICAL_FIELDS:
            total = counters[(corpus, field, "nonempty")]
            exact = counters[(corpus, field, "exact")]
            relocated = counters[(corpus, field, "relocated")]
            excluded = counters[(corpus, field, "excluded")]
            report["by_corpus_field"][corpus][field] = {
                "nonempty": total,
                "exact": exact,
                "relocated": relocated,
                "excluded": excluded,
                "exact_rate": exact / total if total else None,
                "relocation_rate": relocated / total if total else None,
                "exclusion_rate": excluded / total if total else None,
            }
        covered = counters[(corpus, "__coverage__", "covered_chars")]
        chars = counters[(corpus, "__coverage__", "input_chars")]
        report["coverage_by_corpus"][corpus] = {
            "covered_chars": covered,
            "input_chars": chars,
            "coverage_rate": covered / chars if chars else None,
        }
    return report
"""))

cells.append(markdown(r"""
## 5. Full examples, raw contiguous windows, and D4 row gates

Window boundaries come exclusively from raw-text tokenizer offsets. They use
approximately one-quarter overlap. Labels are consulted only after the raw
windows exist, to construct the assistant response.
"""))

cells.append(code(r"""
def target_object(requested: list[str], values: dict[str, list[str]]) -> dict[str, Any]:
    sections = {field: list(values.get(field, [])) for field in requested}
    empty = [field for field in requested if not sections[field]]
    return {"sections": sections, "empty_sections": empty}


def make_messages(context: str, requested: list[str], target: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    requested_json = json.dumps(requested, ensure_ascii=False, separators=(",", ":"))
    user = f"Konteks putusan:\n{context}\n\nBagian yang diminta: {requested_json}"
    answer = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ], answer


def render_chat(messages: list[dict[str, str]]) -> str:
    return text_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )


def token_count(text: str) -> int:
    return len(text_tokenizer(text, add_special_tokens=False)["input_ids"])


def total_chat_tokens(messages: list[dict[str, str]]) -> int:
    return token_count(render_chat(messages))


def raw_windows(text: str) -> list[tuple[int, int]]:
    encoded = text_tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"] if int(b) > int(a)]
    if not offsets:
        return [(0, len(text))]
    if len(offsets) <= WINDOW_CONTEXT_TOKENS:
        return [(0, len(text))]
    stride = max(1, int(WINDOW_CONTEXT_TOKENS * (1.0 - WINDOW_OVERLAP_FRACTION)))
    windows: list[tuple[int, int]] = []
    start_token = 0
    while start_token < len(offsets):
        end_token = min(start_token + WINDOW_CONTEXT_TOKENS, len(offsets))
        start_char = 0 if start_token == 0 else offsets[start_token][0]
        end_char = len(text) if end_token == len(offsets) else offsets[end_token - 1][1]
        windows.append((start_char, end_char))
        if end_token == len(offsets):
            break
        start_token += stride
    return windows


def values_for_window(
    repaired: dict[str, list[LocatedSpan]],
    requested: list[str],
    text: str,
    window: tuple[int, int],
    windows: list[tuple[int, int]],
) -> dict[str, list[str]]:
    window_start, window_end = window
    result: dict[str, list[str]] = {}
    for field in requested:
        output: list[str] = []
        for span in repaired[field]:
            if window_start <= span.start and span.end <= window_end:
                output.append(span.text)
                continue
            fits_any_window = any(a <= span.start and span.end <= b for a, b in windows)
            if not fits_any_window:
                left = max(window_start, span.start)
                right = min(window_end, span.end)
                if right > left and text[left:right].strip():
                    output.append(text[left:right])
        result[field] = output
    return result


def overlapping_fields(repaired: dict[str, list[LocatedSpan]], window: tuple[int, int]) -> set[str]:
    start, end = window
    return {
        field
        for field, spans in repaired.items()
        if any(span.end > start and span.start < end for span in spans)
    }


def assert_row_gates(row: dict[str, Any], enforce_ceiling: bool = True) -> None:
    requested = row["requested_sections"]
    target = json.loads(row["target_json"])
    if list(target) != ["sections", "empty_sections"]:
        raise AssertionError("Assistant target has provenance or unexpected top-level keys")
    if list(target["sections"]) != requested:
        raise AssertionError("Assistant sections do not equal requested_sections in canonical order")
    expected_empty = [field for field in requested if not target["sections"][field]]
    if target["empty_sections"] != expected_empty:
        raise AssertionError("empty_sections is inconsistent")
    for values in target["sections"].values():
        if not isinstance(values, list):
            raise AssertionError("Every section value must be a list")
        for value in values:
            if value not in row["context_text"]:
                raise AssertionError("Non-empty target is not verbatim in context_text")
    if row["context_mode"] not in {"full", "window"}:
        raise AssertionError("Invalid context_mode")
    if enforce_ceiling and row["n_total_tokens"] > BUILD_TOKEN_CEILING:
        raise AssertionError("Built row exceeds the no-truncation ceiling")
    rendered = render_chat(row["messages"])
    if row["target_json"] not in rendered:
        raise AssertionError("Complete assistant answer is absent from rendered conversation")
    if not rendered.rstrip().endswith("<|im_end|>"):
        raise AssertionError("Rendered conversation does not end in the assistant end marker")
    prompt = text_tokenizer.apply_chat_template(
        row["messages"][:2], tokenize=False, add_generation_prompt=True,
    )
    supervised = token_count(rendered) - token_count(prompt)
    if supervised < 1:
        raise AssertionError("Response-only masking would leave no supervised token")


OUTPUT_SCHEMA = pa.schema([
    ("derived_id", pa.string()),
    ("parent_id", pa.string()),
    ("request_group", pa.string()),
    ("requested_sections", pa.list_(pa.string())),
    ("context_mode", pa.string()),
    ("context_text", pa.string()),
    ("target_json", pa.string()),
    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("window_index", pa.int32()),
    ("window_count", pa.int32()),
    ("n_context_tokens", pa.int32()),
    ("n_prompt_tokens", pa.int32()),
    ("n_response_tokens", pa.int32()),
    ("n_total_tokens", pa.int32()),
    ("context_sections", pa.list_(pa.string())),
    ("distractor_sections", pa.list_(pa.string())),
    ("source_sha256", pa.string()),
    ("corpus", pa.string()),
    ("annotator_model", pa.string()),
    ("source_file", pa.string()),
    ("extraction_method", pa.string()),
    ("purpose", pa.string()),
    ("split", pa.string()),
    ("split_seed", pa.int32()),
    ("audit_excluded_values", pa.int32()),
])


def build_output_row(
    source_row: dict[str, Any],
    repaired: dict[str, list[LocatedSpan]],
    group: str,
    context: str,
    mode: str,
    window_index: int,
    window_count: int,
    values: dict[str, list[str]],
    overlap_fields: set[str],
    enforce_ceiling: bool = True,
) -> dict[str, Any]:
    requested = FIELD_GROUPS[group]
    target = target_object(requested, values)
    messages, answer = make_messages(context, requested, target)
    prompt_text = text_tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True,
    )
    excluded = sum(
        max(0, len(normalize_sections(source_row["sections_json"])[field]) - len(repaired[field]))
        for field in requested
    )
    derived_id = f"{source_row['id']}::{mode}::{window_index:04d}::{group}"
    result = {
        "derived_id": derived_id,
        "parent_id": str(source_row["id"]),
        "request_group": group,
        "requested_sections": requested,
        "context_mode": mode,
        "context_text": context,
        "target_json": answer,
        "messages": messages,
        "window_index": int(window_index),
        "window_count": int(window_count),
        "n_context_tokens": token_count(context),
        "n_prompt_tokens": token_count(prompt_text),
        "n_response_tokens": token_count(answer),
        "n_total_tokens": total_chat_tokens(messages),
        "context_sections": [f for f in requested if f in overlap_fields],
        "distractor_sections": [f for f in CANONICAL_FIELDS if f not in requested and f in overlap_fields],
        "source_sha256": str(source_row["source_sha256"]),
        "corpus": str(source_row["corpus"]),
        "annotator_model": str(source_row["annotator_model"]),
        "source_file": None if source_row.get("source_file") is None else str(source_row.get("source_file")),
        "extraction_method": str(source_row.get("extraction_method", "unknown")),
        "purpose": str(source_row.get("purpose", "sft")),
        "split": str(source_row.get("split", "")),
        "split_seed": int(source_row.get("split_seed", SEED)),
        "audit_excluded_values": int(excluded),
    }
    assert_row_gates(result, enforce_ceiling=enforce_ceiling)
    return result


def rows_for_source(source_row: dict[str, Any], counters: collections.Counter) -> Iterator[dict[str, Any]]:
    text = str(source_row["input_text"])
    repaired = audit_and_relocate(source_row, counters)
    all_fields = {field for field, spans in repaired.items() if spans}

    full_rows: list[dict[str, Any]] = []
    for group, requested in FIELD_GROUPS.items():
        values = {field: [span.text for span in repaired[field]] for field in requested}
        full_rows.append(build_output_row(
            source_row, repaired, group, text, "full", 0, 1, values, all_fields,
            enforce_ceiling=False,
        ))
    if all(row["n_total_tokens"] <= BUILD_TOKEN_CEILING for row in full_rows):
        for row in full_rows:
            assert_row_gates(row)
        yield from full_rows
        return

    # Gold annotations do not participate in raw-window boundary construction.
    windows = raw_windows(text)
    for window_index, window in enumerate(windows):
        context = text[window[0]:window[1]]
        overlap_fields = overlapping_fields(repaired, window)
        for group, requested in FIELD_GROUPS.items():
            values = values_for_window(repaired, requested, text, window, windows)
            yield build_output_row(
                source_row, repaired, group, context, "window",
                window_index, len(windows), values, overlap_fields,
            )


def canonical_row_bytes(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
"""))

cells.append(markdown(r"""
## 6. CPU builder checks, deterministic build, and artifact publication

The synthetic checks exercise relocation and overlap logic only; they are not
training or a training subset. Each real split is then built twice. The second
pass must reproduce the exact row stream hash, order, counts, and token counts.
"""))

cells.append(code(r"""
def merge_with_overlap(parts: Iterable[str]) -> str:
    merged = ""
    for part in (p for p in parts if p):
        if not merged:
            merged = part
            continue
        overlap = min(len(merged), len(part))
        while overlap and not merged.endswith(part[:overlap]):
            overlap -= 1
        merged += part[overlap:]
    return merged


def run_cpu_builder_checks() -> None:
    source = "Pembuka\nTUNTUTAN   pidana\nPenutup"
    located, status = locate_verbatim(source, "TUNTUTAN pidana")
    assert status == "relocated"
    assert located is not None and located.text == "TUNTUTAN   pidana"
    assert located.text in source
    missing, missing_status = locate_verbatim(source, "tidak ada")
    assert missing is None and missing_status == "excluded"
    assert merge_with_overlap(["abcdef", "defghi", "ghi"]) == "abcdefghi"
    assert [f for group in FIELD_GROUPS.values() for f in group] == CANONICAL_FIELDS


run_cpu_builder_checks()
print("CPU builder checks passed")


def build_split(split_name: str, output_path: Path | None) -> dict[str, Any]:
    counters: collections.Counter = collections.Counter()
    digest = hashlib.sha256()
    row_count = 0
    total_tokens = 0
    modes: collections.Counter = collections.Counter()
    groups_seen: set[str] = set()
    source_shas: set[str] = set()
    batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    expected_split_value = {"train": "train", "validation": "val", "test": "test"}[split_name]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression="zstd")
    try:
        for source_row in raw[split_name]:
            source_row = dict(source_row)
            if source_row.get("purpose") not in (None, "sft"):
                raise AssertionError(f"Row outside intended SFT partition: {source_row['id']}")
            if source_row.get("split") not in (None, "", expected_split_value):
                raise AssertionError(f"Row outside intended {split_name} partition: {source_row['id']}")
            source_shas.add(str(source_row["source_sha256"]))
            for row in rows_for_source(source_row, counters):
                blob = canonical_row_bytes(row)
                digest.update(blob)
                row_count += 1
                total_tokens += row["n_total_tokens"]
                modes[row["context_mode"]] += 1
                groups_seen.add(row["request_group"])
                if writer is not None:
                    batch.append(row)
                    if len(batch) >= 128:
                        writer.write_table(pa.Table.from_pylist(batch, schema=OUTPUT_SCHEMA))
                        batch.clear()
        if writer is not None and batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=OUTPUT_SCHEMA))
            batch.clear()
    finally:
        if writer is not None:
            writer.close()

    if groups_seen != set(FIELD_GROUPS):
        raise AssertionError(f"{split_name} does not represent all 13 groups")
    return {
        "split": split_name,
        "row_count": row_count,
        "n_total_tokens": total_tokens,
        "row_stream_sha256": digest.hexdigest(),
        "context_modes": dict(sorted(modes.items())),
        "source_sha256": sorted(source_shas),
        "groups": sorted(groups_seen),
        "audit": audit_report(counters),
    }


ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
build_summaries: dict[str, dict[str, Any]] = {}
for split_name in required_splits:
    path = ARTIFACT_DIR / f"{split_name}.parquet"
    first = build_split(split_name, path)
    second = build_split(split_name, None)
    for key in ("row_count", "n_total_tokens", "row_stream_sha256", "context_modes", "groups"):
        if first[key] != second[key]:
            raise AssertionError(f"Non-deterministic {split_name} rebuild at {key}")
    build_summaries[split_name] = first
    print(split_name, {k: first[k] for k in ("row_count", "n_total_tokens", "row_stream_sha256", "context_modes")})

for i, left in enumerate(required_splits):
    for right in required_splits[i + 1:]:
        if set(build_summaries[left]["source_sha256"]) & set(build_summaries[right]["source_sha256"]):
            raise AssertionError(f"Derived split leakage: {left}/{right}")

audit_artifact = {
    "plan": "D",
    "seed": SEED,
    "model_id": MODEL_ID,
    "build_token_ceiling": BUILD_TOKEN_CEILING,
    "trainer_max_length": MAX_SEQ_LENGTH,
    "safety_margin": SAFETY_MARGIN,
    "window_context_tokens": WINDOW_CONTEXT_TOKENS,
    "window_overlap_fraction": WINDOW_OVERLAP_FRACTION,
    "field_groups": FIELD_GROUPS,
    "splits": build_summaries,
}
(ARTIFACT_DIR / "audit_report.json").write_text(
    json.dumps(audit_artifact, ensure_ascii=False, indent=2), encoding="utf-8",
)
print(f"Built dataset and audit report: {ARTIFACT_DIR}")
"""))

cells.append(markdown(r"""
## 7. Deterministic per-epoch token-budget sampling

All raw-window examples are retained. Full-input examples are capped at the
largest `G` that fits the fixed budget, using a seed-3407 document rotation.
All 13 groups must remain covered corpus-wide.
"""))

cells.append(code(r"""
train_path = ARTIFACT_DIR / "train.parquet"
metadata = pq.read_table(
    train_path,
    columns=["parent_id", "request_group", "context_mode", "n_total_tokens", "corpus"],
).to_pandas()


def rotation_start(parent_id: str, epoch: int = 0) -> int:
    payload = f"{SEED}|{epoch}|{parent_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % len(FIELD_GROUPS)


group_names = list(FIELD_GROUPS)
window_mask = metadata["context_mode"].eq("window")
window_tokens = int(metadata.loc[window_mask, "n_total_tokens"].sum())
if window_tokens > EPOCH_TOKEN_BUDGET:
    raise RuntimeError(
        f"All Tier W examples require {window_tokens:,} tokens, above the fixed "
        f"budget {EPOCH_TOKEN_BUDGET:,}; book a longer session before training."
    )

full_parent_ids = sorted(metadata.loc[~window_mask, "parent_id"].unique())


def selected_full_groups(g: int, epoch: int = 0) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    for parent_id in full_parent_ids:
        start = rotation_start(parent_id, epoch)
        selected[parent_id] = {group_names[(start + offset) % 13] for offset in range(g)}
    return selected


chosen_g = None
selected_indices: list[int] = []
selected_tokens = 0
for candidate_g in range(13, 0, -1):
    selected_by_parent = selected_full_groups(candidate_g)
    mask = window_mask | metadata.apply(
        lambda row: row["context_mode"] == "full"
        and row["request_group"] in selected_by_parent[row["parent_id"]],
        axis=1,
    )
    candidate_tokens = int(metadata.loc[mask, "n_total_tokens"].sum())
    if candidate_tokens <= EPOCH_TOKEN_BUDGET:
        chosen_g = candidate_g
        selected_indices = np.flatnonzero(mask.to_numpy()).tolist()
        selected_tokens = candidate_tokens
        break

if chosen_g is None:
    raise RuntimeError("The fixed budget cannot retain all windows plus one rotated full group per document.")

selected_meta = metadata.iloc[selected_indices]
if not selected_meta.loc[selected_meta["context_mode"].eq("window")].index.equals(metadata.loc[window_mask].index):
    raise AssertionError("Tier W sampling dropped or reordered a window row")
if set(selected_meta["request_group"]) != set(FIELD_GROUPS):
    raise AssertionError("Budgeted epoch does not cover every semantic group")

all_artifacts = DatasetDict({
    "train": Dataset.from_parquet(str(ARTIFACT_DIR / "train.parquet")),
    "validation": Dataset.from_parquet(str(ARTIFACT_DIR / "validation.parquet")),
    "test": Dataset.from_parquet(str(ARTIFACT_DIR / "test.parquet")),
})
train_dataset = all_artifacts["train"].select(selected_indices)
validation_dataset = all_artifacts["validation"]
test_dataset = all_artifacts["test"]

budget_report = {
    "epoch": 0,
    "epoch_token_budget": EPOCH_TOKEN_BUDGET,
    "all_window_tokens": window_tokens,
    "chosen_full_groups_per_document": chosen_g,
    "selected_rows": len(selected_indices),
    "selected_n_total_tokens": selected_tokens,
    "selected_group_counts": selected_meta["request_group"].value_counts().sort_index().to_dict(),
}
(ARTIFACT_DIR / "epoch_budget.json").write_text(
    json.dumps(budget_report, indent=2), encoding="utf-8",
)
print(json.dumps(budget_report, indent=2))
"""))

cells.append(markdown(r"""
## 8. Credentials, GPU inventory, 4-bit model, and adapter audit

The W&B key is read only from Colab Secrets or the environment. GPU checks are
diagnostic and intentionally contain no blocking GPU assertion. Loading the
real model starts the sole GPU run path.
"""))

cells.append(code(r"""
try:
    from google.colab import userdata
    wandb_key = userdata.get("WANDB_API_KEY")
except Exception:
    wandb_key = os.environ.get("WANDB_API_KEY")

REPORT_TO = "none"
if wandb_key:
    import wandb
    wandb.login(key=wandb_key, relogin=True)
    REPORT_TO = "wandb"
else:
    print("WANDB_API_KEY not found in Colab Secrets/environment; local artifacts remain enabled.")

gpu_inventory = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    gpu_inventory.update({
        "name": properties.name,
        "total_memory_gib": properties.total_memory / 2**30,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    })
print(json.dumps(gpu_inventory, indent=2))

from unsloth import FastLanguageModel

model, processor = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    use_gradient_checkpointing="unsloth",
)
text_tokenizer = getattr(processor, "tokenizer", processor)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

adapter_parents = sorted({
    name.split(".lora_A", 1)[0]
    for name, _ in model.named_parameters()
    if ".lora_A" in name
})
families = {
    "full_attention": [n for n in adapter_parents if any(x in n for x in ("q_proj", "k_proj", "v_proj", "o_proj"))],
    "linear_attention": [n for n in adapter_parents if any(x in n for x in ("in_proj_qkvz", "in_proj_ba", "out_proj"))],
    "mlp": [n for n in adapter_parents if any(x in n for x in ("gate_proj", "up_proj", "down_proj"))],
}
print("Resolved adapter parent modules:")
for name in adapter_parents:
    print(" ", name)
for family, names in families.items():
    if not names:
        raise AssertionError(f"LoRA adapter audit failed: no {family} coverage")
    print(f"{family}: {len(names)} adapters")
"""))

cells.append(markdown(r"""
## 9. Response-only trainer and zero-drop masking gate

Validation and test are both attached to the trainer; best-model selection is
explicitly tied to validation loss only.
"""))

cells.append(code(r"""
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

FastLanguageModel.for_training(model)


def add_rendered_text(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
    return {
        "text": [
            text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch["messages"]
        ]
    }


def trainer_view(dataset: Dataset) -> Dataset:
    return dataset.map(
        add_rendered_text,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Render exact Plan D chat template",
    )


train_for_trainer = trainer_view(train_dataset)
eval_for_trainer = DatasetDict({
    "validation": trainer_view(validation_dataset),
    "test": trainer_view(test_dataset),
})

trainer = SFTTrainer(
    model=model,
    tokenizer=text_tokenizer,
    train_dataset=train_for_trainer,
    eval_dataset=eval_for_trainer,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_loss",
        greater_is_better=False,
        seed=SEED,
        data_seed=SEED,
        bf16=True,
        fp16=False,
        output_dir=str(ARTIFACT_DIR / "checkpoints"),
        report_to=REPORT_TO,
        run_name="plan-d-qwen3.5-9b",
    ),
)

before_masking = {
    "train": len(trainer.train_dataset),
    **{name: len(ds) for name, ds in trainer.eval_dataset.items()},
}
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)


def assert_zero_masked_rows(name: str, dataset: Dataset) -> None:
    if "labels" not in dataset.column_names:
        raise AssertionError(f"{name}: response masker did not materialize labels")
    bad = sum(not any(token != -100 for token in labels) for labels in dataset["labels"])
    if bad:
        raise AssertionError(f"{name}: response masking left {bad} rows without supervision")


assert len(trainer.train_dataset) == before_masking["train"]
assert_zero_masked_rows("train", trainer.train_dataset)
for name, dataset in trainer.eval_dataset.items():
    assert len(dataset) == before_masking[name]
    assert_zero_masked_rows(name, dataset)
print("Response-only masking dropped zero rows and left supervision in every row.")
"""))

cells.append(markdown(r"""
## 10. One real, resumable training run

The first normal log at step 10 reports measured effective tokens/second. If
the projected run exceeds the configured session, training continues to the
next checkpoint boundary, saves, and stops for deterministic resume. Set
`PLAN_D_RESUME` to that checkpoint path in the next session.
"""))

cells.append(code(r"""
from transformers import TrainerCallback

SAVE_STEPS = 100
estimated_update_steps = max(1, math.ceil(len(train_dataset) / 16))


class FirstLogBudgetCallback(TrainerCallback):
    def __init__(self, total_tokens: int, total_steps: int):
        self.total_tokens = total_tokens
        self.total_steps = total_steps
        self.started = None
        self.start_global_step = 0
        self.stop_at_step = None
        self.first_measurement = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        self.start_global_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.global_step or self.started is None:
            return control
        elapsed = max(1e-9, time.perf_counter() - self.started)
        processed_this_session = self.total_tokens * min(
            1.0,
            max(0, state.global_step - self.start_global_step) / self.total_steps,
        )
        processed_total = self.total_tokens * min(1.0, state.global_step / self.total_steps)
        measured_tps = processed_this_session / elapsed
        remaining_tokens = max(0.0, self.total_tokens - processed_total)
        projected_remaining_seconds = remaining_tokens / max(measured_tps, 1e-9)
        session_remaining_seconds = max(
            0.0,
            SESSION_WALLCLOCK_HOURS * 3600 - (time.perf_counter() - RUN_STARTED_AT),
        )
        measurement = {
            "global_step": state.global_step,
            "effective_tokens_per_second": measured_tps,
            "projected_remaining_hours": projected_remaining_seconds / 3600,
            "session_remaining_hours": session_remaining_seconds / 3600,
        }
        print("REAL-RUN THROUGHPUT", json.dumps(measurement, indent=2))
        if self.first_measurement is None:
            self.first_measurement = measurement
            (ARTIFACT_DIR / "first_log_throughput.json").write_text(
                json.dumps(measurement, indent=2), encoding="utf-8",
            )
            if projected_remaining_seconds > session_remaining_seconds:
                self.stop_at_step = int((state.global_step // SAVE_STEPS + 1) * SAVE_STEPS)
                print(f"Projection exceeds session; will save and stop at checkpoint step {self.stop_at_step}.")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_at_step is not None and state.global_step >= self.stop_at_step:
            control.should_save = True
            control.should_training_stop = True
        return control


budget_callback = FirstLogBudgetCallback(selected_tokens, estimated_update_steps)
trainer.add_callback(budget_callback)

resume_from_checkpoint = os.environ.get("PLAN_D_RESUME") or None
try:
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
except RuntimeError as exc:
    if "out of memory" not in str(exc).lower():
        raise
    next_ceiling = BUILD_TOKEN_CEILING - 1_024
    oom_report = {
        "failed_build_ceiling": BUILD_TOKEN_CEILING,
        "next_build_ceiling": next_ceiling,
        "action": (
            "Restart the real run from the top with PLAN_D_BUILD_CEILING set to "
            f"{next_ceiling}; rebuild the affected raw-window rows and do not truncate. "
            "Do not resume a checkpoint built with the old ceiling."
        ),
    }
    (ARTIFACT_DIR / "oom_rebuild_required.json").write_text(
        json.dumps(oom_report, indent=2), encoding="utf-8",
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise RuntimeError(json.dumps(oom_report, indent=2)) from exc
training_complete = bool(trainer.state.epoch is not None and trainer.state.epoch >= 0.999)
run_report = {
    "resume_from_checkpoint": resume_from_checkpoint,
    "global_step": trainer.state.global_step,
    "epoch": trainer.state.epoch,
    "training_complete": training_complete,
    "metrics": trainer_stats.metrics,
    "first_log_throughput": budget_callback.first_measurement,
}
(ARTIFACT_DIR / "training_report.json").write_text(
    json.dumps(run_report, indent=2, default=str), encoding="utf-8",
)
if not training_complete:
    raise RuntimeError(
        "The real run stopped at a checkpoint boundary. Resume from the latest checkpoint "
        "with PLAN_D_RESUME; evaluation begins only after the budgeted epoch completes."
    )
"""))

cells.append(markdown(r"""
## 11. Deterministic test inference and merged-window evaluation

Overflow documents are inferred window by window, then merged in document
order with exact suffix/prefix overlap deduplication. Headline field metrics
are computed only on this deployment condition. Every `source_sha256` receives
equal aggregate weight.
"""))

cells.append(code(r"""
from rapidfuzz.distance import LCSseq

FastLanguageModel.for_inference(model)


def parse_prediction(text: str) -> tuple[dict[str, Any] | None, bool]:
    candidate = text.strip()
    try:
        value = json.loads(candidate)
        return (value, isinstance(value, dict))
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(candidate[start:end + 1])
                return (value, isinstance(value, dict))
            except json.JSONDecodeError:
                pass
    return None, False


def normalized_predicted_sections(parsed: dict[str, Any] | None) -> dict[str, list[str]]:
    if not parsed or not isinstance(parsed.get("sections"), dict):
        return {}
    result: dict[str, list[str]] = {}
    for key, value in parsed["sections"].items():
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            result[str(key)] = [str(item) for item in value]
    return result


prediction_path = ARTIFACT_DIR / "test_predictions.jsonl"
prediction_records: list[dict[str, Any]] = []
with prediction_path.open("w", encoding="utf-8") as output:
    for index, row in enumerate(test_dataset):
        prompt_messages = row["messages"][:2]
        prompt_ids = text_tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        available = MAX_SEQ_LENGTH - int(prompt_ids.shape[-1])
        max_new_tokens = min(int(row["n_response_tokens"]) + SAFETY_MARGIN, available)
        if max_new_tokens < 1:
            raise AssertionError(f"No generation budget for test row {row['derived_id']}")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=text_tokenizer.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id or text_tokenizer.eos_token_id,
            )
        decoded = text_tokenizer.decode(generated[0, prompt_ids.shape[-1]:], skip_special_tokens=True)
        parsed, valid = parse_prediction(decoded)
        predicted_sections = normalized_predicted_sections(parsed)
        gold = json.loads(row["target_json"])
        predicted_keys = list(predicted_sections)
        requested = list(row["requested_sections"])
        record = {
            "derived_id": row["derived_id"],
            "parent_id": row["parent_id"],
            "source_sha256": row["source_sha256"],
            "corpus": row["corpus"],
            "annotator_model": row["annotator_model"],
            "request_group": row["request_group"],
            "requested_sections": requested,
            "context_mode": row["context_mode"],
            "window_index": row["window_index"],
            "window_count": row["window_count"],
            "n_total_tokens": row["n_total_tokens"],
            "json_valid": valid,
            "exact_requested_key_set": set(predicted_keys) == set(requested),
            "extra_keys": sorted(set(predicted_keys) - set(requested)),
            "missing_keys": sorted(set(requested) - set(predicted_keys)),
            "gold_sections": gold["sections"],
            "predicted_sections": predicted_sections,
            "field_verbatim_checks": {
                field: [value in row["context_text"] for value in predicted_sections.get(field, [])]
                for field in requested
            },
            "raw_generation": decoded,
        }
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        prediction_records.append(record)
        if (index + 1) % 100 == 0:
            print(f"inferred {index + 1}/{len(test_dataset)}")

print(f"Saved {len(prediction_records)} deterministic test generations to {prediction_path}")
"""))

cells.append(code(r"""
def char_prf(prediction: str, reference: str) -> tuple[float, float, float, int]:
    matches = int(LCSseq.similarity(prediction, reference))
    precision = matches / len(prediction) if prediction else float(not reference)
    recall = matches / len(reference) if reference else float(not prediction)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, matches


def token_bucket(n: int) -> str:
    if n < 4096: return "<4K"
    if n < 8192: return "4-8K"
    if n < 16384: return "8-16K"
    if n < 24576: return "16-24K"
    return "24-32K"


by_request: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
for record in prediction_records:
    by_request[(record["parent_id"], record["request_group"])].append(record)

field_rows: list[dict[str, Any]] = []
request_rows: list[dict[str, Any]] = []
for (_, _), records in sorted(by_request.items()):
    records.sort(key=lambda item: item["window_index"])
    first = records[0]
    request_rows.append({
        "source_sha256": first["source_sha256"],
        "corpus": first["corpus"],
        "annotator_model": first["annotator_model"],
        "request_group": first["request_group"],
        "context_mode": first["context_mode"],
        "json_valid": float(all(item["json_valid"] for item in records)),
        "exact_requested_key_set": float(all(item["exact_requested_key_set"] for item in records)),
        "extra_key": float(any(item["extra_keys"] for item in records)),
        "missing_key": float(any(item["missing_keys"] for item in records)),
    })
    for field in first["requested_sections"]:
        gold_parts = [value for item in records for value in item["gold_sections"].get(field, [])]
        pred_parts = [value for item in records for value in item["predicted_sections"].get(field, [])]
        gold_text = merge_with_overlap(gold_parts)
        pred_text = merge_with_overlap(pred_parts)
        precision, recall, f1, matches = char_prf(pred_text, gold_text)
        verbatim_checks = [
            check
            for item in records
            for check in item["field_verbatim_checks"][field]
        ]
        field_rows.append({
            "source_sha256": first["source_sha256"],
            "corpus": first["corpus"],
            "annotator_model": first["annotator_model"],
            "request_group": first["request_group"],
            "field": field,
            "context_mode": first["context_mode"],
            "token_bucket": token_bucket(max(item["n_total_tokens"] for item in records)),
            "requested_field_count": len(first["requested_sections"]),
            "empty_status": "empty" if not gold_text else "non_empty",
            "char_precision": precision,
            "char_recall": recall,
            "char_f1": f1,
            "exact_field_match": float(pred_text == gold_text),
            "generated_verbatim": float(sum(verbatim_checks) / len(verbatim_checks)) if verbatim_checks else 1.0,
            "predicted_empty": not bool(pred_text),
            "gold_empty": not bool(gold_text),
            "matching_chars": matches,
            "predicted_chars": len(pred_text),
            "gold_chars": len(gold_text),
        })

field_df = pd.DataFrame(field_rows)
request_df = pd.DataFrame(request_rows)


def source_weighted_mean(frame: pd.DataFrame, metric: str) -> float:
    return float(frame.groupby("source_sha256", sort=True)[metric].mean().mean())


def source_weighted_breakdown(frame: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    metrics = ["char_precision", "char_recall", "char_f1", "exact_field_match", "generated_verbatim"]
    per_source = frame.groupby(dimensions + ["source_sha256"], dropna=False, sort=True)[metrics].mean().reset_index()
    return per_source.groupby(dimensions, dropna=False, sort=True)[metrics].mean().reset_index()


json_valid_rate = source_weighted_mean(request_df, "json_valid")
key_accuracy = source_weighted_mean(request_df, "exact_requested_key_set")
extra_key_rate = source_weighted_mean(request_df, "extra_key")
missing_key_rate = source_weighted_mean(request_df, "missing_key")
verbatim_rate = source_weighted_mean(field_df, "generated_verbatim")

source_scores = []
for source_sha256, source_frame in field_df.groupby("source_sha256", sort=True):
    empty_tp = int(((source_frame.gold_empty) & (source_frame.predicted_empty)).sum())
    empty_fp = int(((~source_frame.gold_empty) & (source_frame.predicted_empty)).sum())
    empty_fn = int(((source_frame.gold_empty) & (~source_frame.predicted_empty)).sum())
    empty_precision = empty_tp / (empty_tp + empty_fp) if empty_tp + empty_fp else 1.0
    empty_recall = empty_tp / (empty_tp + empty_fn) if empty_tp + empty_fn else 1.0
    empty_f1 = 2 * empty_precision * empty_recall / (empty_precision + empty_recall) if empty_precision + empty_recall else 0.0
    matches = int(source_frame["matching_chars"].sum())
    pred_chars = int(source_frame["predicted_chars"].sum())
    gold_chars = int(source_frame["gold_chars"].sum())
    micro_precision = matches / pred_chars if pred_chars else float(gold_chars == 0)
    micro_recall = matches / gold_chars if gold_chars else float(pred_chars == 0)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    source_scores.append({
        "source_sha256": source_sha256,
        "empty_precision": empty_precision,
        "empty_recall": empty_recall,
        "empty_f1": empty_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    })
source_score_df = pd.DataFrame(source_scores)
empty_precision = float(source_score_df["empty_precision"].mean())
empty_recall = float(source_score_df["empty_recall"].mean())
empty_f1 = float(source_score_df["empty_f1"].mean())
micro_precision = float(source_score_df["micro_precision"].mean())
micro_recall = float(source_score_df["micro_recall"].mean())
micro_f1 = float(source_score_df["micro_f1"].mean())

breakdown_dimensions = [
    ["corpus"], ["request_group"], ["field"], ["context_mode"],
    ["token_bucket"], ["requested_field_count"], ["empty_status"], ["annotator_model"],
]
breakdowns: dict[str, list[dict[str, Any]]] = {}
for dimensions in breakdown_dimensions:
    name = "__".join(dimensions)
    table = source_weighted_breakdown(field_df, dimensions)
    table.to_csv(ARTIFACT_DIR / f"breakdown_{name}.csv", index=False)
    breakdowns[name] = table.to_dict(orient="records")

per_field = source_weighted_breakdown(field_df, ["field"])
global_syntax_pass = (
    json_valid_rate >= 0.99
    and key_accuracy >= 0.995
    and extra_key_rate <= 0.005
    and verbatim_rate >= 0.99
)
field_decisions = []
for item in per_field.to_dict(orient="records"):
    ship = global_syntax_pass and item["generated_verbatim"] >= 0.99
    field_decisions.append({
        **item,
        "decision": "ship" if ship else "iterate_window_size_or_overlap",
        "decision_basis": "published syntax thresholds plus per-field 99% verbatim target",
    })

evaluation_report = {
    "condition": "merged_raw_window",
    "source_weighting": "each source_sha256 weighted equally",
    "headline": {
        "json_valid_rate": json_valid_rate,
        "exact_requested_key_set_accuracy": key_accuracy,
        "extra_key_rate": extra_key_rate,
        "missing_requested_key_rate": missing_key_rate,
        "generated_string_verbatim_rate": verbatim_rate,
        "source_weighted_macro_field_char_f1": source_weighted_mean(field_df, "char_f1"),
        "source_weighted_exact_field_match": source_weighted_mean(field_df, "exact_field_match"),
        "empty_field_precision": empty_precision,
        "empty_field_recall": empty_recall,
        "empty_field_f1": empty_f1,
        "micro_character_precision": micro_precision,
        "micro_character_recall": micro_recall,
        "micro_character_f1": micro_f1,
        "merged_field_f1": source_weighted_mean(field_df, "char_f1"),
    },
    "acceptance_targets": {
        "json_valid_rate_gte": 0.99,
        "requested_key_set_accuracy_gte": 0.995,
        "extra_key_rate_lte": 0.005,
        "generated_string_verbatim_rate_gte": 0.99,
        "global_syntax_pass": global_syntax_pass,
    },
    "field_decisions": field_decisions,
    "breakdowns": breakdowns,
    "verbatim_audit_and_exclusions": audit_artifact["splits"]["test"]["audit"],
    "baselines": {
        "B0": "documented by the existing failed run; not rerun",
        "B1": "priority comparison when a separately budgeted real run is available",
        "D2": "oracle gold-aggregated upper bound only; not part of this implementation",
        "D3": "optional windows-only ablation",
    },
}
(ARTIFACT_DIR / "evaluation_report.json").write_text(
    json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8",
)
print(json.dumps(evaluation_report["headline"], indent=2))
print(f"Evaluation artifacts written to {ARTIFACT_DIR}")
"""))

cells.append(markdown(r"""
## 12. Save the adapter and reproducibility manifest

The manifest records the exact Plan D configuration, build hashes, budget,
training state, and evaluation location. There is no merged-model export or
RAG/GRPO serving path in this notebook.
"""))

cells.append(code(r"""
ADAPTER_DIR = ARTIFACT_DIR / "qwen3_5_9b_plan_d_lora"
model.save_pretrained(str(ADAPTER_DIR))
processor.save_pretrained(str(ADAPTER_DIR))

manifest = {
    "plan": "D",
    "model_id": MODEL_ID,
    "adapter_dir": str(ADAPTER_DIR),
    "seed": SEED,
    "canonical_fields": CANONICAL_FIELDS,
    "field_groups": FIELD_GROUPS,
    "system_prompt": SYSTEM_PROMPT,
    "user_order": "context_first_requested_fields_last",
    "build": {
        "ceiling": BUILD_TOKEN_CEILING,
        "trainer_limit": MAX_SEQ_LENGTH,
        "safety_margin": SAFETY_MARGIN,
        "window_context_tokens": WINDOW_CONTEXT_TOKENS,
        "window_overlap_fraction": WINDOW_OVERLAP_FRACTION,
        "split_summaries": {
            split: {k: v for k, v in summary.items() if k != "source_sha256"}
            for split, summary in build_summaries.items()
        },
    },
    "epoch_budget": budget_report,
    "trainer": {
        "micro_batch": 1,
        "gradient_accumulation": 16,
        "epochs": 1,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "optimizer": "adamw_8bit",
        "weight_decay": 0.001,
        "schedule": "linear",
        "max_grad_norm": 1.0,
        "packing": False,
        "response_only": True,
        "eval_splits": ["validation", "test"],
        "best_model_split": "validation",
    },
    "lora": {
        "r": 32, "alpha": 32, "dropout": 0, "bias": "none",
        "vision": False, "language": True, "attention": True, "mlp": True,
        "resolved_parent_modules": adapter_parents,
    },
    "training_report": run_report,
    "evaluation_report": str(ARTIFACT_DIR / "evaluation_report.json"),
}
(ARTIFACT_DIR / "reproducibility_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
)
print(f"Adapter: {ADAPTER_DIR}")
print(f"Manifest: {ARTIFACT_DIR / 'reproducibility_manifest.json'}")
"""))


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
