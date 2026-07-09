#!/usr/bin/env python3
"""Plan B — windowed line-anchored dataset builder.

Reads the legacy Parquet datasets under ``notebooks/dataset/{sft,grpo,rag}``
(4,407 document rows with their existing leakage-safe purpose/split
assignment) and rewrites every document as a series of line-numbered windows
whose supervision target is a compact JSON of GLOBAL line ranges per section,
instead of the full verbatim-span JSON.

Why: the whole-document targets exceed the 32,768-token context for >50% of
examples (see notebooks/datalog.md), so the assistant target is truncated away
and most of the dataset contributes zero learning signal. Windowed rows are
<=8K tokens each, so 100% of the data becomes usable on an A100-40GB.

Per window the model sees::

    00042|Menimbang, bahwa Terdakwa ...

and must emit::

    {"sections": {"dakwaan": [[120, 187]], ...},
     "sections_absent": ["ahli", ...]}

A deterministic assembler (``assemble_document`` below) maps predicted line
ranges back to verbatim text and merges across windows — hallucination-free by
construction. The builder round-trips every document through the assembler on
the GOLD window targets and asserts the original sections are reproduced
exactly; this is the correctness gate for the whole pipeline.

Usage:
    python notebooks/build_windowed_dataset.py --repo-root .
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Must match build_dataset.py exactly (31 keys, canonical order).
CANONICAL_SECTIONS: list[str] = [
    "judul", "nomor_putusan", "irah_irah", "nama_pengadilan_negeri",
    "keterangan_perkara", "nama_lengkap", "tempat_lahir", "umur_tanggal_lahir",
    "jenis_kelamin", "kebangsaan", "tempat_tinggal", "agama", "pekerjaan",
    "penangkapan", "penahanan", "tuntutan", "dakwaan", "saksi", "ahli",
    "terdakwa", "surat", "petunjuk_barang_bukti", "fakta_hukum",
    "pertimbangan_hukum", "amar_putusan", "hari", "tanggal", "tahun",
    "siapa_yang_memutus", "panitera_pengganti", "tanda_tangan_majelis",
]
assert len(CANONICAL_SECTIONS) == 31

PURPOSES = ("sft", "grpo", "rag")
SPLITS = ("train", "val", "test")

# Window geometry (characters of line-numbered content). ~22K chars is roughly
# 6-7K tokens for Indonesian legal text, leaving room for the system prompt
# and the short JSON target inside a max_seq_length of 8192.
WINDOW_CHARS = 22_000
OVERLAP_FRACTION = 0.15
MIN_OVERLAP_LINES = 2  # guarantees window-clipped pieces of one span overlap

LINE_NO_WIDTH = 5  # 00001| ... supports docs up to 99,999 lines

SYSTEM_PROMPT = (
    "Anda adalah pengekstrak terstruktur putusan pengadilan Indonesia. Anda "
    "menerima SATU POTONGAN (window) dari teks putusan; setiap baris diawali "
    "nomor baris global berformat 'NNNNN|'. Keluarkan SATU objek JSON dengan "
    "tepat dua kunci: 'sections' dan 'sections_absent'. Di 'sections', untuk "
    "setiap bagian kanonik yang muncul dalam potongan ini, berikan daftar "
    "rentang baris inklusif [[awal, akhir], ...] memakai nomor baris global "
    "persis seperti yang tercetak — jangan menyalin teks putusan, jangan "
    "menghitung, cukup salin nomor baris yang terlihat. Semua bagian kanonik "
    "lain dicantumkan sebagai daftar nama di 'sections_absent'. Kunci bagian, "
    "dalam urutan kanonik: " + ", ".join(CANONICAL_SECTIONS) + "."
)


# --------------------------------------------------------------------------
# Reconstruction replay: recover exact char offsets of every gold span
# --------------------------------------------------------------------------

def replay_span_offsets(
    sections: dict[str, list[str]], input_text: str
) -> dict[str, list[tuple[int, int]]]:
    """Replay build_dataset.reconstruct_input to get each span's char range.

    reconstruct_input joins non-empty spans with "\\n" inside a section and
    sections with "\\n\\n", so offsets are fully determined; each is asserted
    against input_text.
    """
    offsets: dict[str, list[tuple[int, int]]] = {k: [] for k in CANONICAL_SECTIONS}
    pos = 0
    first_block = True
    for key in CANONICAL_SECTIONS:
        spans = [s for s in sections.get(key, []) if s.strip()]
        if not spans:
            continue
        if not first_block:
            assert input_text[pos:pos + 2] == "\n\n", f"block sep mismatch at {pos}"
            pos += 2
        first_block = False
        for j, span in enumerate(spans):
            if j:
                assert input_text[pos] == "\n", f"span sep mismatch at {pos}"
                pos += 1
            end = pos + len(span)
            assert input_text[pos:end] == span, (
                f"replay mismatch: section={key} at {pos}"
            )
            offsets[key].append((pos, end))
            pos = end
    assert pos == len(input_text), f"replay did not consume input ({pos} != {len(input_text)})"
    return offsets


def char_to_line_ranges(
    offsets: dict[str, list[tuple[int, int]]], input_text: str
) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    """Convert char ranges to 1-based inclusive global line ranges.

    Every span starts at a line start and ends at a line end by construction
    (spans are joined with newlines), which is asserted.
    """
    lines = input_text.split("\n")
    line_start_of: dict[int, int] = {}
    line_end_of: dict[int, int] = {}
    start_to_line: dict[int, int] = {}
    end_to_line: dict[int, int] = {}
    pos = 0
    for i, line in enumerate(lines, start=1):
        line_start_of[i] = pos
        line_end_of[i] = pos + len(line)
        start_to_line[pos] = i
        end_to_line[pos + len(line)] = i
        pos += len(line) + 1  # skip the "\n"

    line_ranges: dict[str, list[tuple[int, int]]] = {}
    for key, ranges in offsets.items():
        if not ranges:
            continue
        out = []
        for s, e in ranges:
            assert s in start_to_line, f"span does not start at a line start: {key}"
            assert e in end_to_line, f"span does not end at a line end: {key}"
            out.append((start_to_line[s], end_to_line[e]))
        line_ranges[key] = out
    return lines, line_ranges


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

def numbered_line(i: int, line: str) -> str:
    return f"{i:0{LINE_NO_WIDTH}d}|{line}"


def window_lines(lines: list[str]) -> list[tuple[int, int]]:
    """Greedy char-budget packing of whole lines into overlapping windows.

    Returns 1-based inclusive (first_line, last_line) per window. Consecutive
    windows overlap by ~OVERLAP_FRACTION of chars (>= MIN_OVERLAP_LINES lines)
    so any span clipped at a boundary yields OVERLAPPING pieces that the
    assembler can merge back into one range.
    """
    n = len(lines)
    cost = [len(lines[i]) + LINE_NO_WIDTH + 2 for i in range(n)]  # +"|" +"\n"
    windows: list[tuple[int, int]] = []
    a = 0  # 0-based first line of current window
    while a < n:
        total = 0
        b = a
        while b < n and (total + cost[b] <= WINDOW_CHARS or b == a):
            total += cost[b]
            b += 1
        # window covers lines [a, b-1]
        windows.append((a + 1, b))
        if b >= n:
            break
        # walk back from b to build the overlap for the next window
        overlap_target = OVERLAP_FRACTION * total
        back = 0
        acc = 0
        while (b - 1 - back) > a and (
            acc < overlap_target or back < MIN_OVERLAP_LINES
        ):
            acc += cost[b - 1 - back]
            back += 1
        a = max(a + 1, b - back)
    return windows


def clip_ranges(
    line_ranges: dict[str, list[tuple[int, int]]], wa: int, wb: int
) -> dict[str, list[list[int]]]:
    """Intersect every gold line range with window [wa, wb] (inclusive)."""
    out: dict[str, list[list[int]]] = {}
    for key in CANONICAL_SECTIONS:
        ranges = line_ranges.get(key)
        if not ranges:
            continue
        clipped = [
            [max(s, wa), min(e, wb)] for s, e in ranges if s <= wb and e >= wa
        ]
        if clipped:
            out[key] = clipped
    return out


# --------------------------------------------------------------------------
# Deterministic assembler (also used at inference time)
# --------------------------------------------------------------------------

def merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    """Sort and merge OVERLAPPING (not merely adjacent) inclusive ranges.

    Adjacent-but-distinct gold spans must stay separate; pieces of one span
    clipped by different windows always share lines (window overlap >= 2
    lines), so overlap-only merging reassembles spans exactly.
    """
    if not ranges:
        return []
    ranges = sorted([list(r) for r in ranges])
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def assemble_document(
    window_targets: list[dict[str, Any]], input_text: str
) -> dict[str, list[str]]:
    """Reduce per-window line-range predictions to the 31-section span dict.

    ``window_targets`` are parsed JSON objects ({"sections": ..., ...}).
    Invalid ranges are dropped. Returns {section: [verbatim span, ...]}.
    """
    lines = input_text.split("\n")
    n = len(lines)
    collected: dict[str, list[list[int]]] = {k: [] for k in CANONICAL_SECTIONS}
    for tgt in window_targets:
        secs = tgt.get("sections") or {}
        if not isinstance(secs, dict):
            continue
        for key, ranges in secs.items():
            if key not in collected or not isinstance(ranges, list):
                continue
            for r in ranges:
                if (
                    isinstance(r, (list, tuple)) and len(r) == 2
                    and all(isinstance(x, int) for x in r)
                    and 1 <= r[0] <= r[1] <= n
                ):
                    collected[key].append([r[0], r[1]])
    out: dict[str, list[str]] = {}
    for key in CANONICAL_SECTIONS:
        merged = merge_ranges(collected[key])
        out[key] = ["\n".join(lines[s - 1:e]) for s, e in merged]
    return out


# --------------------------------------------------------------------------
# Row building
# --------------------------------------------------------------------------

def build_windows_for_row(row: pd.Series) -> list[dict[str, Any]]:
    input_text: str = row["input_text"]
    sections: dict[str, list[str]] = json.loads(row["sections_json"])

    offsets = replay_span_offsets(sections, input_text)
    lines, line_ranges = char_to_line_ranges(offsets, input_text)
    windows = window_lines(lines)
    n_windows = len(windows)

    out_rows: list[dict[str, Any]] = []
    gold_targets: list[dict[str, Any]] = []
    for w_idx, (wa, wb) in enumerate(windows):
        present = clip_ranges(line_ranges, wa, wb)
        absent = [k for k in CANONICAL_SECTIONS if k not in present]
        target = {"sections": present, "sections_absent": absent}
        gold_targets.append(target)
        target_json = json.dumps(target, ensure_ascii=False)
        window_text = "\n".join(
            numbered_line(i, lines[i - 1]) for i in range(wa, wb + 1)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": window_text},
            {"role": "assistant", "content": target_json},
        ]
        out_rows.append({
            # identity & provenance (doc-level fields carried through)
            "id": f"{row['id']}#w{w_idx:03d}",
            "doc_id": row["id"],
            "corpus": row["corpus"],
            "annotator_model": row["annotator_model"],
            "source_file": row["source_file"],
            "source_sha256": row["source_sha256"],
            "extraction_method": row["extraction_method"],
            # split bookkeeping — identical to the legacy dataset
            "purpose": row["purpose"],
            "split": row["split"],
            "split_seed": row["split_seed"],
            # window geometry
            "window_index": w_idx,
            "n_windows": n_windows,
            "line_start": wa,
            "line_end": wb,
            "n_doc_lines": len(lines),
            # payload
            "input_text": window_text,
            "target_json": target_json,
            "messages": messages,
            "prompt": messages[:2],
            "answer": target_json,
            # statistics
            "n_sections_present": len(present),
            "n_input_chars": len(window_text),
            "n_target_chars": len(target_json),
        })

    # ---- round-trip correctness gate: gold windows -> original sections ----
    assembled = assemble_document(gold_targets, input_text)
    for key in CANONICAL_SECTIONS:
        gold_spans = [s for s in sections.get(key, []) if s.strip()]
        if assembled[key] != gold_spans:
            raise AssertionError(
                f"round-trip failed: id={row['id']} section={key} "
                f"gold={len(gold_spans)} spans, assembled={len(assembled[key])}"
            )
    return out_rows


def write_readme(out_dir: Path, info: dict[str, Any]) -> None:
    yaml_configs = ["configs:"]
    for i, p in enumerate(PURPOSES):
        yaml_configs.append(f"- config_name: {p}")
        if i == 0:
            yaml_configs.append("  default: true")
        yaml_configs.append("  data_files:")
        for split, fname in (("train", "train"), ("validation", "val"), ("test", "test")):
            yaml_configs.append(f"  - split: {split}")
            yaml_configs.append(f"    path: {p}/{fname}.parquet")
    header = "\n".join([
        "---",
        "language:",
        "- id",
        "license: cc-by-4.0",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- legal",
        "- indonesian",
        "- court-decisions",
        "- structured-extraction",
        "- long-documents",
        *yaml_configs,
        "---",
    ])
    body = [
        header,
        "",
        "# Putusan windowed line-anchored extraction dataset (Plan B)",
        "",
        f"Built {info['built_at']} by `notebooks/build_windowed_dataset.py` from the",
        "legacy `Haeryz/putusan-structured-extraction` dataset (same documents, same",
        "leakage-safe purpose/split assignment, seed 3407).",
        "",
        "Each legacy document row (~34K tokens median — longer than a 32K context)",
        "is re-expressed as overlapping line-numbered windows of ~22K characters",
        "(~6-7K tokens). The supervision target per window is a compact JSON of",
        "GLOBAL inclusive line ranges per canonical section:",
        "",
        "```json",
        '{"sections": {"dakwaan": [[120, 187]]}, "sections_absent": ["ahli", "..."]}',
        "```",
        "",
        "The model copies line numbers it can see (never counts, never quotes",
        "document text); a deterministic assembler slices verbatim text from the",
        "ranges and merges overlapping ranges across windows, reproducing the legacy",
        "31-section `target_json` format exactly (asserted at build time by a gold",
        "round-trip for every document). Hallucination-free by construction.",
        "",
        "## Schema (windowed rows)",
        "",
        "| column | type | description |",
        "|---|---|---|",
        "| id | str | `legacy_id#wNNN` |",
        "| doc_id | str | legacy row id (`corpus/annotator_model/sha16`) |",
        "| corpus / annotator_model / source_file / source_sha256 | str | provenance |",
        "| purpose / split / split_seed | str/int | identical to legacy dataset |",
        "| window_index / n_windows | int | window position within document |",
        "| line_start / line_end / n_doc_lines | int | global 1-based inclusive line span of this window |",
        "| input_text | str | line-numbered window text (`NNNNN|content`) |",
        "| target_json | str | window target: sections -> [[start, end], ...] + sections_absent |",
        "| messages | list | system/user/assistant conversation (SFT) |",
        "| prompt | list | system+user (GRPO rollout) |",
        "| answer | str | = target_json (GRPO reference) |",
        "| n_sections_present / n_input_chars / n_target_chars | int | statistics |",
        "",
        "## Row counts (windows; documents in parentheses)",
        "",
        "| purpose | train | val | test |",
        "|---|---|---|---|",
    ]
    for p in PURPOSES:
        cells = []
        for s in SPLITS:
            c = info["row_counts"][p][s]
            d = info["doc_counts"][p][s]
            cells.append(f"{c} ({d})")
        body.append(f"| {p} | {cells[0]} | {cells[1]} | {cells[2]} |")
    body += [
        "",
        f"Total: {info['total_rows']} windows from {info['total_docs']} document rows.",
        "",
        "Reassembly reference implementation: `assemble_document` in",
        "`notebooks/build_windowed_dataset.py`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--legacy-dir", default="notebooks/dataset", type=Path)
    ap.add_argument("--out-dir", default="notebooks/dataset/windowed_dataset", type=Path)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    legacy_dir = repo_root / args.legacy_dir
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    row_counts: dict[str, dict[str, int]] = {p: {} for p in PURPOSES}
    doc_counts: dict[str, dict[str, int]] = {p: {} for p in PURPOSES}
    total_rows = 0
    total_docs = 0
    win_char_max = 0
    for purpose in PURPOSES:
        for split in SPLITS:
            src = legacy_dir / purpose / f"{split}.parquet"
            df = pd.read_parquet(src)
            out_rows: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                out_rows.extend(build_windows_for_row(row))
            sub = out_dir / purpose
            sub.mkdir(parents=True, exist_ok=True)
            wdf = pd.DataFrame(out_rows)
            wdf.to_parquet(sub / f"{split}.parquet", index=False)
            row_counts[purpose][split] = len(wdf)
            doc_counts[purpose][split] = len(df)
            total_rows += len(wdf)
            total_docs += len(df)
            win_char_max = max(win_char_max, int(wdf["n_input_chars"].max()))
            print(f"  {purpose}/{split}: {len(df)} docs -> {len(wdf)} windows "
                  f"(max window {int(wdf['n_input_chars'].max())} chars)")
            del df, wdf, out_rows

    info = {
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": "Haeryz/putusan-structured-extraction (legacy parquets)",
        "window_chars": WINDOW_CHARS,
        "overlap_fraction": OVERLAP_FRACTION,
        "min_overlap_lines": MIN_OVERLAP_LINES,
        "line_no_width": LINE_NO_WIDTH,
        "system_prompt": SYSTEM_PROMPT,
        "canonical_sections": CANONICAL_SECTIONS,
        "row_counts": row_counts,
        "doc_counts": doc_counts,
        "total_rows": total_rows,
        "total_docs": total_docs,
        "max_window_chars_observed": win_char_max,
        "round_trip": "PASSED for every document (gold windows -> exact sections)",
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(out_dir, info)
    print(f"Total: {total_docs} document rows -> {total_rows} windowed rows")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
