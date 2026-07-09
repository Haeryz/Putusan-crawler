#!/usr/bin/env python3
"""Stage 0 — Data foundation for the Putusan structured-extractor pipeline.

Reads ONLY the extracted ``LLM-aggregator/<CORPUS>/<MODEL>/output/*.json`` files
(never the raw ``.txt``) and builds metadata-rich Parquet datasets under
``notebooks/dataset/``.

Pipeline:
    1. Load every usable extraction record (one row per corpus/model/doc —
       all ~4500 model extractions are kept; nothing is collapsed).
    2. Cross-model completion: if a record has an empty section but another
       model extracted that section for the SAME source document
       (matched by ``source_sha256``), the section is filled from that model.
       Per-section provenance is recorded in ``cross_model_fill``.
    3. Leakage-safe splitting: unique documents (by ``source_sha256``) are
       partitioned per corpus into three purposes — SFT 70% / GRPO 15% /
       RAG 15% — then each purpose is split into train 80% / val 10% /
       test 10%. ALL rows of a document land in the same purpose+split,
       so no document ever appears in two splits.
    4. Emit ``notebooks/dataset/{sft,grpo,rag}/{train,val,test}.parquet``
       with full metadata columns (provenance, section statistics, sizes),
       plus ``dataset_info.json`` and a ``README.md`` schema card.

Usage:
    python notebooks/build_dataset.py --repo-root . --seed 3407

Invariants asserted at build time:
    * zero ``source_sha256`` overlap across any two of the nine splits;
    * every span in every target is a substring of its reconstructed input.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

# Canonical section order — used for input reconstruction. Must match
# RAG/ORCHESTRATION.md and LLM-aggregator/Putusan-schema.md exactly (31 keys).
CANONICAL_SECTIONS: list[str] = [
    "judul", "nomor_putusan", "irah_irah", "nama_pengadilan_negeri",
    "keterangan_perkara", "nama_lengkap", "tempat_lahir", "umur_tanggal_lahir",
    "jenis_kelamin", "kebangsaan", "tempat_tinggal", "agama", "pekerjaan",
    "penangkapan", "penahanan", "tuntutan", "dakwaan", "saksi", "ahli",
    "terdakwa", "surat", "petunjuk_barang_bukti", "fakta_hukum",
    "pertimbangan_hukum", "amar_putusan", "hari", "tanggal", "tahun",
    "siapa_yang_memutus", "panitera_pengganti", "tanda_tangan_majelis",
]
assert len(CANONICAL_SECTIONS) == 31, "expected exactly 31 canonical sections"

# Gemini produced no usable output and is dropped from the inventory.
SKIP_MODELS = {"Gemini", "gemini"}

EXPECTED_ROWS_PER_MODEL_DIR = 500

PURPOSES = ("sft", "grpo", "rag")
SPLITS = ("train", "val", "test")
PURPOSE_FRACTIONS = {"sft": 0.70, "grpo": 0.15, "rag": 0.15}
SPLIT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}

# Indonesian extraction instruction used as the SFT system prompt.
SYSTEM_PROMPT = (
    "Anda adalah pengekstrak terstruktur putusan pengadilan Indonesia. "
    "Diberikan badan teks putusan, keluarkan SATU objek JSON dengan tepat 31 "
    "kunci bagian (dalam urutan kanonik). Setiap nilai adalah daftar kutipan "
    "verbatim (extractive) yang disalin persis dari teks sumber — jangan pernah "
    "memparafrasekan, meringkas, atau mengarang. Jika sebuah bagian tidak ada, "
    "gunakan daftar kosong dan cantumkan kuncinya di 'empty_sections'. Kunci "
    "bagian, dalam urutan: " + ", ".join(CANONICAL_SECTIONS) + "."
)


def iter_output_files(repo_root: Path):
    """Yield (corpus, model, path) for every output JSON, skipping dropped models."""
    agg = repo_root / "LLM-aggregator"
    for output_dir in sorted(agg.glob("*/*/output")):
        model = output_dir.parent.name
        corpus = output_dir.parent.parent.name
        if model in SKIP_MODELS:
            continue
        for path in sorted(output_dir.glob("*.json")):
            yield corpus, model, path


def load_record(path: Path) -> dict[str, Any] | None:
    """Load a single extraction JSON; return None if unusable."""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if rec.get("status") != "completed":
        return None
    if not isinstance(rec.get("sections"), dict):
        return None
    return rec


def normalize_sections(rec: dict[str, Any]) -> dict[str, list[str]]:
    """Return sections as {key: list[str]} for all 31 canonical keys (missing -> [])."""
    raw = rec.get("sections", {}) or {}
    out: dict[str, list[str]] = {}
    for key in CANONICAL_SECTIONS:
        val = raw.get(key, [])
        if val is None:
            val = []
        if isinstance(val, str):
            val = [val]
        out[key] = [str(s) for s in val]
    return out


def section_is_empty(spans: list[str]) -> bool:
    return not any(s.strip() for s in spans)


def count_non_empty(sections: dict[str, list[str]]) -> int:
    return sum(1 for v in sections.values() if not section_is_empty(v))


def reconstruct_input(sections: dict[str, list[str]]) -> str:
    """Join all non-empty section spans in canonical order into one text blob.

    This is a .txt-free reconstruction of the decision body — the model input.
    Spans within a section are separated by newlines; sections by blank lines,
    so every emitted span remains a verbatim substring of the reconstruction.
    """
    blocks: list[str] = []
    for key in CANONICAL_SECTIONS:
        spans = [s for s in sections[key] if s.strip()]
        if spans:
            blocks.append("\n".join(spans))
    return "\n\n".join(blocks)


def record_sha(rec: dict[str, Any], path: Path) -> str:
    """source_sha256 if present, else a deterministic fallback from source_file."""
    sha = rec.get("source_sha256")
    if sha:
        return str(sha)
    basis = (rec.get("source_file") or path.stem).encode("utf-8")
    return "fallback:" + hashlib.sha256(basis).hexdigest()


def cross_model_fill(records: list[dict[str, Any]]) -> None:
    """Fill empty sections of each record from sibling extractions of the same doc.

    ``records`` are the mutable row dicts. For every (corpus, sha) group, each
    record's empty sections are filled from the sibling model with the most
    non-empty sections that has that section. Provenance goes to
    ``rec["cross_model_fill"]`` as {section_key: donor_model}.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["corpus"], r["sha"]), []).append(r)

    for group in groups.values():
        models_here = sorted({g["model"] for g in group})
        for r in group:
            r["models_covering_doc"] = models_here
        if len(group) < 2:
            continue
        # Donor preference: most complete sibling first (stable tie-break by model).
        for r in group:
            donors = sorted(
                (g for g in group if g is not r),
                key=lambda g: (-g["non_empty_original"], g["model"]),
            )
            fill: dict[str, str] = {}
            for key in CANONICAL_SECTIONS:
                if not section_is_empty(r["sections"][key]):
                    continue
                for d in donors:
                    if not section_is_empty(d["sections"][key]):
                        r["sections"][key] = list(d["sections"][key])
                        fill[key] = d["model"]
                        break
            r["cross_model_fill"] = fill


def build_target(rec: dict[str, Any], sections: dict[str, list[str]]) -> dict[str, Any]:
    """The supervision target: full 31-key sections + empty_sections + record id."""
    empty = [k for k in CANONICAL_SECTIONS if section_is_empty(sections[k])]
    return {
        "status": "completed",
        "source_file": rec.get("source_file"),
        "source_sha256": rec.get("source_sha256"),
        "sections": sections,
        "empty_sections": empty,
    }


def partition(items: list[str], fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Deterministically shuffle ``items`` and assign each to a named bucket."""
    items = sorted(items)
    random.Random(seed).shuffle(items)
    names = list(fractions)
    n = len(items)
    counts = [int(round(n * fractions[name])) for name in names]
    counts[-1] = n - sum(counts[:-1])  # remainder goes to the last bucket
    assignment: dict[str, str] = {}
    idx = 0
    for name, c in zip(names, counts):
        for it in items[idx:idx + c]:
            assignment[it] = name
        idx += c
    return assignment


def assign_splits(records: list[dict[str, Any]], seed: int) -> None:
    """Per corpus: purpose 70/15/15 over unique shas, then train/val/test 80/10/10.

    Assignment is by document sha, so every row of a document (one per model)
    lands in the same purpose+split — zero cross-split leakage.
    """
    by_corpus: dict[str, set[str]] = {}
    for r in records:
        by_corpus.setdefault(r["corpus"], set()).add(r["sha"])

    purpose_of: dict[tuple[str, str], str] = {}
    split_of: dict[tuple[str, str], str] = {}
    for corpus, shas in sorted(by_corpus.items()):
        purpose_assign = partition(sorted(shas), PURPOSE_FRACTIONS, seed)
        per_purpose: dict[str, list[str]] = {p: [] for p in PURPOSES}
        for sha, purpose in purpose_assign.items():
            per_purpose[purpose].append(sha)
        for purpose, p_shas in per_purpose.items():
            # Vary the seed per corpus+purpose so shuffles are independent
            # (hashlib, not hash(): built-in str hashing is per-process random).
            digest = hashlib.sha256(f"{corpus}|{purpose}".encode()).hexdigest()
            sub_seed = seed + int(digest[:8], 16)
            split_assign = partition(p_shas, SPLIT_FRACTIONS, sub_seed)
            for sha, split in split_assign.items():
                purpose_of[(corpus, sha)] = purpose
                split_of[(corpus, sha)] = split

    for r in records:
        key = (r["corpus"], r["sha"])
        r["purpose"] = purpose_of[key]
        r["split"] = split_of[key]


def make_row(r: dict[str, Any], seed: int) -> dict[str, Any]:
    """One flat, metadata-rich Parquet row (uniform schema across all files)."""
    sections = r["sections"]
    target = build_target(r["rec"], sections)
    target_json = json.dumps(target, ensure_ascii=False)
    input_text = r["input"]
    empty = target["empty_sections"]
    fill = r.get("cross_model_fill", {})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": target_json},
    ]
    return {
        # identity & provenance
        "id": f"{r['corpus']}/{r['model']}/{r['sha'][:16]}",
        "corpus": r["corpus"],
        "annotator_model": r["model"],
        "source_file": r["rec"].get("source_file"),
        "source_sha256": r["sha"],
        "extraction_method": r["rec"].get("method") or "codex_manual_extractive",
        # split bookkeeping
        "purpose": r["purpose"],
        "split": r["split"],
        "split_seed": seed,
        # payload
        "input_text": input_text,
        "target_json": target_json,
        "sections_json": json.dumps(sections, ensure_ascii=False),
        "messages": messages,          # SFT-style conversation
        "prompt": messages[:2],        # GRPO-style prompt (system + user)
        "answer": target_json,         # GRPO-style reference answer
        # section statistics & cross-model completion provenance
        "n_sections": len(CANONICAL_SECTIONS),
        "n_nonempty_sections": count_non_empty(sections),
        "empty_sections": empty,
        "n_empty_sections": len(empty),
        "cross_model_fill_json": json.dumps(fill, ensure_ascii=False),
        "n_sections_filled_cross_model": len(fill),
        "models_covering_doc": r.get("models_covering_doc", [r["model"]]),
        # size statistics
        "n_input_chars": len(input_text),
        "n_input_words": len(input_text.split()),
        "n_target_chars": len(target_json),
    }


def write_readme(out_dir: Path, info: dict[str, Any]) -> None:
    lines = [
        "# Putusan structured-extraction dataset",
        "",
        f"Built {info['built_at']} by `notebooks/build_dataset.py` (seed {info['seed']}).",
        "",
        "Indonesian court-decision (putusan) extractive-structuring dataset over three",
        "corpora (Anak, Asusila, TPPO). Each row is one model extraction of one source",
        "document into 31 canonical sections of verbatim spans. Empty sections were",
        "completed from sibling model extractions of the same document where available",
        "(`cross_model_fill_json` records per-section donor provenance).",
        "",
        "## Files",
        "",
        "`{sft,grpo,rag}/{train,val,test}.parquet` — purposes are document-disjoint",
        "(SFT 70% / GRPO 15% / RAG 15% of unique documents per corpus), then each",
        "purpose is split train 80% / val 10% / test 10%. All rows of a document",
        "(one per annotator model) share the same purpose+split, so there is zero",
        "document leakage across any pair of files.",
        "",
        "## Schema (uniform across all files)",
        "",
        "| column | type | description |",
        "|---|---|---|",
        "| id | str | `corpus/annotator_model/sha16` |",
        "| corpus | str | Anak, Asusila, or TPPO |",
        "| annotator_model | str | model that produced the extraction (GPT/Deepseek/Qwen) |",
        "| source_file / source_sha256 | str | source decision text identity |",
        "| extraction_method | str | extraction protocol tag |",
        "| purpose / split | str | sft-grpo-rag / train-val-test |",
        "| split_seed | int | RNG seed used for the deterministic split |",
        "| input_text | str | decision body reconstructed from section spans |",
        "| target_json | str | full 31-key supervision target (JSON) |",
        "| sections_json | str | the 31 sections alone (JSON) |",
        "| messages | list<struct{role,content}> | system/user/assistant conversation (SFT) |",
        "| prompt | list<struct{role,content}> | system+user only (GRPO rollout prompt) |",
        "| answer | str | reference answer for reward computation (GRPO) |",
        "| n_nonempty_sections / n_empty_sections / empty_sections | int/list | section coverage |",
        "| cross_model_fill_json | str | {section: donor_model} completion provenance |",
        "| n_sections_filled_cross_model | int | sections completed from a sibling model |",
        "| models_covering_doc | list<str> | all models that extracted this document |",
        "| n_input_chars / n_input_words / n_target_chars | int | size statistics |",
        "",
        "## Row counts",
        "",
        "| purpose | train | val | test |",
        "|---|---|---|---|",
    ]
    for p in PURPOSES:
        c = info["row_counts"][p]
        lines.append(f"| {p} | {c['train']} | {c['val']} | {c['test']} |")
    lines += [
        "",
        f"Total rows: {info['total_rows']} "
        f"(expected {info['expected_rows']}; shortfall is missing/failed",
        "extractions — see `dataset_info.json` `per_dir_file_counts`).",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--out-dir", default="notebooks/dataset", type=Path)
    ap.add_argument("--seed", default=3407, type=int)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir) if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load every usable extraction record — one row per corpus/model/doc.
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    per_dir_counts: dict[str, int] = {}
    for corpus, model, path in iter_output_files(repo_root):
        per_dir_counts[f"{corpus}/{model}"] = per_dir_counts.get(f"{corpus}/{model}", 0) + 1
        rec = load_record(path)
        if rec is None:
            skipped.append(str(path.relative_to(repo_root)))
            continue
        sections = normalize_sections(rec)
        records.append({
            "corpus": corpus,
            "model": model,
            "sha": record_sha(rec, path),
            "rec": rec,
            "sections": sections,
            "non_empty_original": count_non_empty(sections),
        })

    # 2. Cross-model completion of empty sections (same doc, sibling model).
    cross_model_fill(records)

    # 3. Reconstruct inputs AFTER filling, so filled spans stay substrings.
    kept: list[dict[str, Any]] = []
    for r in records:
        r["input"] = reconstruct_input(r["sections"])
        if r["input"].strip():
            kept.append(r)
        else:
            skipped.append(f"{r['corpus']}/{r['model']}/{r['rec'].get('source_file')}")
    records = kept

    # 4. Leakage-safe purpose + split assignment by document sha.
    assign_splits(records, args.seed)

    # 5. Build rows, assert the per-span substring invariant.
    rows_by_file: dict[tuple[str, str], list[dict[str, Any]]] = {
        (p, s): [] for p in PURPOSES for s in SPLITS
    }
    shas_by_file: dict[tuple[str, str], set[str]] = {
        (p, s): set() for p in PURPOSES for s in SPLITS
    }
    for r in records:
        for key, spans in r["sections"].items():
            for span in spans:
                if span.strip() and span not in r["input"]:
                    raise AssertionError(
                        f"span not substring of input: sha={r['sha']} "
                        f"section={key} span={span[:60]!r}"
                    )
        fkey = (r["purpose"], r["split"])
        rows_by_file[fkey].append(make_row(r, args.seed))
        shas_by_file[fkey].add(r["sha"])

    # 6. Assert zero document overlap across ALL nine files.
    file_keys = list(shas_by_file)
    for i, a in enumerate(file_keys):
        for b in file_keys[i + 1:]:
            overlap = shas_by_file[a] & shas_by_file[b]
            assert not overlap, f"document leakage between {a} and {b}: {len(overlap)} shas"

    # 7. Write Parquet + dataset card.
    row_counts: dict[str, dict[str, int]] = {p: {} for p in PURPOSES}
    for (purpose, split), rows in rows_by_file.items():
        sub = out_dir / purpose
        sub.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(sub / f"{split}.parquet", index=False)
        row_counts[purpose][split] = len(df)

    total_rows = len(records)
    n_filled_rows = sum(1 for r in records if r.get("cross_model_fill"))
    n_filled_sections = sum(len(r.get("cross_model_fill", {})) for r in records)
    info = {
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "canonical_sections": CANONICAL_SECTIONS,
        "system_prompt": SYSTEM_PROMPT,
        "purpose_fractions": PURPOSE_FRACTIONS,
        "split_fractions": SPLIT_FRACTIONS,
        "expected_rows": EXPECTED_ROWS_PER_MODEL_DIR * len(per_dir_counts),
        "total_rows": total_rows,
        "unique_documents": len({(r["corpus"], r["sha"]) for r in records}),
        "rows_with_cross_model_fill": n_filled_rows,
        "sections_filled_cross_model": n_filled_sections,
        "per_dir_file_counts": per_dir_counts,
        "skipped_files": skipped,
        "row_counts": row_counts,
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(out_dir, info)

    print(f"Loaded {total_rows} rows ({len(skipped)} skipped) "
          f"from {len(per_dir_counts)} corpus/model dirs "
          f"(expected {info['expected_rows']})")
    print(f"Unique documents: {info['unique_documents']}; "
          f"cross-model fill: {n_filled_sections} sections across {n_filled_rows} rows")
    for p in PURPOSES:
        c = row_counts[p]
        print(f"  {p:5s}: train={c['train']:5d}  val={c['val']:4d}  test={c['test']:4d}"
              f"  -> {out_dir / p}")
    print(f"Dataset card: {out_dir / 'README.md'}; stats: {out_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
