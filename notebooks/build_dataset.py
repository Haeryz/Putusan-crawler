#!/usr/bin/env python3
"""Stage 0 — Data foundation for the Putusan structured-extractor pipeline.

Reads ONLY the extracted ``LLM-aggregator/<CORPUS>/<MODEL>/output/*.json`` files
(never the raw ``.txt``), deduplicates by ``source_sha256``, reconstructs the model
INPUT from each record's own section spans, and emits a 70/15/15 split (per corpus,
by ``source_sha256``) into ``data/train/{sft,grpo,benchmark}.jsonl``.

See ``RAG/ORCHESTRATION.md`` §"Stage 0 — Data foundation" for the contract. The
notebook ``notebooks/Qwen3_5_(4B)_Vision.ipynb`` (Stage 1 SFT) consumes ``sft.jsonl``.

Usage:
    python notebooks/build_dataset.py \
        --repo-root . \
        --out-dir data/train \
        --seed 3407

Invariants asserted at build time:
    * zero ``source_sha256`` overlap across the three splits;
    * every span in every target is a substring of its reconstructed input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

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


def count_non_empty(sections: dict[str, list[str]]) -> int:
    return sum(1 for v in sections.values() if any(s.strip() for s in v))


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


def build_target(rec: dict[str, Any], sections: dict[str, list[str]]) -> dict[str, Any]:
    """The supervision target: full 31-key sections + empty_sections + record id."""
    empty = [k for k in CANONICAL_SECTIONS if not any(s.strip() for s in sections[k])]
    return {
        "status": "completed",
        "source_file": rec.get("source_file"),
        "source_sha256": rec.get("source_sha256"),
        "sections": sections,
        "empty_sections": empty,
    }


def record_sha(rec: dict[str, Any], path: Path) -> str:
    """source_sha256 if present, else a deterministic fallback from source_file."""
    sha = rec.get("source_sha256")
    if sha:
        return str(sha)
    basis = (rec.get("source_file") or path.stem).encode("utf-8")
    return "fallback:" + hashlib.sha256(basis).hexdigest()


def dedup(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the same decision (by sha) to one logical doc; keep the most complete."""
    best: dict[str, dict[str, Any]] = {}
    for r in records:
        sha = r["sha"]
        cur = best.get(sha)
        if cur is None or r["non_empty"] > cur["non_empty"]:
            if cur is not None:
                # record cross-model agreement as provenance
                r["also_extracted_by"] = sorted(
                    set(cur.get("also_extracted_by", []))
                    | {cur["model"]}
                )
            best[sha] = r
        else:
            cur.setdefault("also_extracted_by", [])
            if r["model"] not in cur["also_extracted_by"]:
                cur["also_extracted_by"].append(r["model"])
    return list(best.values())


def split_by_sha(shas: list[str], seed: int) -> dict[str, set[str]]:
    """Deterministic 70/15/15 split of a sha list into sft/grpo/benchmark."""
    shas = sorted(shas)
    random.Random(seed).shuffle(shas)
    n = len(shas)
    n_sft = int(round(n * 0.70))
    n_grpo = int(round(n * 0.15))
    return {
        "sft": set(shas[:n_sft]),
        "grpo": set(shas[n_sft:n_sft + n_grpo]),
        "benchmark": set(shas[n_sft + n_grpo:]),
    }


def make_sft_row(rec_input: str, target: dict[str, Any], meta: dict[str, Any]) -> dict:
    answer = json.dumps(target, ensure_ascii=False, indent=2)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rec_input},
            {"role": "assistant", "content": answer},
        ],
        "meta": meta,
    }


def make_grpo_row(rec_input: str, target: dict[str, Any], meta: dict[str, Any]) -> dict:
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": rec_input},
    ]
    return {
        "prompt": prompt,
        "answer": json.dumps(target, ensure_ascii=False),
        "meta": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--out-dir", default="data/train", type=Path)
    ap.add_argument("--seed", default=3407, type=int)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir) if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load every usable extraction record.
    records: list[dict[str, Any]] = []
    skipped = 0
    for corpus, model, path in iter_output_files(repo_root):
        rec = load_record(path)
        if rec is None:
            skipped += 1
            continue
        sections = normalize_sections(rec)
        rec_input = reconstruct_input(sections)
        if not rec_input.strip():
            skipped += 1
            continue
        records.append({
            "corpus": corpus,
            "model": model,
            "sha": record_sha(rec, path),
            "non_empty": count_non_empty(sections),
            "input": rec_input,
            "target": build_target(rec, sections),
            "sections": sections,
        })

    # 2. Dedup by source_sha256 (one logical doc; keep most complete extraction).
    deduped = dedup(records)

    # 3. Split per corpus, 70/15/15 by sha.
    by_corpus: dict[str, list[str]] = {}
    for r in deduped:
        by_corpus.setdefault(r["corpus"], []).append(r["sha"])
    split_of: dict[str, str] = {}
    for corpus, shas in by_corpus.items():
        assignment = split_by_sha(shas, args.seed)
        for split_name, sha_set in assignment.items():
            for sha in sha_set:
                split_of[sha] = split_name

    # 4. Emit rows + assert the per-span substring invariant.
    writers = {name: (out_dir / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ("sft", "grpo", "benchmark")}
    seen_by_split: dict[str, set[str]] = {name: set() for name in writers}
    counts = {name: 0 for name in writers}
    try:
        for r in deduped:
            split = split_of[r["sha"]]
            seen_by_split[split].add(r["sha"])
            # Invariant: every emitted span is a substring of the reconstructed input.
            for key, spans in r["sections"].items():
                for span in spans:
                    if span.strip() and span not in r["input"]:
                        raise AssertionError(
                            f"span not substring of input: sha={r['sha']} "
                            f"section={key} span={span[:60]!r}"
                        )
            meta = {
                "corpus": r["corpus"],
                "model": r["model"],
                "source_sha256": r["sha"],
                "source_file": r["target"].get("source_file"),
                "also_extracted_by": r.get("also_extracted_by", []),
            }
            if split == "grpo":
                row = make_grpo_row(r["input"], r["target"], meta)
            else:
                row = make_sft_row(r["input"], r["target"], meta)
            writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for w in writers.values():
            w.close()

    # 5. Assert zero sha overlap across splits.
    s, g, b = seen_by_split["sft"], seen_by_split["grpo"], seen_by_split["benchmark"]
    assert not (s & g) and not (s & b) and not (g & b), "source_sha256 overlap across splits"

    print(f"Loaded {len(records)} records ({skipped} skipped) -> {len(deduped)} unique docs")
    for name in ("sft", "grpo", "benchmark"):
        print(f"  {name:9s}: {counts[name]:5d} rows -> {out_dir / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
