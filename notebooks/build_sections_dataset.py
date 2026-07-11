# -*- coding: utf-8 -*-
"""Build the `sft_sections` config: per-section extraction examples from the whole-doc SFT data.

Each parent `sft` row (full putusan body -> 31-section JSON) becomes SIX child examples,
each keeping the FULL document as input but asking for only 1-2 sections — matching
inference, where the user uploads a whole PDF and asks for one or two sections.

Section tiers are DATA-DRIVEN, measured on sft/train.parquet (2,468 docs):

  empty-prone  (teach "when to say empty"):   ahli 67.4% empty, penangkapan 29.0%, surat 15.8%
  long-hard    (huge verbatim bodies):        pertimbangan_hukum med 16.7k chars, saksi 13.5k,
                                              dakwaan 10.3k, terdakwa 4.5k, fakta_hukum 4.2k
  medium:                                     tuntutan, penahanan, petunjuk_barang_bukti, amar_putusan
  trivial      (identity/date, <1% empty,     the remaining 19 sections — need coverage only,
                median 4-226 chars):          so they share one two-section example per doc

Per document at deterministic index i (sorted by id within its split):
  1. ahli                                  (always — the hardest section)
  2. penangkapan or surat                  (alternating)
  3. two of the five long-hard sections    (rotating, step 2)
  4. one medium section                    (rotating)
  5. one PAIR of trivial sections          (rotating through all 19; trains the 2-question format)

No RNG anywhere — reruns are byte-identical. The old `sft` config is not touched.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).parent / "dataset" / "sft"
OUT_DIR = Path(__file__).parent / "dataset" / "sft_sections"

CANONICAL_ORDER = [
    "judul", "nomor_putusan", "irah_irah", "nama_pengadilan_negeri", "keterangan_perkara",
    "nama_lengkap", "tempat_lahir", "umur_tanggal_lahir", "jenis_kelamin", "kebangsaan",
    "tempat_tinggal", "agama", "pekerjaan", "penangkapan", "penahanan", "tuntutan",
    "dakwaan", "saksi", "ahli", "terdakwa", "surat", "petunjuk_barang_bukti",
    "fakta_hukum", "pertimbangan_hukum", "amar_putusan", "hari", "tanggal", "tahun",
    "siapa_yang_memutus", "panitera_pengganti", "tanda_tangan_majelis",
]

EMPTY_PRONE = ["ahli"]                      # always asked
EMPTY_PRONE_ALT = ["penangkapan", "surat"]  # alternating
LONG_HARD = ["pertimbangan_hukum", "saksi", "dakwaan", "terdakwa", "fakta_hukum"]
MEDIUM = ["tuntutan", "penahanan", "petunjuk_barang_bukti", "amar_putusan"]
TRIVIAL = [s for s in CANONICAL_ORDER
           if s not in EMPTY_PRONE + EMPTY_PRONE_ALT + LONG_HARD + MEDIUM]
assert len(TRIVIAL) == 19

SYSTEM_TEMPLATE = (
    "Anda adalah pengekstrak terstruktur putusan pengadilan Indonesia. Diberikan badan teks "
    "putusan, keluarkan SATU objek JSON yang hanya berisi bagian yang diminta. Setiap nilai "
    "adalah daftar kutipan verbatim (extractive) yang disalin persis dari teks sumber — jangan "
    "pernah memparafrasekan, meringkas, atau mengarang. Jika bagian yang diminta tidak ada "
    "dalam teks, gunakan daftar kosong dan cantumkan kuncinya di 'empty_sections'. "
    "Bagian yang diminta: {keys}."
)


def requests_for_doc(i: int) -> list[list[str]]:
    """The six deterministic section requests for the doc at split index i."""
    return [
        EMPTY_PRONE,
        [EMPTY_PRONE_ALT[i % 2]],
        [LONG_HARD[i % 5]],
        [LONG_HARD[(i + 2) % 5]],
        [MEDIUM[i % 4]],
        [TRIVIAL[(2 * i) % 19], TRIVIAL[(2 * i + 1) % 19]],
    ]


def build_split(split_name: str, parquet_name: str) -> pd.DataFrame:
    # The parent train split carries 7 exact-content duplicates (same sha under an
    # "X.txt" / "X-2.txt" filename pair, hence the same id) — keep one of each.
    df = (pd.read_parquet(SRC_DIR / parquet_name)
          .drop_duplicates("id").sort_values("id").reset_index(drop=True))
    rows = []
    for i, r in df.iterrows():
        sections = json.loads(r["target_json"])["sections"]
        input_text = r["input_text"]
        for keys in requests_for_doc(i):
            asked = {k: list(sections.get(k, [])) for k in keys}
            for k, spans in asked.items():
                for span in spans:
                    assert span in input_text, f"span not in input: {r['id']} {k}"
            empty = [k for k, v in asked.items() if not v]
            target = json.dumps({"sections": asked, "empty_sections": empty},
                                ensure_ascii=False)
            system = SYSTEM_TEMPLATE.format(keys=", ".join(keys))
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": target},
            ]
            rows.append({
                "id": f"{r['id']}::{'+'.join(keys)}",
                "parent_id": r["id"],
                "corpus": r["corpus"],
                "annotator_model": r["annotator_model"],
                "source_file": r["source_file"],
                "source_sha256": r["source_sha256"],
                "extraction_method": r["extraction_method"],
                "purpose": r["purpose"],
                "split": split_name,
                "split_seed": r["split_seed"],
                "requested_sections": keys,
                "n_requested": len(keys),
                "n_empty_requested": len(empty),
                "input_text": input_text,
                "target_json": target,
                "messages": messages,
                "prompt": messages[:2],
                "answer": target,
                "n_input_chars": len(input_text),
                "n_target_chars": len(target),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for split_name, parquet_name in [("train", "train.parquet"),
                                     ("validation", "val.parquet"),
                                     ("test", "test.parquet")]:
        out = build_split(split_name, parquet_name)
        out.to_parquet(OUT_DIR / parquet_name, index=False)
        n_docs = out["parent_id"].nunique()
        empty_rate = (out["n_empty_requested"] > 0).mean()
        all_stats.append((split_name, len(out), n_docs, empty_rate))
        print(f"{split_name}: {len(out)} examples from {n_docs} docs "
              f"({empty_rate:.1%} contain an empty-section ask) "
              f"-> {OUT_DIR / parquet_name}")

    # Section coverage sanity check on train.
    train = pd.read_parquet(OUT_DIR / "train.parquet",
                            columns=["requested_sections"])
    counts = pd.Series(
        [k for keys in train["requested_sections"] for k in keys]
    ).value_counts()
    missing = set(CANONICAL_ORDER) - set(counts.index)
    assert not missing, f"sections never asked: {missing}"
    print("\nTrain ask-count per section (min..max): "
          f"{counts.min()}..{counts.max()}")
    print(counts.to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
