"""Precompute section-sliced Qwen3.5 evaluation prompts and upload them to W&B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from tqdm.auto import tqdm
from transformers import AutoTokenizer

MODEL_NATIVE_MAX_LENGTH = 262_144
MAX_GENERATION_TOKENS_CAP = 131_072
MIN_GENERATION_TOKENS = 256
THINKING_TOKEN_RESERVE = 1_024
OUTPUT_TOKEN_HEADROOM = 1.15
CONTEXT_ALIGNMENT = 256
DEFAULT_ENABLE_THINKING = True
NON_THINKING_MIN_GENERATION_TOKENS = 64
NON_THINKING_OUTPUT_TOKEN_HEADROOM = 1.05
NON_THINKING_TOKEN_RESERVE = 32

SECTION_GUIDANCE = {
    "judul": "Judul: ambil judul PUTUSAN/P U T U S A N di awal dokumen; berhenti sebelum nomor perkara.",
    "nomor_putusan": "Nomor Putusan: ambil nomor perkara yang mengikuti judul; jangan sertakan nama pengadilan atau irah-irah.",
    "irah_irah": "Irah-irah: ambil formula DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA, termasuk variasi spasi/OCR, tanpa bagian pengadilan sesudahnya.",
    "nama_pengadilan_negeri": "Nama Pengadilan Negeri: ambil nama pengadilan tingkat pertama pada kalimat pembuka, bukan seluruh uraian jenis perkara.",
    "keterangan_perkara": "Keterangan Perkara: ambil uraian bahwa pengadilan mengadili perkara, acara pemeriksaan, tingkat, dan subjek perkara sampai sebelum identitas.",
    "nama_lengkap": "Nama Lengkap: ambil hanya nilai identitas pada label 1. Nama lengkap. Untuk beberapa Anak/Terdakwa, pertahankan semua nilai dan label I/II/III dalam urutan dokumen.",
    "tempat_lahir": "Tempat Lahir: ambil hanya nilai pada label 2. Tempat lahir untuk setiap subjek; berhenti sebelum umur/tanggal lahir.",
    "umur_tanggal_lahir": "Umur/Tanggal Lahir: ambil hanya nilai umur dan/atau tanggal lahir pada label 3 untuk setiap subjek.",
    "jenis_kelamin": "Jenis Kelamin: ambil hanya nilai pada label 4 untuk setiap subjek.",
    "kebangsaan": "Kebangsaan: ambil hanya nilai pada label 5; Kewarganegaraan/nasionalitas adalah alias. Jangan menyerap Pendidikan yang kadang muncul sesudahnya.",
    "tempat_tinggal": "Tempat Tinggal: ambil hanya alamat pada label 6 untuk setiap subjek; berhenti sebelum Agama.",
    "agama": "Agama: ambil hanya nilai pada label 7 untuk setiap subjek; berhenti sebelum Pekerjaan.",
    "pekerjaan": "Pekerjaan: ambil hanya nilai pada label 8 untuk setiap subjek; berhenti sebelum kalimat penangkapan/penahanan atau prosedur berikutnya.",
    "penangkapan": "Penangkapan: ambil hanya kalimat/perintah penangkapan beserta tanggal dan referensinya. Jangan memasukkan satu pun tahap penahanan.",
    "penahanan": "Penahanan: ambil seluruh tahap dan perpanjangan penahanan, termasuk penangguhan, pembantaran, pengalihan, atau penahanan dalam perkara lain. Jangan memasukkan penangkapan.",
    "tuntutan": "Tuntutan: mulai pada pembacaan tuntutan pidana dan salin seluruh amar tuntutan bernomor, pidana/denda yang diminta, status tahanan, barang bukti/restitusi bila ada, dan biaya; berhenti pada pembelaan atau dakwaan berikutnya.",
    "dakwaan": "Dakwaan: mulai saat Terdakwa/Anak didakwa berdasarkan surat dakwaan dan ambil lengkap semua bentuk tunggal, alternatif, subsidair, kumulatif, atau gabungan beserta uraian perbuatan dan pasal; pada acara singkat, catatan dakwaan adalah dakwaan.",
    "saksi": "Saksi: ambil seluruh keterangan saksi penuntut, korban/anak korban/anak saksi, saksi meringankan (a de charge), dan verbalisan, termasuk nama, sumpah, butir keterangan, serta tanggapan Terdakwa/Anak.",
    "ahli": "Ahli: ambil seluruh keterangan ahli penuntut maupun pembela, termasuk keterangan ahli yang dibacakan di persidangan; jangan mengisi dari keterangan saksi biasa.",
    "terdakwa": "Terdakwa: ambil keterangan Terdakwa/Para Terdakwa atau Anak sendiri di persidangan, dari formula telah memberikan keterangan sampai sebelum kelompok alat bukti/fakta berikutnya.",
    "surat": "Surat: ambil alat bukti surat, dokumen, dan alat bukti elektronik pada bagian Surat/bukti surat; jangan mencampurnya dengan daftar barang bukti fisik.",
    "petunjuk_barang_bukti": "Petunjuk/Barang Bukti: ambil inventaris barang bukti yang diajukan Penuntut Umum. Pembahasan hukum tentang nasib/disposisi barang bukti yang muncul kemudian bukan inventaris ini.",
    "fakta_hukum": "Fakta Hukum: mulai pada formula berdasarkan alat bukti diperoleh fakta hukum/fakta-fakta hukum dan ambil daftar faktanya; berhenti tepat sebelum Majelis mulai analisis hukum atau unsur.",
    "pertimbangan_hukum": "Pertimbangan Hukum: mulai ketika Majelis mempertimbangkan dakwaan/unsur. Sertakan analisis semua unsur, kesimpulan pembuktian, alasan pembenar/pemaaf, sanksi, tahanan, restitusi/kompensasi, disposisi barang bukti, keadaan memberatkan/meringankan, biaya, dan Mengingat/Memperhatikan; berhenti sebelum MENGADILI.",
    "amar_putusan": "Amar Putusan: mulai pada MENGADILI/M E N G A D I L I dan ambil setiap perintah bernomor—status terbukti/bebas/lepas, pidana, denda/restitusi, tahanan, barang bukti, dan biaya—sampai sebelum Demikianlah diputuskan.",
    "hari": "Hari: ambil nama hari tanggal musyawarah putusan dari formula Demikianlah diputuskan, bukan hari pengucapan bila berbeda.",
    "tanggal": "Tanggal: ambil tanggal musyawarah putusan persis seperti span sumber yang sudah dipotong; jangan mengubah atau membuang tahunnya bila tercantum.",
    "tahun": "Tahun: ambil hanya tahun musyawarah putusan dari formula penutup.",
    "siapa_yang_memutus": "Siapa yang Memutus: salin seluruh span keputusan tentang hakim yang memutus, termasuk formula dan peran yang sudah tercakup dalam span; jangan mempersempit span menjadi nama saja.",
    "panitera_pengganti": "Panitera Pengganti: ambil nama Panitera/Panitera Pengganti yang membantu persidangan dari paragraf penutup atau blok tanda tangan.",
    "tanda_tangan_majelis": "Tanda Tangan Majelis: ambil blok tanda tangan/nama Hakim Ketua, Hakim Anggota, dan Panitera Pengganti pada akhir dokumen, mempertahankan susunan aslinya.",
}

COMMON_CONTRACT_TEMPLATE = """Anda mengekstrak SATU bagian putusan pengadilan Indonesia dari teks sumber yang sudah dipotong khusus untuk bagian tersebut.
Bagian yang diminta: {section}.
Keluarkan SATU objek JSON saja, tanpa markdown, penjelasan, analisis, reasoning, atau teks lain.
Bentuk wajib: {{"sections": {{"{section}": ["kutipan"]}}, "empty_sections": []}}.
Objek sections harus berisi tepat satu kunci, yaitu {section}; jangan keluarkan 30 kunci lain.
Teks sumber berisi nol atau lebih blok <span>...</span>. Salin seluruh isi setiap blok sebagai tepat satu item array, pertahankan urutannya, dan jangan sertakan tag <span>. Jangan mempersempit atau membuang bagian dari isi blok.
Setiap nilai harus array string. Setiap string harus kutipan verbatim dan kontigu dari teks sumber: jangan meringkas, memparafrasekan, memperbaiki OCR, menormalkan ejaan/spasi, menggabungkan potongan tak-kontigu, atau mengarang.
Jika ada beberapa span terpisah, pertahankan sebagai beberapa item array dalam urutan sumber.
Jika teks sumber kosong atau bagian tidak ada, gunakan [] dan isi empty_sections dengan ["{section}"]. Jika ada kutipan, empty_sections harus [].

Panduan bagian ini:
{guidance}
"""

CORPUS_ADDENDA = {
    "Anak": "Korpus Anak: subjek adalah Anak, bukan Terdakwa dewasa. Identitas jamak tetap berurutan. Penahanan mencakup seluruh LPAS/LPKS dan perpanjangannya. Saksi mencakup anak korban dan anak saksi. Keterangan Anak sendiri masuk terdakwa. Laporan Penelitian Kemasyarakatan, rekomendasi Pembimbing Kemasyarakatan, orang tua/wali/pendamping, kepentingan terbaik Anak, pilihan tindakan/pidana, dan alasan sanksi Anak masuk pertimbangan_hukum bila berada dalam analisis Majelis. Dakwaan acara singkat dapat disebut catatan dakwaan.",
    "Asusila": "Korpus Asusila/Pidana Biasa: subjek adalah Terdakwa/Para Terdakwa dan semua identitas harus berurutan. Penahanan Rutan dapat memuat Penyidik, perpanjangan Penuntut Umum, Ketua PN, Penuntut Umum, Hakim/Majelis, Ketua PT, serta tahap lanjutan. Dakwaan dapat tunggal, alternatif, subsidairitas, kumulatif, atau gabungan. Saksi mencakup korban, a de charge, dan verbalisan. Surat mencakup dokumen/elektronik. Restitusi atau kompensasi masuk pertimbangan atau amar sesuai letaknya.",
    "TPPO": "Korpus TPPO: pertahankan semua Terdakwa dalam urutan. Pada penahanan cari Khusus Penahanan Tindak Pidana TPPO dan ambil seluruh tahap sampai perpanjangan kedua Ketua PT bila ada. Tuntutan dapat memuat restitusi. Pertimbangan TPPO mencakup bagian KHUSUS PERKARA TPPO, restitusi, tenggang pembayaran, penyitaan/lelang, pidana pengganti, serta disposisi barang bukti. Amar harus memuat semua perintah restitusi termasuk tenggang 14 hari, penyitaan/lelang harta, atau pidana pengganti. Jangan pindahkan pembahasan disposisi barang bukti ke inventaris petunjuk_barang_bukti.",
}

OUTPUT_COLUMNS = [
    "no", "dataset_id", "source_row_no", "parent_id", "corpus", "section", "source_file",
    "source_sha256", "annotator_model", "extraction_method", "purpose", "split",
    "span_count", "is_empty", "sliced_input", "sliced_input_chars", "sliced_input_tokens",
    "question", "system_prompt", "prompt", "prompt_sha256", "gold_answer",
    "prompt_tokens_estimate", "gold_tokens", "max_new_tokens", "sequence_token_budget",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("notebooks/dataset/sft/test.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/precomputed/qwen3-5-4b-sliced-section-eval.parquet"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/precomputed/qwen3-5-4b-sliced-section-eval-summary.json"),
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--entity", default="haeriz42069-universitas-muhammadiyah-malang")
    parser.add_argument("--project", default="Sinergi-training")
    parser.add_argument("--artifact", default="qwen3-5-4b-sliced-section-eval-inputs")
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_THINKING,
        help="Render Qwen3.5 thinking prompts (use --no-thinking for direct extraction).",
    )
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--upload-existing",
        action="store_true",
        help="Validate and upload the existing --output/--summary files without recomputing.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_examples(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"id", "corpus", "source_file", "source_sha256", "target_json"}
    if missing := required - set(frame.columns):
        raise RuntimeError(f"Missing dataset columns: {sorted(missing)}")
    examples: list[dict[str, Any]] = []
    source_rows = frame.to_dict("records")
    for source_row_no, row in enumerate(
        tqdm(source_rows, desc="Slicing target_json", unit="row"), start=1
    ):
        sections = json.loads(row["target_json"])["sections"]
        if set(sections) != set(SECTION_GUIDANCE):
            raise RuntimeError(f"Canonical section mismatch for {row['id']}")
        if row["corpus"] not in CORPUS_ADDENDA:
            raise RuntimeError(f"Unknown corpus {row['corpus']!r}")
        for section, guidance in SECTION_GUIDANCE.items():
            spans = list(sections[section])
            sliced_input = "\n".join(f"<span>\n{span}\n</span>" for span in spans)
            question_source = sliced_input if sliced_input else "[TIDAK ADA BLOK <span>]"
            question = f"Bagian yang diminta: {section}.\n\nTEKS SUMBER:\n{question_source}"
            system_prompt = (
                COMMON_CONTRACT_TEMPLATE.format(section=section, guidance=guidance)
                + "\n"
                + CORPUS_ADDENDA[row["corpus"]]
            )
            gold_answer = json.dumps(
                {"sections": {section: spans}, "empty_sections": [section] if not spans else []},
                ensure_ascii=False,
            )
            examples.append({
                "dataset_id": f"row-{source_row_no:04d}::{row['id']}::{section}",
                "source_row_no": source_row_no,
                "parent_id": row["id"],
                "corpus": row["corpus"],
                "section": section,
                "source_file": row["source_file"],
                "source_sha256": row["source_sha256"],
                "annotator_model": row.get("annotator_model"),
                "extraction_method": row.get("extraction_method"),
                "purpose": row.get("purpose"),
                "split": row.get("split"),
                "span_count": len(spans),
                "is_empty": not spans,
                "sliced_input": sliced_input,
                "sliced_input_chars": len(sliced_input),
                "question": question,
                "system_prompt": system_prompt,
                "gold_answer": gold_answer,
            })
    return examples


def tokenize_examples(
    examples: list[dict[str, Any]], tokenizer: Any, batch_size: int, enable_thinking: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(examples), desc="Rendering + tokenizing", unit="example")
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["question"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for row in batch
        ]
        prompt_ids = tokenizer(prompts, add_special_tokens=False, padding=False)["input_ids"]
        sliced_input_ids = tokenizer(
            [row["sliced_input"] for row in batch], add_special_tokens=False, padding=False
        )["input_ids"]
        gold_ids = tokenizer(
            [row["gold_answer"] for row in batch], add_special_tokens=False, padding=False
        )["input_ids"]
        for row, prompt, prompt_tokens, sliced_tokens, gold_tokens in zip(
            batch, prompts, prompt_ids, sliced_input_ids, gold_ids, strict=True
        ):
            prompt_count = len(prompt_tokens)
            gold_count = len(gold_tokens)
            min_generation_tokens = (
                MIN_GENERATION_TOKENS if enable_thinking else NON_THINKING_MIN_GENERATION_TOKENS
            )
            token_headroom = (
                OUTPUT_TOKEN_HEADROOM if enable_thinking else NON_THINKING_OUTPUT_TOKEN_HEADROOM
            )
            token_reserve = THINKING_TOKEN_RESERVE if enable_thinking else NON_THINKING_TOKEN_RESERVE
            max_new_tokens = min(
                MAX_GENERATION_TOKENS_CAP,
                max(
                    min_generation_tokens,
                    math.ceil(gold_count * token_headroom + token_reserve),
                ),
            )
            if max_new_tokens < gold_count:
                raise RuntimeError(f"Generation cap is below gold tokens for {row['dataset_id']}")
            sequence_budget = prompt_count + max_new_tokens
            if sequence_budget > MODEL_NATIVE_MAX_LENGTH:
                raise RuntimeError(
                    f"{row['dataset_id']} needs {sequence_budget:,} tokens, above native context"
                )
            rows.append({
                "no": len(rows) + 1,
                **row,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "sliced_input_tokens": len(sliced_tokens),
                "prompt_tokens_estimate": prompt_count,
                "gold_tokens": gold_count,
                "max_new_tokens": max_new_tokens,
                "sequence_token_budget": sequence_budget,
            })
        progress.update(len(batch))
    progress.close()
    return rows


def build_summary(
    frame: pd.DataFrame,
    dataset_path: Path,
    output_path: Path,
    tokenizer_name: str,
    tokenizer: Any,
    elapsed_seconds: float,
    enable_thinking: bool,
) -> dict[str, Any]:
    prompt_tokens = frame["prompt_tokens_estimate"]
    sliced_input_tokens = frame["sliced_input_tokens"]
    gold_tokens = frame["gold_tokens"]
    sequence_budgets = frame["sequence_token_budget"]
    max_model_length = int(
        min(
            MODEL_NATIVE_MAX_LENGTH,
            math.ceil(sequence_budgets.max() / CONTEXT_ALIGNMENT) * CONTEXT_ALIGNMENT,
        )
    )
    return {
        "schema_version": 1,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_dataset": str(dataset_path.resolve()),
        "source_dataset_sha256": sha256_file(dataset_path),
        "output_parquet": output_path.name,
        "tokenizer": tokenizer_name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "enable_thinking": enable_thinking,
        "source_rows": int(frame["source_row_no"].nunique()),
        "unique_parent_ids": int(frame["parent_id"].nunique()),
        "section_examples": len(frame),
        "canonical_sections": int(frame["section"].nunique()),
        "empty_examples": int(frame["is_empty"].sum()),
        "prompt_tokens": {
            "total": int(prompt_tokens.sum()),
            "mean": float(prompt_tokens.mean()),
            "p50": int(prompt_tokens.quantile(0.50)),
            "p95": int(prompt_tokens.quantile(0.95)),
            "max": int(prompt_tokens.max()),
        },
        "sliced_input_tokens": {
            "total": int(sliced_input_tokens.sum()),
            "mean": float(sliced_input_tokens.mean()),
            "p50": int(sliced_input_tokens.quantile(0.50)),
            "p95": int(sliced_input_tokens.quantile(0.95)),
            "max": int(sliced_input_tokens.max()),
        },
        "gold_tokens": {
            "total": int(gold_tokens.sum()),
            "mean": float(gold_tokens.mean()),
            "p50": int(gold_tokens.quantile(0.50)),
            "p95": int(gold_tokens.quantile(0.95)),
            "max": int(gold_tokens.max()),
        },
        "generation_budget_tokens_total": int(frame["max_new_tokens"].sum()),
        "sequence_budget_tokens_total": int(sequence_budgets.sum()),
        "sequence_budget_tokens_max": int(sequence_budgets.max()),
        "max_model_length": max_model_length,
        "precompute_elapsed_seconds": elapsed_seconds,
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.upload_existing:
        print(f"Validating existing Parquet: {args.output.resolve()}")
        result = pd.read_parquet(args.output)
        if result.columns.tolist() != OUTPUT_COLUMNS:
            raise RuntimeError("Existing Parquet columns do not match OUTPUT_COLUMNS")
        if result["dataset_id"].duplicated().any():
            raise RuntimeError("Existing Parquet contains duplicate dataset_id values")
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        if summary["section_examples"] != len(result):
            raise RuntimeError("Summary row count does not match existing Parquet")
    else:
        started = time.perf_counter()
        print(f"Loading {args.dataset.resolve()}")
        source = pd.read_parquet(args.dataset)
        print(f"Loading tokenizer {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        examples = build_examples(source)
        expected = len(source) * len(SECTION_GUIDANCE)
        if len(examples) != expected:
            raise RuntimeError(f"Expected {expected} examples, got {len(examples)}")
        rows = tokenize_examples(examples, tokenizer, args.batch_size, args.thinking)
        result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
        elapsed = time.perf_counter() - started
        summary = build_summary(
            result, args.dataset, args.output, args.tokenizer, tokenizer, elapsed, args.thinking
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Parquet: {args.output.resolve()} ({args.output.stat().st_size / 2**20:.1f} MiB)")
    print(f"Summary: {args.summary.resolve()}")
    if args.no_upload:
        return

    os.environ.setdefault("WANDB_DIR", str((Path.cwd() / "wandb").resolve()))
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name="precompute-qwen3-5-4b-sliced-section-eval",
        job_type="dataset-preprocessing",
        config=summary,
    )
    artifact = wandb.Artifact(
        name=args.artifact,
        type="dataset",
        description=(
            "Ready-to-generate Qwen3.5-4B section-sliced evaluation prompts and exact "
            "token budgets derived from sft/test.parquet target_json."
        ),
        metadata=summary,
    )
    artifact.add_file(str(args.output), name=args.output.name)
    artifact.add_file(str(args.summary), name=args.summary.name)
    logged = run.log_artifact(artifact, aliases=["latest"])
    logged.wait()
    print(f"Uploaded artifact: {logged.qualified_name}")
    print(f"Artifact URL: {logged.url}")
    run.finish()


if __name__ == "__main__":
    main()
