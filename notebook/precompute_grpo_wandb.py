from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebooks" / "dataset" / "grpo"
OUTPUT_DIR = ROOT / "outputs" / "grpo" / "qwen3-5-4b-prepared"
ADAPTER_DIR = OUTPUT_DIR / "source-adapter"

WANDB_ENTITY = "haeriz42069-universitas-muhammadiyah-malang"
WANDB_PROJECT = "Sinergi-training"
SOURCE_ADAPTER = f"{WANDB_ENTITY}/{WANDB_PROJECT}/qwen3-5-4b-lora:v0"
ARTIFACT_NAME = "qwen3-5-4b-grpo-core-prepared"
ARTIFACT_TYPE = "tokenized-dataset"
DATASET_IDENTITY = "Haeryz/putusan-structured-extraction"
DATASET_CONFIG = "grpo"
PREPARATION_VERSION = 1

CORE_SECTIONS = (
    "dakwaan",
    "tuntutan",
    "saksi",
    "terdakwa",
    "fakta_hukum",
    "pertimbangan_hukum",
    "amar_putusan",
    "petunjuk_barang_bukti",
    "surat",
)

SECTION_GUIDANCE = {
    "dakwaan": "Ambil lengkap semua bentuk dakwaan beserta uraian perbuatan dan pasal.",
    "tuntutan": "Ambil lengkap seluruh amar tuntutan bernomor sampai sebelum pembelaan atau dakwaan berikutnya.",
    "saksi": "Ambil seluruh keterangan saksi, termasuk nama, sumpah, butir keterangan, dan tanggapan subjek.",
    "terdakwa": "Ambil keterangan Terdakwa/Para Terdakwa atau Anak sendiri di persidangan.",
    "fakta_hukum": "Ambil daftar fakta hukum dan berhenti sebelum analisis hukum atau unsur.",
    "pertimbangan_hukum": "Ambil seluruh analisis hukum Majelis sampai sebelum MENGADILI.",
    "amar_putusan": "Ambil setiap perintah mulai MENGADILI sampai sebelum Demikianlah diputuskan.",
    "petunjuk_barang_bukti": "Ambil inventaris barang bukti yang diajukan Penuntut Umum.",
    "surat": "Ambil alat bukti surat, dokumen, dan elektronik; jangan campur daftar barang bukti fisik.",
}

CORPUS_ADDENDA = {
    "Anak": "Korpus Anak: subjek adalah Anak, bukan Terdakwa dewasa.",
    "Asusila": "Korpus Asusila/Pidana Biasa: subjek adalah Terdakwa/Para Terdakwa.",
    "TPPO": "Korpus TPPO: pertahankan semua Terdakwa, restitusi, barang bukti, dan amar.",
}


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def system_prompt(section: str, corpus: str) -> str:
    return f"""Ekstrak tepat SATU bagian putusan pengadilan Indonesia: {section}.
Jawab dengan SATU objek JSON tanpa markdown, penjelasan, atau reasoning.
Bentuk wajib: {{"sections": {{"{section}": ["kutipan"]}}, "empty_sections": []}}.
`sections` harus memiliki tepat kunci `{section}`. Semua item harus kutipan verbatim
dan kontigu dari teks sumber. Jangan meringkas, memparafrasekan, memperbaiki OCR,
atau mengarang. Jika tidak ada, gunakan [] dan empty_sections ["{section}"].
Panduan: {SECTION_GUIDANCE[section]}
{CORPUS_ADDENDA[corpus]}"""


def render_document_rows(
    row: dict[str, Any], source_row_no: int, tokenizer: Any
) -> list[dict[str, Any]]:
    sections = json.loads(row["target_json"])["sections"]
    rendered: list[dict[str, Any]] = []
    for section in CORE_SECTIONS:
        spans = [str(value) for value in sections[section]]
        answer = json.dumps(
            {
                "sections": {section: spans},
                "empty_sections": [] if spans else [section],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages = [
            {
                "role": "system",
                "content": system_prompt(section, str(row["corpus"])),
            },
            {
                "role": "user",
                "content": f"Bagian: {section}\n\nTEKS PUTUSAN:\n{row['input_text']}",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered.append(
            {
                "id": f"{row['id']}::{source_row_no}::{section}",
                "parent_id": str(row["id"]),
                "corpus": str(row["corpus"]),
                "section": section,
                "prompt": prompt,
                "answer": answer,
                "input_text": str(row["input_text"]),
                "is_empty": not spans,
            }
        )
    return rendered


def prepare_split(source: Path, destination: Path, tokenizer: Any) -> dict[str, Any]:
    frame = pd.read_parquet(source)
    required = {"id", "corpus", "input_text", "target_json"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{source} is missing columns: {sorted(missing)}")

    prepared: list[dict[str, Any]] = []
    records = frame.to_dict("records")
    for source_row_no, record in enumerate(
        tqdm(records, desc=f"Rendering {source.stem}", unit="document")
    ):
        rows = render_document_rows(record, source_row_no, tokenizer)
        prompts = [row["prompt"] for row in rows]
        answers = [row["answer"] + tokenizer.eos_token for row in rows]
        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answers, add_special_tokens=False)["input_ids"]
        for row, prompt_tokens, answer_tokens in zip(
            rows, prompt_ids, answer_ids, strict=True
        ):
            row["prompt_tokens"] = len(prompt_tokens)
            row["answer_tokens"] = len(answer_tokens)
            row["sequence_tokens"] = len(prompt_tokens) + len(answer_tokens)
            row["source_row_no"] = source_row_no
            prepared.append(row)

    output = pd.DataFrame(prepared)
    if len(output) != len(frame) * len(CORE_SECTIONS):
        raise RuntimeError("Prepared row count does not match documents x core sections")
    if output["id"].duplicated().any():
        raise RuntimeError("Prepared IDs are not unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False, compression="zstd")
    lengths = output["sequence_tokens"].to_numpy(dtype=np.int64)
    return {
        "source_rows": len(frame),
        "prepared_rows": len(output),
        "source_sha256": sha256(source),
        "prepared_sha256": sha256(destination),
        "prepared_bytes": destination.stat().st_size,
        "sequence_tokens": {
            "min": int(lengths.min()),
            "p50": int(np.percentile(lengths, 50)),
            "p90": int(np.percentile(lengths, 90)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
        },
    }


def download_tokenizer() -> tuple[Any, str]:
    import wandb
    from transformers import AutoTokenizer

    artifact = wandb.Api().artifact(SOURCE_ADAPTER, type="model")
    required_files = (
        "adapter/adapter_config.json",
        "adapter/chat_template.jinja",
        "adapter/tokenizer.json",
        "adapter/tokenizer_config.json",
    )
    for name in required_files:
        artifact.get_path(name).download(root=str(ADAPTER_DIR))
    adapter = ADAPTER_DIR / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{SOURCE_ADAPTER} has no adapter/adapter_config.json")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    return tokenizer, artifact.digest


def upload(prepared_dir: Path, manifest: dict[str, Any]) -> str:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="precompute-qwen3-5-4b-grpo-core",
        job_type="dataset-preparation",
        settings=wandb.Settings(console="off"),
        config={
            "dataset": DATASET_IDENTITY,
            "dataset_config": DATASET_CONFIG,
            "source_adapter": SOURCE_ADAPTER,
            "preparation_version": PREPARATION_VERSION,
        },
    )
    artifact = wandb.Artifact(
        ARTIFACT_NAME,
        type=ARTIFACT_TYPE,
        description="Qwen3.5-rendered GRPO prompts with exact token counts",
        metadata=manifest,
    )
    for name in ("train.parquet", "validation.parquet", "manifest.json"):
        artifact.add_file(
            str(prepared_dir / name),
            name=name,
            policy="immutable",
        )
    alias = f"prep-v{PREPARATION_VERSION}"
    print("Logging prepared files to W&B...", flush=True)
    run.log_artifact(artifact, aliases=["latest", alias])
    print("Finishing W&B run to flush and commit the artifact...", flush=True)
    run.finish(exit_code=0)
    reference = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{ARTIFACT_NAME}:latest"
    for attempt in range(60):
        try:
            committed = wandb.Api().artifact(reference, type=ARTIFACT_TYPE)
            if committed.state == "COMMITTED":
                return committed.qualified_name
        except wandb.errors.CommError:
            pass
        time.sleep(2)
    raise RuntimeError(f"W&B artifact did not become committed: {reference}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--upload-existing", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wandb_root = OUTPUT_DIR / "wandb"
    wandb_cache = OUTPUT_DIR / "wandb-cache"
    wandb_data = OUTPUT_DIR / "wandb-data"
    task_temp = OUTPUT_DIR / "tmp"
    wandb_root.mkdir(parents=True, exist_ok=True)
    wandb_cache.mkdir(parents=True, exist_ok=True)
    wandb_data.mkdir(parents=True, exist_ok=True)
    task_temp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_root))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data))
    os.environ["TEMP"] = str(task_temp)
    os.environ["TMP"] = str(task_temp)
    os.environ["TMPDIR"] = str(task_temp)
    tempfile.tempdir = str(task_temp)

    load_env(ROOT / "trainer" / "sft" / ".env")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required")

    import wandb

    if args.upload_existing:
        manifest_path = OUTPUT_DIR / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing prepared manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split_name in ("train", "validation"):
            path = OUTPUT_DIR / f"{split_name}.parquet"
            expected = manifest["splits"][split_name]["prepared_sha256"]
            if sha256(path) != expected:
                raise RuntimeError(f"Prepared {split_name} hash mismatch")
        qualified_name = upload(OUTPUT_DIR, manifest)
        print(f"Committed W&B artifact: {qualified_name}")
        return

    print("Resolving W&B tokenizer artifact...", flush=True)
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    tokenizer, adapter_digest = download_tokenizer()
    print(f"Tokenizer ready: {type(tokenizer).__name__}", flush=True)

    with tempfile.TemporaryDirectory(prefix=".prepare-grpo-", dir=OUTPUT_DIR) as temp:
        temporary = Path(temp)
        split_stats = {
            "train": prepare_split(
                SOURCE_DIR / "train.parquet", temporary / "train.parquet", tokenizer
            ),
            "validation": prepare_split(
                SOURCE_DIR / "val.parquet",
                temporary / "validation.parquet",
                tokenizer,
            ),
        }
        manifest = {
            "preparation_version": PREPARATION_VERSION,
            "dataset": DATASET_IDENTITY,
            "dataset_config": DATASET_CONFIG,
            "source_adapter": SOURCE_ADAPTER,
            "source_adapter_digest": adapter_digest,
            "tokenizer_class": type(tokenizer).__name__,
            "core_sections": list(CORE_SECTIONS),
            "splits": split_stats,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for name in ("train.parquet", "validation.parquet", "manifest.json"):
            destination = OUTPUT_DIR / name
            if destination.exists():
                destination.unlink()
            shutil.move(str(temporary / name), destination)

    if args.no_upload:
        print(json.dumps(manifest, indent=2))
        return
    qualified_name = upload(OUTPUT_DIR, manifest)
    print(f"Committed W&B artifact: {qualified_name}")


if __name__ == "__main__":
    main()
