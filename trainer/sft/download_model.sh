#!/usr/bin/env bash
# Download the Qwen base weights onto the Pod before installing the GPU stack.
#
# Run this first, then setup_runpod.sh. Splitting the download out keeps the
# ~9.3 GB transfer in its own visible step, instead of hiding it behind the
# silent Unsloth import at stage 3 of a training run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${HF_HOME:-}" ]]; then
  echo "HF_HOME is not set. The weights would land on the container disk and"
  echo "be wiped by the next Pod Stop. Export it onto the volume first:"
  echo
  echo "    export HF_HOME=/workspace/.cache/huggingface"
  echo
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# uv builds an ephemeral environment before Python prints anything, so say so
# here rather than leaving the terminal blank for the first half minute.
echo "Preparing the download environment with uv (first run takes ~30s)..."

# The Xet transport stalls on this repo and emits no progress of its own, and it
# measured slower than plain HTTP, so ask for HTTP explicitly. Setting this also
# suppresses the fallback warning that appears when hf_xet is simply absent.
export HF_HUB_DISABLE_XET=1

# --no-project keeps this independent of .venv, so it runs before setup does.
uv run --no-project --with "huggingface_hub" python - <<'PY'
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, constants, snapshot_download

from trainer.sft.config import ModelConfig

model_name = ModelConfig().model_name
print(f"\nDownloading {model_name}.")
print("There is no pre-quantized 4-bit repo for this model, so the full bf16")
print("checkpoint is fetched once and quantized at load time.")

info = HfApi().model_info(model_name, files_metadata=True)
total = sum(sibling.size or 0 for sibling in info.siblings)
print(f"{len(info.siblings)} files, {total / 1e9:.2f} GB total.\n")

# Report real bytes landing on disk. The per-file bars only redraw in place on a
# tty, so on their own they still read as frozen in a piped or logged run.
target = Path(constants.HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}"


def downloaded_bytes() -> int:
    if not target.exists():
        return 0
    size = 0
    for path in target.rglob("*"):
        try:
            if path.is_file():
                size += path.stat().st_size
        except OSError:
            pass
    return size


done = threading.Event()


def report() -> None:
    start = time.monotonic()
    base = downloaded_bytes()
    while not done.wait(15):
        elapsed = time.monotonic() - start
        current = downloaded_bytes()
        rate = (current - base) / 1e6 / elapsed if elapsed else 0.0
        percent = 100 * current / total if total else 0.0
        remaining = (
            f", ~{(total - current) / 1e6 / rate / 60:.0f} min left"
            if rate > 0.1 and current < total
            else ""
        )
        print(
            f"  {current / 1e9:5.2f} / {total / 1e9:.2f} GB "
            f"({percent:5.1f}%) at {rate:5.1f} MB/s{remaining}",
            flush=True,
        )


heartbeat = threading.Thread(target=report, daemon=True)
heartbeat.start()
start = time.monotonic()
try:
    path = snapshot_download(repo_id=model_name, max_workers=8)
finally:
    done.set()
    heartbeat.join(timeout=5)
print(f"\nDone in {(time.monotonic() - start) / 60:.1f} min: {path}")
PY

echo
echo "Weights are cached under ${HF_HOME}."
echo "Next: bash ${SCRIPT_DIR}/setup_runpod.sh"
