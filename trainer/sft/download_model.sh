#!/usr/bin/env bash
# Download the Qwen base weights onto the Pod before installing the GPU stack.
#
# Run this first, then setup_runpod.sh. Splitting the download out keeps the
# ~9.3 GB transfer in a step that shows real progress bars, instead of hiding it
# behind the silent Unsloth import at stage 3 of a training run.
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
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# --no-project keeps this independent of .venv, so it runs before setup does.
# hf_xet is the native transport for Xet-backed repos like Qwen3.5-4B; without
# it huggingface_hub falls back to plain HTTP through the xet-bridge.
uv run --no-project --with "huggingface_hub[hf_xet]" python - <<'PY'
import time

from huggingface_hub import snapshot_download

from trainer.sft.config import ModelConfig

model_name = ModelConfig().model_name
print(f"Downloading {model_name}.")
print("There is no pre-quantized 4-bit repo for this model, so the full bf16")
print("checkpoint (~9.3 GB) is fetched once and quantized at load time.")
start = time.monotonic()
path = snapshot_download(repo_id=model_name, max_workers=8)
print(f"Done in {(time.monotonic() - start) / 60:.1f} min: {path}")
PY

echo
echo "Weights are cached under ${HF_HOME}."
echo "Next: bash ${SCRIPT_DIR}/setup_runpod.sh"
