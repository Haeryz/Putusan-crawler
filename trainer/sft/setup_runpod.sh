#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# The 4-bit warm load below reads the cache download_model.sh wrote. Without the
# same HF_HOME it would silently re-download ~9.3 GB onto the container disk.
if [[ -z "${HF_HOME:-}" ]]; then
  echo "HF_HOME is not set, so the weights cached by download_model.sh would"
  echo "not be found and would re-download onto the container disk. Export it:"
  echo
  echo "    export HF_HOME=/workspace/.cache/huggingface"
  echo
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.12 .venv
source .venv/bin/activate

# Match the versions validated by the source notebook.
uv pip install "torch==2.8.0" "triton>=3.3.0" torchvision bitsandbytes "xformers==0.0.32.post2"
uv pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install --no-deps "torchcodec==0.7.0"
uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" "trl==0.22.2" unsloth unsloth_zoo
uv pip install "transformers==5.2.0"
uv pip install --no-build-isolation flash-linear-attention "causal_conv1d==1.6.0"
uv pip install --no-deps "apache-tvm-ffi==0.1.9" "tilelang==0.1.8" "torchao>=0.16.0"
# hf_xet is the native transport for Xet-backed repos like Qwen3.5-4B; without
# it huggingface_hub warns and falls back to plain HTTP through the xet-bridge.
uv pip install datasets wandb numpy tqdm peft hf_xet
uv pip install -e .

python - <<'PY'
import torch

print(f"torch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"visible GPUs={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    gib = properties.total_memory / 1024**3
    print(f"GPU {index}: {properties.name}, {gib:.2f} GiB")
if torch.cuda.device_count() < 1:
    raise SystemExit("The SFT profile requires one visible GPU.")
PY

# Load the cached weights once, in 4-bit, so the silent Unsloth import and its
# Triton kernel compile happen here rather than at stage 3 of a training run.
# This also fails fast if the checkpoint cannot be quantized on this GPU.
python - <<'PY'
import time

from trainer.sft.config import ModelConfig
from trainer.sft.transformer import load_base_model

config = ModelConfig()
print()
print(f"Warming the 4-bit load of {config.model_name}.")
start = time.monotonic()
model, tokenizer = load_base_model(config)
print(f"Model loaded in 4-bit in {(time.monotonic() - start) / 60:.1f} min.")
PY

echo
echo "RunPod SFT environment is ready."
echo "Activate it with: source ${REPO_ROOT}/.venv/bin/activate"
echo "Then run: cd ${REPO_ROOT}/trainer/sft && python main.py"
