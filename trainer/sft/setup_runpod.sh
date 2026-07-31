#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
SETUP_MARKER="${VENV_DIR}/.runpod-setup.sha256"
SETUP_HASH="$(sha256sum "${SCRIPT_DIR}/setup_runpod.sh" | cut -d ' ' -f 1)"
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

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

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  echo "Reusing the existing virtual environment at ${VENV_DIR}."

  if [[ -f "${SETUP_MARKER}" ]] && [[ "$(<"${SETUP_MARKER}")" == "${SETUP_HASH}" ]]; then
    source "${VENV_DIR}/bin/activate"
    echo "RunPod SFT environment is already up to date."
    echo "Activate it with: source ${VENV_DIR}/bin/activate"
    echo "Then run: python -m trainer.sft.run_all"
    exit 0
  fi
elif [[ -e "${VENV_DIR}" ]]; then
  echo "${VENV_DIR} exists but is not a valid virtual environment." >&2
  echo "Move or remove it, then run this script again." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --python 3.12 "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Match the versions validated by the source notebook.
uv pip install "torch==2.8.0" "triton>=3.3.0" torchvision bitsandbytes "xformers==0.0.32.post2"
uv pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install --no-deps "torchcodec==0.7.0"
uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" "unsloth>=2026.4.2" unsloth_zoo
uv pip install "transformers>=5.5.0,<5.6" "trl>=0.28.0,<0.29" "huggingface_hub>=1.5.0" "datasets==4.3.0"
uv pip install --no-build-isolation flash-linear-attention "causal_conv1d==1.6.0"
uv pip install --no-deps "apache-tvm-ffi==0.1.9" "tilelang==0.1.8" "torchao>=0.16.0"
# Training is text-only for every profile. Do not install librosa: its optional
# numba/llvmlite audio stack can select Python-incompatible legacy releases on
# Python 3.12 RunPod images, and no SFT code path consumes audio.
uv pip install wandb numpy tqdm peft accelerate sentencepiece protobuf timm packaging
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

# Compile kernels and verify each architecture, text-only LoRA policy,
# tokenizer, dataset row, Hub access, W&B login, and one finite forward loss.
python -m trainer.sft.preflight --deep

printf '%s\n' "${SETUP_HASH}" > "${SETUP_MARKER}"

echo
echo "RunPod SFT environment is ready."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
echo "Then run: python -m trainer.sft.run_all"
