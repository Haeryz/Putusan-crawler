# Qwen 3.5 per-section SFT

This package is the maintainable Python version of
`notebooks/Qwen3_5_(4B)(1)fsadfhasdkfd (1).ipynb`. It trains a language-only
`Qwen/Qwen3.5-4B` extractor on the `sft_sections` dataset with 4-bit QLoRA
and response-only supervision. Its default performance profile targets two
A100 80GB GPUs on one node.

## Module map

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed model, data, and training defaults |
| `data.py` | Dataset splits, chat formatting, length cache, context sizing |
| `transformer.py` | A100 validation, Unsloth loading, complete Qwen LoRA coverage |
| `training.py` | TRL trainer construction and response masking |
| `inference.py` | Structured-output generation |
| `checkpoint.py` | Adapter load/save and explicit merged/Hub export |
| `tracking.py` | W&B run lifecycle and completed model-artifact upload |
| `main.py` | Explicit start-to-finish workflow and automatic DDP launch |
| `pipeline.py` | Compatibility import for `main.run_training` |
| `cli.py` | Command-line overrides |

Heavy ML imports are local to their functions, so configuration and pure data
helpers remain importable on machines without the GPU stack.

## Setup

Use the pinned GPU installation cell in the source notebook (Torch 2.8,
Transformers 5.2, TRL 0.22.2, Unsloth, flash-linear-attention, and
causal-conv1d). Authenticate Hugging Face as needed and set `WANDB_API_KEY`
when using the default `report_to="wandb"`. Never store tokens in this repo.

## Train

From the repository root:

```powershell
uv run python trainer/sft/main.py --max-steps 100
```

Useful overrides:

```powershell
uv run python trainer/sft/main.py `
  --model Qwen/Qwen3.5-4B `
  --dataset Haeryz/putusan-structured-extraction `
  --dataset-config sft_sections `
  --max-seq-length 49152 `
  --per-device-batch-size 2 `
  --gradient-accumulation-steps 2 `
  --output-dir outputs/sft/checkpoints `
  --adapter-dir qwen_extractor_sft_lora `
  --wandb-project putusan-sft `
  --wandb-artifact-name qwen-extractor-sft-lora
```

The default batch is:

```text
2 examples/GPU x 2 GPUs x 2 accumulation steps = 8 examples/update
```

This preserves the old single-A100-40GB profile's effective batch of 8, but
uses both GPUs and the larger memory to replace eight serial micro-batches
with two. Direct Python execution automatically relaunches `main.py` with
two local DDP workers. The child processes still validate that the expected
two-worker environment exists.

The hardware guard requires two visible A100 GPUs with at least 78 GiB each
(the binary capacity reported for an A100 80GB). `--allow-non-a100` bypasses
the model-name and VRAM check, but the two-process DDP launch remains required.

Long examples can make memory usage vary. If micro-batch 2 is too large on a
particular software build, retain the two-GPU speedup and effective batch with:

```powershell
uv run torchrun --standalone --nproc_per_node=2 -m trainer.sft `
  --per-device-batch-size 1 `
  --gradient-accumulation-steps 4
```

The pipeline caches lengths at `outputs/sft/cache/token_lengths.npy`, chooses
the 95th percentile rounded up to 256 tokens, caps it at 49,152, evaluates
every 20 steps, and saves Stage-1 adapters to `qwen_extractor_sft_lora`.
Only rank 0 measures and writes the shared length cache and saves adapters.
Rank 0 also uploads the saved adapter directory to W&B and waits for the
artifact commit before reporting success.

The launch strategy follows the
[Unsloth DDP guide](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp).
The effective-batch calculation follows the
[Transformers Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer).

For the local-VS-Code/remote-RunPod workflow, follow
[`RUNPOD.md`](RUNPOD.md).

## Use individual pieces

```python
from trainer.sft.config import RunConfig
from trainer.sft.data import format_dataset, load_splits
from trainer.sft.transformer import prepare_model

config = RunConfig()
model, tokenizer = prepare_model(config.model)
train, validation = load_splits(config.data)
train = format_dataset(train, tokenizer)
```

Pass only system and user turns to `trainer.sft.inference.generate`.
Adapter loading, publishing, and merged-model export are available from
`trainer.sft.checkpoint`. Hub publishing remains an explicit action, so a
training run cannot accidentally push artifacts.
