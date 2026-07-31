# Multi-model putusan SFT

This package fine-tunes three Hugging Face models in this fixed order:

1. `Qwen/Qwen3.5-4B`
2. `google/gemma-4-E2B-it`
3. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

One command runs a deep preflight and then launches each model in a fresh
Python process:

```bash
python -m trainer.sft.run_all
```

The separate processes release GPU memory and close the current W&B run before
the next model starts. Each profile also has separate token-length caches,
Trainer checkpoints, final LoRA directories, W&B runs, and artifact names
under `outputs/sft/<model-slug>/`.

## Text-only policy

The model architecture information and processors were checked against the
three Hugging Face repositories.

| Profile | Hub input components | Training policy |
| --- | --- | --- |
| Qwen 3.5 | text, image/video vision encoder | `text_only=True`; vision excluded from LoRA |
| Gemma 4 | text, image/video vision encoder, audio encoder | language attention/MLP LoRA only; vision, audio, and multimodal projectors frozen |
| DeepSeek R1 Distill Qwen 1.5B | text only | language LoRA |

The dataset formatter rejects list-based image/audio content. After LoRA is
attached, the workflow scans trainable parameter names and aborts if a Qwen
vision component or Gemma vision/audio/projector component became trainable.
Gemma's processor-added `<bos>` is removed before training, matching the
official Unsloth Gemma 4 text recipe.

## Setup and preflight

Create the ignored environment file and fill in all values:

```bash
cp trainer/sft/.env.example trainer/sft/.env
bash trainer/sft/download_model.sh
bash trainer/sft/setup_runpod.sh
source .venv/bin/activate
```

`setup_runpod.sh` installs the Gemma-compatible floors
`transformers>=5.5.0`, `trl>=0.28.0`, `unsloth>=2026.4.2`,
`huggingface_hub>=1.5.0`, and `datasets==4.3.0`.

The mini preflight can also be run independently:

```bash
# Fast: versions, credentials, Hub metadata/access, chat templates,
# dataset schema, and W&B authentication.
python -m trainer.sft.preflight

# Deep: all fast checks plus GPU validation, 4-bit load, language-only LoRA,
# tokenization, and one finite forward loss for every model.
python -m trainer.sft.preflight --deep
```

The all-model runner uses the deep check by default. Use
`--quick-preflight` only when weights were already validated, or
`--skip-preflight` when deliberately resuming without repeating it.

Keep `HF_HOME` on persistent RunPod storage.

## Running

Typical unattended run:

```bash
python -m trainer.sft.run_all \
  --save-steps 5 \
  --wandb-project putusan-sft \
  --wandb-run-prefix production
```

The default is one complete training epoch. `--max-steps N` is only an
explicit smoke/debug override and takes precedence over the epoch count.

Run only one profile when debugging:

```bash
python -m trainer.sft --model gemma
```

Accepted profile names are `qwen`, `gemma`, and `deepseek`; their exact Hub
repository names are accepted too. `--allow-non-a100` keeps CUDA mandatory but
disables the A100 name/VRAM guard.

Every run automatically scans all versions of that model's configured W&B
checkpoint collection. It filters by base model, dataset, and dataset config,
compares the highest remote step with the highest complete local
`checkpoint-N`, and atomically downloads W&B only when it is newer. The
existing Trainer resume logic then starts from the selected checkpoint.
No manual artifact download is needed, including after losing the RunPod
volume.

Use `--no-wandb-resume` only when intentionally starting from local state or
from scratch. Every newly saved checkpoint is synchronously committed to W&B
before training proceeds.

The final local output is a LoRA adapter, not a merged base model and not an
automatic Hugging Face upload. Merging/publishing remains an explicit later
operation in `checkpoint.py`.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Three architecture/modality profiles and isolated paths |
| `preflight.py` | Fail-fast environment and real-model smoke test |
| `run_all.py` | Qwen → Gemma → DeepSeek process orchestration |
| `transformer.py` | Unsloth loaders, language LoRA, frozen-tower checks |
| `data.py` | Text-only chat formatting and per-tokenizer context sizing |
| `training.py` | TRL trainer and profile-specific response masking |
| `checkpoint.py` | Atomic checkpoint restore, adapter save, and explicit merge/Hub export |
| `tracking.py` | W&B checkpoint scan/restore lifecycle and artifact uploads |
