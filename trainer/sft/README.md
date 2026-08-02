# Multi-model putusan SFT

This package retains profiles for three Hugging Face models. Qwen training is
complete, so the sequential runner trains only the outstanding models:

1. `google/gemma-4-E2B-it`
2. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

One command runs a fast preflight and then launches each model in a fresh
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

The runner uses the fast metadata/service check by default so it does not load
every model twice. Use `--deep-preflight` for a full forward smoke test, or
`--skip-preflight` when deliberately resuming without repeating checks.

Keep `HF_HOME` on persistent RunPod storage.

## Running

Run one model from environment validation through a complete epoch, local LoRA
save, and confirmed W&B adapter upload (no merge):

```bash
python trainer/sft/main.py --modelname qwen
python trainer/sft/main.py --modelname gemma
python trainer/sft/main.py --modelname deepseek
```

For a shorter half-epoch run, add `--half-epoch`:

```bash
python trainer/sft/main.py --modelname qwen --half-epoch
```

`--half-epoch` processes approximately half of the shuffled training split and
is mutually exclusive with `--num-train-epochs`. It still performs evaluation,
checkpointing, final LoRA saving, and the confirmed W&B upload.

The equivalent package form is
`python -m trainer.sft --modelname <qwen|gemma|deepseek>`. The command loads
`trainer/sft/.env`, auto-detects and uses all visible GPUs, restores the
matching cache/checkpoint from `Sinergi-training`, trains only the selected
model, saves its LoRA under `outputs/sft/<model-slug>/lora/`, waits for the
`<model-slug>-lora:latest` W&B upload, and exits without merging.

For Gemma and DeepSeek, every whole-document row is expanded into the same 31
per-section units used by the Qwen evaluation notebook. A unit contains only
that section's gold source spans in `<span>` blocks and its one-section JSON
answer. The measured reference artifact has combined prompt+gold p50 530, p95
8,724, and 94.57% coverage at 8,192 tokens. Both profiles therefore use an
8,192-token cap while selecting their tokenizer-specific p95 at runtime.

Training has no persistent inference KV cache. Shorter units instead reduce
attention/recurrent activations and padding. Batches are grouped by length.

| Model | Micro-batch/GPU | Accumulation | Samples/step/GPU | Conservative peak |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B | 17 | 1 | 17 | 69.91 GiB |
| DeepSeek 1.5B | 24 | 1 | 24 | 68.77 GiB |

The safe limit is 70.20 GiB: 90% of the minimum accepted 78 GiB device. The
budget keeps embeddings and frozen modality towers in BF16, budgets remaining
linear weights at 0.625 byte/parameter for NF4 metadata, adds 2 GiB for
LoRA/gradients/8-bit Adam, every BF16 layer boundary times a 2.5
forward/backward workspace factor, 4 GiB fused-loss workspace, 8 GiB runtime
reserve, and then 25% headroom. Batch 18 for Gemma estimates 72.48 GiB and
batch 25 for DeepSeek 70.83 GiB,
so both are rejected. Automatic first-step batch fallback is also enabled.

Typical unattended run:

```bash
python -m trainer.sft.run_all \
  --wandb-project Sinergi-training \
  --wandb-run-prefix production
```

The default is one complete training epoch. `--max-steps N` is only an
explicit smoke/debug override and takes precedence over the epoch count.
The production defaults evaluate every 38 optimizer steps and save/upload a
resumable checkpoint every 50 steps. Pass `--evaluations-per-epoch K` to
replace the fixed evaluation interval with an automatically calculated one.

Run only one profile when debugging:

```bash
python -m trainer.sft --model gemma
```

Both entry points auto-detect and use every CUDA GPU visible inside the running
machine. RunPod marketplace availability is not visible to the process: if the
pod is assigned two GPUs, the job uses two; if it is assigned three, the job
uses three. Use `--gpu-count N` to select fewer visible GPUs explicitly.

List every option, its default, the effective-batch formula, and examples with:

```bash
python -m trainer.sft --help
python -m trainer.sft.run_all --help
```

Accepted profile names are `qwen`, `gemma`, and `deepseek`; their exact Hub
repository names are accepted too. `--allow-non-a100` keeps CUDA mandatory but
disables the A100 name/VRAM guard.

### Choosing evaluation frequency

For `N` training rows, per-device batch `B`, `G` GPUs, and accumulation `A`,
the optimizer steps per epoch are approximately `S = ceil(N / (B*G*A))`.
For `K` validations per epoch, use `eval_steps = floor(S/K)`. The fixed
production default is 38; optional `--evaluations-per-epoch 4` resolves as:

The exact cadence is resolved after the 31-way expansion because Gemma and
DeepSeek use different micro-batches. Prefer `--evaluations-per-epoch 4` over
a hard-coded interval.

To enforce a wall-time budget after measuring one training step (`t_step`) and
one complete validation (`T_eval`), compute `T_train = S*t_step`. If validation
may consume at most fraction `f` of total runtime, the largest affordable count
is `floor(f*T_train / ((1-f)*T_eval))`.

### Preparing the complete dataset off-pod

Run this locally before renting the training pod:

```bash
python -m trainer.sft.precompute_dataset
```

The command reads `HF_TOKEN` and `WANDB_API_KEY` from `trainer/sft/.env`. For
Gemma and DeepSeek independently it performs the complete CPU preparation:
31-way section slicing, chat-template rendering, length measurement, 8,192
truncation, tokenization, and response-only label masking. It saves train and
validation `input_ids`/`labels` and uploads the model-specific
`<slug>-section-sliced-prepared-sft:latest` W&B artifact.

On the A100 pod, `run_all` downloads those artifacts and passes their IDs and
labels directly to TRL with `skip_prepare_dataset=True`. It does not load the
raw Hugging Face dataset and does not slice, render, measure, truncate,
tokenize, or mask it again. Only artifact download, model loading, and training
remain. If an artifact is missing or incompatible, the trainer clearly reports
that and falls back to local preparation rather than silently using bad IDs.

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

## Outputs and later-session merging

Training writes final LoRA adapters to these profile-specific directories:

```text
outputs/sft/qwen3-5-4b/lora/
outputs/sft/gemma-4-e2b/section-sliced/lora/
outputs/sft/deepseek-r1-distill-qwen-1-5b/section-sliced/lora/
```

It also waits for matching W&B model artifacts named `qwen3-5-4b-lora`,
`gemma-4-e2b-section-sliced-lora`, and
`deepseek-r1-distill-qwen-1-5b-section-sliced-lora`, each with the
`latest` alias. Do not terminate the training pod until it prints
`W&B artifact uploaded` for the last model.

On a later pod, restore and merge one adapter with:

```bash
export WANDB_ENTITY="your-wandb-entity"
python -m trainer.sft.merge --model qwen
python -m trainer.sft.merge --model gemma
python -m trainer.sft.merge --model deepseek
```

The command prefers a local LoRA directory and otherwise downloads its final
artifact from W&B. Merged 16-bit model directories are written under
`outputs/sft/<model-slug>/merged-16bit/`. These merged weights are local and
large, so put `outputs/` on a persistent RunPod volume or upload each merged
directory before terminating that pod.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Three architecture/modality profiles and isolated paths |
| `preflight.py` | Fail-fast environment and real-model smoke test |
| `run_all.py` | Gemma then DeepSeek process orchestration |
| `transformer.py` | Unsloth loaders, language LoRA, frozen-tower checks |
| `section_slicing.py` | Notebook-equivalent per-section gold-span examples |
| `data.py` | Slicing, text-only formatting, and context sizing |
| `memory.py` | Conservative A100 memory calculation and fail-fast guard |
| `training.py` | TRL trainer and profile-specific response masking |
| `checkpoint.py` | Atomic checkpoint restore, adapter save, and explicit merge/Hub export |
| `merge.py` | Later-session W&B adapter restore and 16-bit merge command |
| `tracking.py` | W&B checkpoint scan/restore lifecycle and artifact uploads |
