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

Formatted conversations are capped at the profile's 49,152-token context.
Oversized rows are truncated from the middle of the user content. This keeps
the instruction marker, the end of the source document, the assistant marker,
and the supervised answer so Unsloth can retain the row for response-only
training. Context coverage is reported but does not stop the run.

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

| Dataset config | 2 GPUs | 3 GPUs |
| --- | ---: | ---: |
| `sft` | 155 steps/epoch, evaluate every 38 | 103, every 25 |
| `sft_sections` | 923 steps/epoch, evaluate every 230 | 616, every 154 |

To enforce a wall-time budget after measuring one training step (`t_step`) and
one complete validation (`T_eval`), compute `T_train = S*t_step`. If validation
may consume at most fraction `f` of total runtime, the largest affordable count
is `floor(f*T_train / ((1-f)*T_eval))`.

### Precomputing token lengths off-pod

Token lengths depend on the model tokenizer and formatted dataset, so each
profile has its own cache. They can be measured on a local CPU before renting
the training pod; CUDA does not accelerate tokenizer work:

```bash
python -m trainer.sft.precompute_lengths
```

The command reads `HF_TOKEN` and `WANDB_API_KEY` from `trainer/sft/.env`,
downloads only the dataset and tokenizer files, measures all three profiles,
and uploads `<slug>-token-lengths:latest` to W&B. Training then downloads the
matching cache during stage 8 and skips measurement.

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
outputs/sft/gemma-4-e2b/lora/
outputs/sft/deepseek-r1-distill-qwen-1-5b/lora/
```

It also waits for matching W&B model artifacts named `qwen3-5-4b-lora`,
`gemma-4-e2b-lora`, and `deepseek-r1-distill-qwen-1-5b-lora`, each with the
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
| `run_all.py` | Qwen → Gemma → DeepSeek process orchestration |
| `transformer.py` | Unsloth loaders, language LoRA, frozen-tower checks |
| `data.py` | Text-only chat formatting and per-tokenizer context sizing |
| `training.py` | TRL trainer and profile-specific response masking |
| `checkpoint.py` | Atomic checkpoint restore, adapter save, and explicit merge/Hub export |
| `merge.py` | Later-session W&B adapter restore and 16-bit merge command |
| `tracking.py` | W&B checkpoint scan/restore lifecycle and artifact uploads |
