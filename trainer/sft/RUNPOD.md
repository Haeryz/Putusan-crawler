# Run the Gemma and DeepSeek SFT workflow on RunPod

Use a persistent volume mounted at `/workspace`; Pod container storage is
erased on Stop. Keep the repository, virtual environment, Hugging Face cache,
outputs, and W&B cache on that volume.

## Prepare the remote workspace

```bash
cd /workspace
git clone YOUR_REPOSITORY_URL Sinergi
cd Sinergi

cp trainer/sft/.env.example trainer/sft/.env
chmod 600 trainer/sft/.env
```

Edit the ignored `.env` with:

```dotenv
HF_TOKEN=...
WANDB_API_KEY=...
HF_HOME=/workspace/.cache/huggingface
```

## Download and validate

```bash
bash trainer/sft/download_model.sh
bash trainer/sft/setup_runpod.sh
source .venv/bin/activate
```

The downloader caches Gemma 4 E2B IT and DeepSeek R1 Distill Qwen 1.5B.
Setup installs the compatible GPU stack and ends with:

```bash
python -m trainer.sft.preflight --deep
```

### Python 3.12 `llvmlite==0.36.0` setup failure

Pull the latest `master` and rerun setup:

```bash
git pull origin master
bash trainer/sft/setup_runpod.sh
```

The SFT workflow is text-only, so setup intentionally omits the optional
`librosa -> numba -> llvmlite` audio dependency chain. A failed setup does not
write its completion marker; rerunning safely reuses the partially created
`.venv` and completes the remaining installation.

That test validates package versions, credentials, access to both model repos,
the real dataset schema, chat-template response markers, W&B authentication,
GPU availability, 4-bit loading, LoRA placement, frozen multimodal towers,
tokenization, and a finite forward loss for both models. Training does not
begin if any check fails.

## Train both outstanding models

Use `tmux` for an unattended job:

```bash
tmux new -s putusan-sft
cd /workspace/Sinergi
source .venv/bin/activate
python -m trainer.sft.run_all --wandb-run-prefix run-001
```

The production default is one complete epoch. Do not pass `--max-steps`
unless intentionally limiting a smoke/debug run.
The default cadence is `--eval-steps 38 --save-steps 50`. Build and upload the
fully tokenized, response-masked train/validation datasets locally before
renting the pod:

```bash
python -m trainer.sft.precompute_dataset
```

The pod then restores model-specific prepared artifacts from W&B and skips all
dataset tokenization. If the startup log says `falling back to local
preparation`, the prepared artifact was not found or did not match the model.
DeepSeek's prepared artifact is ordered with the deterministic
`hard_sections_first_v1` curriculum and is consumed with a sequential sampler;
its batch size, accumulation, LR, optimizer, and LoRA settings are unchanged.
Run the short no-evaluation profile with
`python -m trainer.sft --model deepseek --max-steps 300 --no-eval`.

Detach with `Ctrl+B`, then `D`; reattach with:

```bash
tmux attach -t putusan-sft
```

The runner performs only the fast metadata/service preflight by default, then
starts Gemma and DeepSeek in isolated processes. If a model fails, it exits
immediately and does not start later models. Fix the problem and rerun.

At startup, each model scans every version in its own W&B checkpoint
collection. Only artifacts whose base model, dataset, and dataset config match
the current job are eligible. The workflow compares the highest compatible
W&B step with the highest complete local step, downloads W&B atomically when
it is newer, and resumes Trainer automatically. This also restores training
after the persistent volume is lost; no manual W&B download or directory
copying is required.

To deliberately prevent remote restore for one invocation:

```bash
python -m trainer.sft.run_all --no-wandb-resume
```

For a one-step integration run:

```bash
python -m trainer.sft.run_all \
  --max-steps 1 \
  --eval-steps 1 \
  --save-steps 1 \
  --wandb-run-prefix smoke
```

## Persistent outputs

Each model uses:

```text
outputs/sft/<model-slug>/
├── cache/token_lengths.npy
├── checkpoints/checkpoint-N/
└── lora/
```

For Gemma and DeepSeek, these directories are under an additional
`section-sliced/` component and include `prepared-dataset/manifest.json` plus
the saved Arrow dataset.

W&B names use the section-sliced slug:

- `<slug>-section-sliced-checkpoint`, aliases `latest` and `step-N`
- `<slug>-section-sliced-lora`, alias `latest`
- `<slug>-section-sliced-token-lengths`, alias `latest`
- `<slug>-section-sliced-prepared-sft`, alias `latest`

Training saves LoRA adapters only. It never merges with the base weights and
never uploads to Hugging Face automatically.

After a completed ephemeral training pod has been terminated, a new pod can
download the final LoRA artifacts from W&B and merge them without resuming
training:

```bash
export WANDB_ENTITY="your-wandb-entity"
python -m trainer.sft.merge --model qwen
python -m trainer.sft.merge --model gemma
python -m trainer.sft.merge --model deepseek
```

Wait for `W&B artifact uploaded` before terminating the training pod. A
checkpoint artifact is resumable training state; the `<slug>-lora:latest`
artifact is the completed adapter used by the merge command.

Recommended persistent shell settings:

```bash
export HF_HOME=/workspace/.cache/huggingface
export WANDB_CACHE_DIR=/workspace/.cache/wandb
```

Do not commit `.env`, model weights, checkpoints, adapters, downloaded data,
browser profiles, or W&B logs.
