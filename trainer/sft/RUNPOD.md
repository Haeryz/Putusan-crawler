# Run the three-model SFT workflow on RunPod

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

The downloader caches Qwen 3.5 4B, Gemma 4 E2B IT, and DeepSeek R1 Distill
Qwen 1.5B.
Setup installs the compatible GPU stack and ends with:

```bash
python -m trainer.sft.preflight --deep
```

That test validates package versions, credentials, access to all model repos,
the real dataset schema, chat-template response markers, W&B authentication,
GPU availability, 4-bit loading, LoRA placement, frozen multimodal towers,
tokenization, and a finite forward loss for every model. Training does not
begin if any check fails.

## Train all models

Use `tmux` for an unattended job:

```bash
tmux new -s putusan-sft
cd /workspace/Sinergi
source .venv/bin/activate
python -m trainer.sft.run_all --wandb-run-prefix run-001
```

The production default is one complete epoch. Do not pass `--max-steps`
unless intentionally limiting a smoke/debug run.
The default cadence is `--eval-steps 38 --save-steps 50`. Token-length caches
can be prepared before renting the pod with:

```bash
python -m trainer.sft.precompute_lengths
```

Detach with `Ctrl+B`, then `D`; reattach with:

```bash
tmux attach -t putusan-sft
```

The runner performs the deep preflight again by default, then starts Qwen,
Gemma, and DeepSeek in isolated processes. If a model fails, the command exits
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

W&B names use the same slug:

- `<slug>-checkpoint`, aliases `latest` and `step-N`
- `<slug>-lora`, alias `latest`
- `<slug>-token-lengths`, alias `latest`

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
