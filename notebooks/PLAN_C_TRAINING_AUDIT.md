# Plan C Training and W&B Audit

Date: 2026-07-10

## Binary answers

| Question | Answer |
|---|---|
| Does optimizer-based training code exist? | **YES** |
| Does faithful, complete Plan C training exist? | **NO** |
| Does W&B receive manually logged loss values? | **YES** |
| Does the notebook define proper W&B loss graphs? | **NO** |
| Does W&B monitor weights, gradients, or the graph? | **NO** |

## Exact training locations

- Stage 1, cell 18: `scaled.backward()` and `optimizer.step()`.
- Stage 2, cell 20: structured-loss backpropagation and `optimizer.step()`.
- Stage 3, cell 22: cached-representation backpropagation, QLoRA/projector
  surrogate backpropagation, and `optimizer.step()`.
- Cell 30 invokes enabled stages through `STAGE_PLAN` and `stage_fn()`.

## Loss history and graphs

The notebook manually logs these training-history keys:

- `stage1/loss`
- `stage2/loss`
- `stage2/semicrf`
- `stage2/boundary`
- `stage2/fine`
- `stage2/coarse`
- `stage2/presence`
- `stage3/total`
- `stage3/semicrf`
- `stage3/boundary`
- `stage3/fine`
- `stage3/coarse`
- `stage3/presence`

Validation keys include Stage 1 coarse/fine macro-F1 and Stage 2/3 span
macro-F1. W&B may auto-create panels for raw history keys, but the notebook does
not define a dashboard or loss charts. It has no `wandb.define_metric(...)`, no
shared `train/loss`, and no binding that makes `global_step` the step axis.

The logged loss values are also incorrect as aggregate training measurements:

- Stage 1 logs only the final chunk's `loss` at an optimizer update.
- Stage 2 logs only the final document's loss/components in the accumulation
  group.
- Stage 3 logs only the final document's loss/components in the accumulation
  group.

They are not accumulated-batch means, so any resulting W&B loss curves are
misleading.

## What W&B otherwise tracks

While `wandb.init()` is active, W&B automatically samples available CPU, RAM,
disk I/O, network, and NVIDIA GPU utilization/memory/temperature/power metrics,
and captures console logs. The notebook also sends configuration, summary
values, validation/baseline/gate scalars, and tokenization, embedding-cache, and
checkpoint Artifacts.

The notebook explicitly sets `WANDB_WATCH=false` and never calls
`wandb.watch(...)` or `RUN.watch(...)`. It therefore does not log weight
histograms, gradient histograms, per-layer gradient norms, activations, or the
PyTorch graph.

## Why this is not faithful Plan C training

The critical defects are:

1. Units are constructed from gold `sections_json` spans, leaking target
   boundaries into the model input.
2. No label-free preprocessing path exists for unseen text.
3. Round-trip and truncation promotion gates are hardcoded `True`.
4. A2-A5, component ablations, and three reporting seeds are not orchestrated.
5. Best checkpoints are saved but not restored between stages.
6. Stage 1 weights every chunk as a document, so source balancing is wrong.
7. The Stage 2 embedding Artifact is uploaded but not restored on resume.

## Bottom line

**Training loops exist: YES. Complete Plan C training: NO. Proper W&B loss
graphs: NO. Automatic model/gradient monitoring: NO.**
