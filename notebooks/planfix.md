# How Unsloth's 500k-Context Stack Works — and How This Notebook Uses It

Reference: [Unsloth blog — 500K context length fine-tuning](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning).
Source files inspected: [`unsloth_zoo/tiled_mlp.py`](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/tiled_mlp.py), [`unsloth_zoo/gradient_checkpointing.py`](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/gradient_checkpointing.py), [`unsloth_zoo/loss_utils.py`](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/loss_utils.py).

The stack is three independent memory optimizations. Combined, Unsloth reports >6.4× longer trainable context vs. their previous 80k baseline (500k+ on an H100-80GB for gpt-oss-20b, 290k under QLoRA). Each attacks a different term of the activation-memory budget:

| # | Technique | Attacks | Enabled in this notebook by |
|---|-----------|---------|------------------------------|
| 1 | Offloaded gradient checkpointing | stored per-layer activations | `use_gradient_checkpointing = "unsloth"` in `from_pretrained` |
| 2 | Tiled MLP (Arctic Long Sequence Training) | MLP intermediate activations *within* one layer | `unsloth_tiled_mlp = True` in `from_pretrained` |
| 3 | Fused/chunked cross-entropy | the `[seq × vocab]` logits tensor | automatic when labels are passed; pinned by `UNSLOTH_RETURN_LOGITS = "0"` in the trainer cell |

**Key architectural fact for this notebook:** all three are patched onto the **base model at load time** (or into the loss path at training time). None of them live in `FastLanguageModel.get_peft_model`. That is why bypassing Unsloth's `get_peft_model` with plain `peft.LoraConfig` (which we had to do — see §5) does **not** lose any of the 500k stack, and why the notebook re-verifies each piece with asserts instead of trusting this claim.

Notation used throughout:

- $B$ — micro-batch size (here $B=1$)
- $S$ — sequence length (here capped at $S = 48{,}896$)
- $H$ — hidden size, $I$ — MLP intermediate size, $N$ — number of decoder layers, $V$ — vocabulary size ($V \approx 248{,}000$ for Qwen3.5; its EOS id is 248,046)
- $b$ — bytes per element ($b=2$ for bf16, $b=4$ for fp32)

---

## 1. Offloaded gradient checkpointing (`use_gradient_checkpointing = "unsloth"`)

### The problem

Without checkpointing, backprop must keep every intermediate activation of every layer alive until its backward pass. Per layer this is a small constant $c$ times the hidden-state size (attention scores, projections, MLP intermediates, norms), so total activation memory is

$$
M_{\text{no-ckpt}} \;\approx\; N \cdot c \cdot B \, S \, H \, b .
$$

Standard (vanilla) gradient checkpointing stores only each layer's **input** hidden state and recomputes the rest during backward:

$$
M_{\text{ckpt}} \;\approx\; N \cdot B \, S \, H \, b
\qquad\text{(one boundary tensor per layer, still on GPU).}
$$

At long context even that is large. For this notebook's shape ($N \cdot S \cdot H \cdot 2$ bytes at $S=48{,}896$), every layer boundary tensor is $S \cdot H \cdot 2$ bytes — the blog notes a *single decoder layer's activations can exceed 2 GB* in long-context scenarios.

### What Unsloth does differently

`patch_unsloth_smart_gradient_checkpointing()` replaces `torch.utils.checkpoint.CheckpointFunction` with its own `torch.autograd.Function`, **`UnslothCheckpointFunction`**, which offloads each boundary tensor to **pinned CPU RAM** instead of keeping it on the GPU:

- CPU buffers are allocated pinned: `torch.empty(..., device="cpu", pin_memory=True)`, so DMA transfers can run asynchronously (`x.copy_(arg, non_blocking=True)`).
- Two CUDA streams per GPU — `MAIN_STREAM` (compute) and `EXTRA_STREAM` (copies). The copy stream waits for compute (`EXTRA_STREAM.wait_stream(MAIN_STREAM)`), then the device→host copy overlaps with the next layer's compute.
- During backward, the tensor is prefetched host→device on `EXTRA_STREAM` while `MAIN_STREAM` is still busy with the previous layer's backward, then `MAIN_STREAM.wait_stream(EXTRA_STREAM)` before recomputation.
- **Double buffering**: with enough VRAM headroom, a second GPU staging buffer (`GPU_BUFFERS_B`, with per-buffer CUDA events) lets the H2D transfer of layer $k{-}1$'s activations overlap with recomputation of layer $k$. This is exactly the line the training run printed: `Unsloth: Double buffering enabled (parallel H2D + compute) for backward pass.` (Disable with `UNSLOTH_DISABLE_DOUBLE_BUFFER=1`.)
- Tensors smaller than a `MINIMUM_SIZE` threshold (2 MB) are not offloaded — the transfer wouldn't pay for itself.

GPU-resident activation memory therefore drops from $N$ boundary tensors to roughly the **working set of one layer** (the layer currently being recomputed, plus one staging buffer):

$$
M_{\text{unsloth-ckpt}}^{\text{GPU}} \;\approx\; \mathcal{O}(1) \cdot B\,S\,H\,b
\qquad\text{instead of}\qquad
\mathcal{O}(N) \cdot B\,S\,H\,b ,
$$

with the $N$ tensors parked in CPU RAM. Because every transfer is hidden behind compute on the second stream, the measured cost is *at most 0.1% training overhead* (down from 1–3% in their April 2024 version). Note this is the overhead of *offloading relative to ordinary checkpointing*; checkpointing itself always adds one extra forward pass per layer (≈ $\tfrac{1}{3}$ more forward FLOPs).

### In this notebook

- Enabled once in `FastLanguageModel.from_pretrained(..., use_gradient_checkpointing="unsloth")`. Because the patch replaces `torch.utils.checkpoint` machinery globally and marks boundaries on the model's layers, it survives the plain-PEFT wrap.
- The LoRA cell asserts it is still active after wrapping: `base_lm.is_gradient_checkpointing` / `gradient_checkpointing`.
- `model.enable_input_require_grads()` is called after `peft.get_peft_model` — under checkpointing with frozen (4-bit) embeddings, some input to the checkpointed segment must require grad or backward never reaches the LoRA weights. Unsloth's own wrapper does this internally; going through plain PEFT we do it explicitly.

---

## 2. Tiled MLP — Arctic Long Sequence Training (`unsloth_tiled_mlp = True`)

### The problem

Inside one transformer MLP (SwiGLU), the forward computes `down_proj(act(gate_proj(x)) * up_proj(x))`. The intermediate tensors have shape $[B\,S,\; I]$ with $I \gg H$ (typically $I \approx 3H$–$5H$). Even with gradient checkpointing, this peak occurs *during the recomputation* of a single layer, so it is the binding constraint at long context:

$$
M_{\text{MLP}} \;\approx\; 3 \, B\,S\,I\,b
\qquad (\text{gate, up, and activation intermediates alive simultaneously}).
$$

### What Unsloth does (from `tiled_mlp.py`)

The patcher walks the model and, for every module whose name ends in `.mlp`, `.ffn`, `.feed_forward`, `.ff`, `.densereludense`, or `.block_sparse_moe` (plus Nemotron `.mixer`), saves the original forward (`mlp_module._unsloth_forward = mlp_module.__class__.forward`) and rebinds `forward` to a custom autograd Function, `TiledMLP.apply(...)`.

**Tiling (Arctic mode, the default):** the input is flattened to $[B\,S,\; H]$ and split along the token dimension into tiles of `chunk_size = max(1, H)` tokens — i.e., the tile length *equals the hidden size*:

$$
n_{\text{shards}} \;=\; \left\lceil \frac{B\,S}{H} \right\rceil ,
$$

which is exactly what the notebook's load cell prints ("Arctic tile size = hidden_size"). There is also a `target_gb` mode that instead solves for the largest tile that fits a VRAM budget $G$ (bytes-per-token model from the source, $\text{hd}=H$, $\text{mlp}=I$):

$$
\text{max\_flat\_qlen} \;=\; \left\lceil \frac{G \cdot 1024^3 / b \;-\; 3\,H\,I}{10\,H + 3\,I + H} \right\rceil .
$$

**Memory effect:** each tile's forward materializes only $[H,\, I]$-sized intermediates instead of $[B S,\, I]$:

$$
M_{\text{MLP}}^{\text{tiled}} \;\approx\; 3\,H\,I\,b \;+\; \underbrace{B\,S\,H\,b}_{\text{output buffer}} ,
\qquad
\frac{M_{\text{MLP}}^{\text{tiled}}}{M_{\text{MLP}}} \;\approx\; \frac{H}{B\,S} \;(\text{for the intermediates}) .
$$

For this notebook's $S = 48{,}896$ with tile length $H$, the intermediates shrink by a factor of $\sim S/H$ (dozens of tiles — the load cell prints the exact count as $\lceil S/H \rceil$). Unsloth measures the end-to-end effect as **~40% lower total VRAM** at long context.

**Autograd mechanics (why "~3 forwards, 1 backward"):** `TiledMLP` is a custom `torch.autograd.Function`, *not* `torch.utils.checkpoint`:

- *Forward:* runs each tile under `torch.no_grad()`, writing results into a pre-allocated output tensor — nothing is saved for backward except the input.
- *Backward:* replays RNG state via `torch.random.fork_rng()`, re-runs the forward **per tile** with grad enabled, and calls `torch.autograd.backward(outputs, grad_output_shard)` per tile, accumulating input grads into a pre-allocated `x_gradients = torch.zeros_like(x)` through `narrow()` slices. LoRA weight grads accumulate across tiles automatically (autograd sums into `.grad`).

Counting passes for one MLP per training step: (1) the no-grad tiled forward, (2) a second forward when the *outer* gradient checkpointing recomputes the layer, (3) a third forward inside `TiledMLP.backward`'s per-tile replay — then one real backward. Hence the blog's "one MLP now performs ~3 forward passes and 1 backward pass per step", and the measured cost of roughly **1.3× step time** on a single GPU in exchange for the memory.

### In this notebook

- Enabled in `from_pretrained(..., unsloth_tiled_mlp=True)`.
- The load cell verifies the patch reached **every** language block by counting modules ending in `.mlp` that carry `_original_forward`, and compares against `text_config.num_hidden_layers`.
- The LoRA cell re-runs the same count **after** the plain-PEFT wrap. This works because the tiled forward calls `self.gate_proj(x)`, `self.up_proj(x)`, `self.down_proj(...)` *by attribute*: when PEFT swaps those `Linear4bit` submodules for `lora.Linear4bit` wrappers, the tiled forward transparently executes base + LoRA per tile. The patch itself sits on the *parent* MLP module, which PEFT never replaces.

---

## 3. Fused & chunked cross-entropy (automatic; pinned via `UNSLOTH_RETURN_LOGITS=0`)

### The problem — and exactly how our OOM happened

The LM head projects $[B S,\, H] \to [B S,\, V]$. The logits tensor alone is

$$
M_{\text{logits}} \;=\; B\,S\,V\,b .
$$

For this notebook at full context in bf16:

$$
M_{\text{logits}}^{\text{bf16}} = 48{,}896 \times 248{,}000 \times 2 \;\approx\; 2.43 \times 10^{10} \text{ B} \;\approx\; 22.6 \text{ GiB},
$$

and if anything upcasts it to fp32 it doubles to $\approx 45$ GiB. That is precisely the crash observed at the step-20 evaluation: `accelerate`'s mixed-precision wrapper (`convert_to_fp32`) tried to allocate **42.44 GiB**. Back-solving

$$
S \;=\; \frac{42.44 \times 2^{30}}{V \cdot 4} \;=\; \frac{4.557\times 10^{10}}{248{,}000 \times 4} \;\approx\; 45{,}900 \text{ tokens},
$$

i.e., the eval batch it died on was a ~45.9k-token document — consistent with the p95-length validation examples. The GPU only has 39.49 GiB total; the allocation could never succeed.

### What Unsloth does (from `loss_utils.py`)

`fused_linear_cross_entropy(hidden_states, lm_weight, labels, num_items_in_batch, ...)` never materializes the logits. It delegates to Apple's [`cut_cross_entropy`](https://github.com/apple/ml-cross-entropy) `linear_cross_entropy` kernel, which fuses the LM-head matmul with the log-sum-exp/CE reduction: logits are computed **blockwise in SRAM/registers** and reduced immediately, so global memory only ever holds per-token losses ($B S \times 4$ bytes — with $S{=}48{,}896$ that's ~0.2 MB instead of ~22.6 GiB):

$$
\frac{M_{\text{fused}}}{M_{\text{logits}}} \;\approx\; \frac{b_{\text{fp32}}}{V \cdot b} \;=\; \frac{4}{248{,}000 \times 2} \;\approx\; 8 \times 10^{-6}.
$$

Details from the source: `shift=True` performs the causal shift ($\text{labels}[..., {:}{-1}] = \text{labels}[..., 1{:}]$) inside the kernel; the loss is normalized as $\mathcal{L} = \mathcal{L}_{\text{sum}} / \texttt{num\_items\_in\_batch}$ so gradient accumulation across micro-batches stays exact; chunk/block sizes are chosen at runtime from available VRAM ("dynamic sequence chunking" — blog: *60% lower VRAM use with 3.2× longer context*, no accuracy loss, fp32 upcasting handled inside the kernel). When the fused path is active, the model's forward returns an `EmptyLogits` placeholder instead of a logits tensor; Unsloth patches `accelerate`'s `recursively_apply` (visible in our traceback as `import_fixes.py`) so the fp32-conversion wrapper skips it. Fallback: if `cut_cross_entropy` is unavailable or `ignore_index != -100`, it reverts to standard chunked PyTorch CE.

### In this notebook

Two settings in the trainer cell close the eval loophole that caused the OOM:

- `os.environ["UNSLOTH_RETURN_LOGITS"] = "0"` — tells Unsloth's patched forward to take the fused path (return `EmptyLogits`) whenever labels are present, in evaluation as well as training.
- `prediction_loss_only = True` in `SFTConfig` — the eval loop keeps only the scalar loss; even if some path did surface logits, they are dropped before `accelerate` gathers/upcasts anything.

---

## 4. Putting the budget together for this exact run

A100-SXM4-40GB ($39.49$ GiB usable), Qwen3.5-4B under QLoRA, $B{=}1$, $S \le 48{,}896$:

$$
M_{\text{total}} \;=\;
\underbrace{M_{\text{weights}}^{\text{4-bit}}}_{\approx\, P \cdot 0.5\,\text{B} \,+\, \text{quant overhead}}
+ \underbrace{M_{\text{LoRA}} + M_{\text{Adam8bit}}}_{64.9\text{M params: adapters, grads, 8-bit moments}}
+ \underbrace{M_{\text{act}}^{\text{GPU}}}_{\substack{\mathcal{O}(1)\,S H b\ \text{(offloaded ckpt)} \\ +\, 3 H I b\ \text{(tiled MLP)}}}
+ \underbrace{M_{\text{loss}}}_{\approx S \cdot 4\,\text{B (fused CE)}}
$$

- 4-bit base of a 4.27B-parameter model ≈ 2.7 GiB + quantization state; LoRA trains 64,929,792 params (1.52%).
- Each stack member removes what would otherwise be the dominant term: without offloaded checkpointing, $N$ boundary tensors of $S H b$; without tiled MLP, $3 S I b$ per recomputed layer; without fused CE, $S V b$ (the ~22.6 GiB / 42.4 GiB term that actually killed the first run).
- The observed steady-state training footprint was ~25 GiB allocated — the stack is what makes $S \approx 49$k fit on 40 GB at all.

Observed throughput math (why run shape matters):

$$
t_{\text{train}} \approx \frac{100 \text{ steps}}{21/35.7\,\text{min}^{-1}} \approx 2.8\ \text{h}, \qquad
t_{\text{eval}} \approx \frac{1{,}818}{0.76\ \text{it/s}} \approx 40\ \text{min per eval} .
$$

With `eval_steps = 20`, five evals add $\approx 3.3$ h — more than the training itself — while `max_steps = 100` covers only $\frac{100 \times 8}{14{,}178} \approx 5.6\%$ of one epoch. A full epoch at this speed is $\lceil 14{,}178/8 \rceil = 1{,}773$ steps $\approx 50$ h of pure training. Any change to epochs/eval cadence should start from these measured numbers.

---

## 5. Why this notebook applies LoRA with plain PEFT (and why that's safe)

On the installed git build, two Unsloth defects block its own LoRA API for hybrid Qwen3.5 (8 full-attention + 24 linear-attention layers):

1. `from_pretrained` **auto-attaches** LoRA with a default target regex covering only `q/k/v/o_proj` + MLP → the 24 linear-attention layers' projections (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) get **no adapters** (observed: `q/k/v/o_proj: 8` each, `gate/up/down_proj: 32` each).
2. `FastLanguageModel.get_peft_model` intersects even an **explicit** `target_modules` list with its `finetune_attention` filter, which only knows `q/k/v/o` names — it prints *"Explicit target_modules are constrained by the finetune_(vision|language|attention|mlp) filters"* and silently drops the linear-attention modules again.

The notebook therefore: detects incomplete coverage → `model.unload()` (adapters are untrained, nothing lost) → enumerates every `nn.Linear` leaf name outside `lm_head`/embeddings/vision → applies `peft.LoraConfig` + `peft.get_peft_model` directly (honors the list verbatim) → `enable_input_require_grads()` → asserts $r{=}32$, $\alpha{=}32$, linear-attention coverage, tiled-MLP patch count, and gradient checkpointing.

What is genuinely traded away: Unsloth's fast-LoRA kernel dispatch for the adapter matmuls (a speed optimization in the same code path that drops the linear-attention targets). The 500k-context stack — §1, §2, §3 — is untouched, because none of it lives in `get_peft_model`. The asserts make this checkable on every run rather than an article of faith.
