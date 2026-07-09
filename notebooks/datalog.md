# Data log — Qwen3.5-9B SFT extractor


## Token-length audit (SFT split, `Haeryz/putusan-structured-extraction`)

Measured on the fully-templated `text` field (system + user putusan body + assistant
31-section JSON) using the model's **inner text tokenizer** (see processor note below).

| percentile | tokens  |
| ---------- | ------- |
| p50        | 34,499  |
| p90        | 74,328  |
| p95        | 90,106  |
| max        | 448,111 |

`max_seq_length = 32768` → `MAX_LENGTH` (90th-pct heuristic, capped) resolves to **32,768**.

### The problem

The 90th-percentile `max_length` heuristic is **invalid for this dataset**: even the
**median (34,499) exceeds the 32,768 cap**, so **more than half of all training examples
are longer than the context window.**

This does not merely truncate context — it corrupts the training signal:

1. Truncation keeps the **front** of the sequence and drops the tail.
2. The assistant JSON target sits at the **end** of the sequence.
3. `train_on_responses_only` computes loss **only** on that trailing JSON.
4. Therefore, for every example over 32,768 tokens (the majority), the JSON target is
   truncated away → all labels become `-100` (ignored) → those steps contribute
   **zero / NaN learning signal**. The majority of the dataset is wasted or corrupt,
   silently.

### Hardware constraints (from the run banner)

- Actual GPU: **`NVIDIA A100-SXM4-40GB`, 39.494 GB usable** — NOT the 80 GB the notebook
  comment assumes. Raising `max_seq_length` to fit p90 (74k) on a 9B model at 40 GB is
  not feasible.
- p95 = 90,106; max = 448,111. Extractive putusan bodies concatenate every verbatim
  span, so they are genuinely very long.

Conclusion: the current config is not a tunable — the whole-document SFT approach does
not fit the data on this hardware.

## Processor / tokenizer note

`Qwen/Qwen3.5-9B` ships **`processor_class: "Qwen3VLProcessor"`** (verified in the repo's
`preprocessor_config.json`; repo also carries `preprocessor_config.json` +
`video_preprocessor_config.json`). So `FastLanguageModel.from_pretrained` returns a
**multimodal processor**, not a plain tokenizer.

- The processor's `__call__` takes **`images` first**, then `text` — calling it
  positionally on a raw string (`tokenizer(t, ...)`) routes the text into
  `image_utils.load_image()` and raises `UnidentifiedImageError` /
  `ValueError: Incorrect image source`.
- Fix used across text-only cells: `text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)`
  pulls the inner `Qwen2Tokenizer`; `getattr` falls back harmlessly if the object is
  already a plain tokenizer.
- Applied in: the token-length measurement cell, the `SFTTrainer` cell
  (`tokenizer = text_tokenizer`), and **all three inference cells** (base-model preview,
  test-split inference, load-adapters cell).
- Second processor gotcha in the inference cells: the `Qwen3VLProcessor`'s
  `apply_chat_template` defaults to `tokenize=False`, so it returns a **str** and ignores
  `return_tensors="pt"` → `str.to("cuda")` raises `AttributeError: 'str' object has no
  attribute 'to'`. Using the inner `text_tokenizer.apply_chat_template(...)` tokenizes and
  returns the tensor as expected. `TextStreamer` is also given `text_tokenizer`.
- `train_on_responses_only` needs no change — it inherits the trainer's tokenizer.

## LoRA target-module note (Qwen3.5 hybrid attention)

`Qwen/Qwen3.5-9B` config: `architectures: ["Qwen3_5ForConditionalGeneration"]`, hybrid
attention with `layer_types` alternating linear/full at `full_attention_interval: 4`
(≈3 of every 4 of the 32 layers are linear attention).

- An explicit `target_modules = ["q_proj","k_proj","v_proj","o_proj", ...]` list only
  matches the ~8 **full-attention** layers; the ~24 linear-attention layers use
  differently-named projections (`in_proj_qkvz`, `in_proj_ba`, `out_proj`) and would get
  **no adapters**.
- Fix: drop the explicit list and use Unsloth's filters — `finetune_vision_layers=False`
  (freeze vision tower), `finetune_language_layers=True`,
  `finetune_attention_modules=True`, `finetune_mlp_modules=True` — so adapters attach to
  the linear-attention projections too. Verify with a `Counter` over `lora_A` param names
  (use `name.split(".")[-4]` to get the module name).
