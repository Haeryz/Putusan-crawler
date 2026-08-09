# Notebook Error-Repair Harness

This harness governs agent work on `notebook/qwen3_5_4b_merge_fp16_evaluate_weave.ipynb`, especially when a Colab/A100 run reports an exception. The objective is to repair the demonstrated failure while preserving working behavior. An error is not permission to rewrite an entire cell, phase, or notebook section.

## Non-negotiable rule

Make the smallest evidence-supported change that fixes the observed error.

Do not replace a whole cell or section merely because doing so is easier than understanding it. Preserve existing cell IDs, phase boundaries, variable names, artifact contracts, progress reporting, output schema, and verified performance settings unless the traceback proves one of them is the cause.

A broad rewrite is allowed only when all of the following are true:

1. The exact failure and root cause have been identified.
2. A localized patch has been shown to be impossible or unsafe.
3. Every affected upstream and downstream contract has been enumerated.
4. The proposed rewrite is explained before it is applied.
5. The user explicitly approves the broader rewrite.

Without those five conditions, patch locally.

## Required debugging workflow

### 1. Capture the failure before editing

Record:

- the complete exception type, message, and traceback;
- the failing cell index, cell ID, and nearest Markdown heading;
- the exact line or call that failed;
- whether the kernel was fresh or carried state from earlier cells;
- GPU name and VRAM from `nvidia-smi` or `torch.cuda.get_device_properties`;
- relevant package versions, especially `vllm`, `torch`, `transformers`, `unsloth`, `wandb`, `weave`, `pandas`, and `pyarrow`;
- the last successful progress/backup message;
- whether the local output Parquet exists and how many rows it contains.

Never diagnose from a paraphrased error if the traceback is available.

### 2. Trace definitions and consumers

Before changing the failing line, search the notebook for every definition and use of the involved names. Determine:

- which earlier cell creates the value;
- which later cells consume it;
- whether it survives the Phase 1 kernel restart;
- whether it belongs to the immutable W&B input artifact;
- whether changing it alters memory calculations, generation behavior, progress, Parquet columns, or W&B metadata.

Inspect the actual installed API or authoritative documentation when the error concerns a changing external library. Do not guess an argument name or silently delete a feature because a constructor rejects it.

### 3. Form one falsifiable root-cause hypothesis

State the root cause in one concrete sentence. Examples:

- “The installed vLLM build renamed or removed this `LLM` keyword.”
- “Phase 2 references a value that existed only before the kernel restart.”
- “The downloaded artifact filename differs from the filename expected by the notebook.”
- “The result dictionary and `OUTPUT_COLUMNS` no longer have identical ordered keys.”

Separate the root cause from secondary warnings. Fix only the proven cause first.

### 4. Patch the narrowest surface

Preferred repair order:

1. one literal or keyword;
2. one expression;
3. a few adjacent lines in the failing cell;
4. one small helper function;
5. one complete cell only when its internal contract is irreparably inconsistent;
6. multiple cells or a section only with the broad-rewrite approval described above.

Use `apply_patch`. Do not regenerate the notebook wholesale, reserialize every cell, clear outputs globally, reorder unrelated cells, or apply formatting churn.

Preserve unrelated user changes and existing notebook outputs. Do not reset the worktree.

### 5. Verify at the same scope as the failure

At minimum, every repair must pass:

1. JSON parsing and `nbformat.validate`;
2. Python AST parsing for every code cell after excluding notebook magics and shell lines;
3. a search proving removed or renamed symbols have no stale consumers;
4. a check that `OUTPUT_COLUMNS` exactly matches the ordered keys written for each result row;
5. the smallest deterministic test that reproduces the failed contract;
6. `uv run pytest` unless the failure environment makes it impossible;
7. rerunning the failing cell from the required kernel state when an A100 runtime is available.

Passing unrelated unit tests is not proof that a GPU/runtime failure is fixed. Conversely, inability to run an A100 locally is not permission to claim runtime verification.

## Notebook invariants that repairs must preserve

Unless the user explicitly changes the requirements, the following are contracts:

- Input data is the immutable W&B artifact `qwen3-5-4b-sliced-section-eval-inputs-no-thinking:v0`, derived from `notebooks/dataset/sft/test.parquet`.
- The full evaluation contains 9,176 section examples from 296 source rows.
- Median sliced-source length is 27 tokens; median rendered non-thinking prompt length is 493 tokens.
- Phase 2 does not tokenize or rebuild prompts on the A100.
- Qwen thinking is disabled for direct extraction.
- Every example retains its full precomputed generation allowance. Runtime estimation must not truncate outputs.
- The three-hour value is an optimization target and ETA, not an abort condition.
- If the run exceeds three hours, generation continues until the complete evaluation file is written.
- vLLM may schedule up to 128 live sequences with asynchronous scheduling and chunked prefill.
- Requests are grouped for prefix-cache reuse and restored to original `no` order in saved results.
- vLLM’s inner progress and the outer total-row progress/ETA remain visible.
- The Parquet backup is rewritten after every submission chunk and must remain readable.
- `OUTPUT_COLUMNS` and each saved result dictionary have exactly the same ordered fields.
- Per-row Weave tracing remains disabled by default because network calls can damage throughput; W&B run and artifact logging remain intact.
- No download URL, token, secret, copied browser profile, model output, or runtime artifact is committed accidentally.

If a proposed fix changes one of these invariants, stop and explain the tradeoff before editing.

## Failure-specific guidance

### Unsupported vLLM argument

Inspect `inspect.signature(LLM)` and the installed vLLM version. Patch only the incompatible argument or add a version-gated compatibility shim. Do not remove memory budgeting, prefix caching, chunked prefill, asynchronous scheduling, or progress as a group.

### CUDA out-of-memory

Capture allocated/reserved/free VRAM and determine whether the failure occurs during model load, CUDA graph capture, prefill, or decode. Change the binding memory/concurrency knob only. Do not shorten prompts, remove rows, reduce gold-derived output allowances, or rewrite the evaluation loop without evidence.

### W&B artifact failure

Distinguish authentication, entity/project, artifact version, filename, download, and network failures. Preserve the immutable version reference. Never silently fall back to another dataset or recompute prompts on the A100.

### Progress display issue

Repair the relevant `tqdm` construction or update call. Progress failure must not alter generation order, token budgets, checkpoint cadence, or output contents.

### Parquet/schema failure

Compare `OUTPUT_COLUMNS`, result keys, dtypes, and the existing partial file. Patch the mismatched field only. Preserve already generated rows whenever they are readable.

### Kernel restart/state failure

Identify which phase owns the missing value. Phase 2 must be self-contained after the Phase 1 restart. Add the missing import, constant, or artifact-derived initialization to Phase 2 rather than rerunning the merge phase or coupling both phases again.

## Required agent report

After a repair, report:

```text
Observed failure:
Root cause:
Minimal change:
Cells/lines changed:
Contracts explicitly preserved:
Verification performed:
What remains unverified and why:
```

Do not say “fixed” when only syntax or unrelated tests were checked. State exactly what evidence supports the claim.