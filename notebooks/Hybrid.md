# Hybrid — Merging the Plan C Extractor Back into a Chat-Capable Qwen LLM

**Question this document answers:** Plan C deliberately trains `Qwen/Qwen3.5-9B` as a
*reading encoder* — the LM head is frozen and never called, so the trained system cannot
speak. Can the trained weights be merged back into base Qwen so that the **final build is
an LLM** that (a) retains the extraction knowledge learned in Plan C, (b) still chats like
Qwen, and (c) can be post-trained with GRPO in `qwen_GRPO.ipynb`?

**Short answer:** Yes, with one precision. The QLoRA adapter — where all of Plan C's
backbone learning lives — merges directly into base Qwen by task-vector arithmetic, and
the result generates text, because Qwen3.5 is decoder-only and Plan C never removed the
LM head, only froze it. The structured heads (BiGRU, boundary filter, semi-CRF) are a
separate computation graph over hidden states and **mathematically cannot be folded into
transformer weights**; they survive as an optional sidecar. The path to a single LLM that
*speaks* the extraction is: merge the adapter, then SFT on the **existing** gold
line-anchored windowed data (Plan B format — the annotations were already produced by
GPT-class annotators; no new data generation is involved), then GRPO with the verifiable
rewards already implemented in `qwen_GRPO.ipynb`. Every step below is grounded in
published work (§7).

**Status:** design specification. No code yet.

---

## 1. What Plan C actually saved

From `Qwen3_5_(4B).ipynb` (persisted only as W&B Artifacts, project `sinergi-plan-c`,
pinned to a base `model_commit`):

| artifact | contents | mergeable into Qwen weights? |
|---|---|---|
| `adapter/adapter_model.safetensors` | PEFT LoRA `r=32, α=32`, `task_type="FEATURE_EXTRACTION"`, targets: full-attention `q/k/v/o_proj`, Gated DeltaNet `in_proj_qkv/z/b/a, out_proj`, MLP `gate/up/down_proj` | **yes** — standard LoRA delta `ΔW = BA·(α/r)` |
| `tokenizer/` | unchanged Qwen tokenizer | n/a |
| `training_state.pt` → `unit_pooler` | attention pool + `Linear(4096→512)` + LayerNorm | **no** — external module |
| `training_state.pt` → `heads` | 2-layer BiGRU (256/dir), boundary MLP, span scorer, semi-CRF transitions, fine/coarse/presence heads | **no** — external modules |

Two facts follow:

1. **Nothing was amputated from Qwen.** `lm_head`, embeddings, and every backbone weight
   are exactly the base checkpoint's; Plan C only *adds* low-rank deltas. "Giving the
   model back its decoder" requires no surgery — the decoder was never removed, merely
   unused. Merging the adapter yields a full `Qwen3_5ForConditionalGeneration` that
   `generate()`s.
2. **The extraction *decision procedure* does not live in the backbone.** Viterbi over
   the semi-CRF graph is what turns hidden states into spans. Merging the LoRA transfers
   the *representations* Plan C learned (what legal section boundaries look like), not
   the *ability to output* spans. That ability must be given to the LLM in generative
   form (§4), or kept as the sidecar heads (§3).

One caveat on merge mechanics: the notebook loads the multimodal wrapper
(`AutoModelForImageTextToText` fallback `AutoModelForCausalLM`), so adapter keys are
namespaced under the `language_model` submodule. The merge must be applied against that
submodule of the same `model_commit`, not a differently-flattened checkpoint.

---

## 2. Merge step — the Plan C LoRA as a task vector

Treat the adapter as a **task vector** τ = W_planC − W_base (Ilharco et al., ICLR 2023)
and merge with an explicit scaling coefficient:

```text
W_hybrid = W_base + λ · ΔW_lora ,   λ ∈ [0, 1] chosen on validation
```

λ = 1 is `merge_and_unload` / naive merging; the literature says not to default to it:

- **Task Arithmetic** (Ilharco et al., ICLR 2023): scaling a task vector trades task
  gain against interference with pretrained behavior; the coefficient is a tunable, not
  a constant.
- **TIES-Merging** (Yadav et al., NeurIPS 2023): trim small-magnitude deltas and resolve
  sign conflicts before merging to reduce interference.
- **DARE** (Yu et al., ICML 2024): 90–99% of SFT delta parameters can be dropped (with
  rescaling) without losing the fine-tuned ability — a strong tool here because the Plan
  C LoRA was trained on a *feature-extraction* objective, and sparsifying its delta
  minimizes disturbance to generation.
- **Model Soups** (Wortsman et al., ICML 2022): weight-space averaging of fine-tunes
  preserves and can improve robustness — the general evidence that interpolation in
  weight space is well-behaved for same-init models.

**Why chat degradation is the expected failure mode, and why it is repairable.** The
Plan C objective never touched next-token prediction, and fine-tuning on a narrow
objective distorts pretrained features in the direction of that objective (Kumar et al.,
ICLR 2022); even benign fine-tuning measurably erodes aligned chat behavior (Qi et al.,
ICLR 2024). Hence: (i) merge at swept λ (e.g. 0.3/0.5/0.7/1.0 with TIES/DARE
preprocessing), (ii) evaluate each candidate on a fixed chat regression suite
(Indonesian + English prompts, judged pass/fail) *and* an extraction probe, (iii) if no
λ preserves chat, apply the **Chat Vector** result (Huang et al., ACL 2024): chat
ability is itself a task vector `τ_chat = W_instruct − W_base` that can be *added back*
by pure arithmetic — published precedent that instruction-following survives and can be
restored through exactly this kind of weight arithmetic, with no retraining.

Output of this stage: `W_hybrid`, a standard full-precision Qwen3.5-9B checkpoint that
chats, carries Plan C's legal-boundary representations, and is loadable by Unsloth/TRL.

---

## 3. Dual-capability backbone — heads as a sidecar, one set of weights

The merged backbone supports two output paths simultaneously:

```text
                       ┌── lm_head ──────────────► chat / free-form generation
document → W_hybrid ───┤
                       └── UnitPooler → BiGRU → boundary filter → semi-CRF/Viterbi
                            → deterministic serializer ──► guaranteed-verbatim JSON
```

This is not an ad-hoc construction; it is the architecture class validated by:

- **GRIT / GritLM** (Muennighoff et al., ICLR 2025): a single LLM trained jointly on a
  representation objective and a generative objective loses neither — GritLM-7B is
  simultaneously state-of-the-art on MTEB and the best generative model at its size.
  Plan C's structured loss is a representation-side objective in exactly this sense.
- **LLM2Vec** (BehnamGhader et al., COLM 2024): decoder-only LLMs are strong text
  encoders with minimal adaptation — the premise Plan C already exploits.

Practical consequences:

- The Plan C heads (~a few M parameters, from `training_state.pt`) keep working on top
  of `W_hybrid` **if** λ = 1 and no further backbone training occurs; any λ < 1 or later
  SFT/GRPO shifts hidden states, after which the heads need a short **re-calibration
  pass** (Stage-2-style heads-only training on cached embeddings — cheap, no QLoRA).
- Optionally, a continued joint objective `L = L_causal + L_planC_structured`
  (GRIT-style multi-task) trains one set of weights to serve both paths explicitly.
  This is the maximal-fidelity variant; the λ-merge without joint training is the
  minimal-cost variant.
- The sidecar path retains Plan C's hard guarantee — output text is Python-sliced from
  the source, hallucination structurally impossible — and remains the right tool for
  trusted batch extraction even after the LLM path exists.

---

## 4. Teaching the LLM to *speak* the extraction (the GRPO-ready path)

GRPO requires a generative policy: TRL's `GRPOTrainer` samples completions and scores
them. The semi-CRF heads cannot be that policy. So the extraction ability must also be
expressed **generatively** — and the repo already owns the correct format and data for
this: **Plan B line-anchored windowed extraction** (`Haeryz/putusan-windowed-extraction`).

- **The data already exists and is already gold.** The annotations were produced by the
  GPT-class annotator pipeline (`LLM-aggregator`, 4,420 outputs) and reshaped into
  windowed line-range targets by `build_windowed_dataset.py`. This stage is ordinary SFT
  on existing supervision — *no* new teacher labeling, *no* re-distillation.
- **The output contract preserves Plan C's anti-hallucination property.** The model
  emits line indices it can literally see in the numbered window
  (`{"sections": {key: {"lines": [[s,e], …]}}}`), and a deterministic serializer copies
  the verbatim source text. Emitting references-into-the-input rather than content is
  the pointer-network principle (Vinyals et al., NeurIPS 2015); copy-not-generate is
  CopyNet (Gu et al., ACL 2016). The generated *legal text* therefore still cannot be
  hallucinated — only the *choice* of line ranges can be wrong, and that choice is
  exactly what the verifiable rewards score.
- **Malformed output is suppressible at decode time.** JSON/schema validity can be
  enforced by constrained decoding — incremental-parsing rejection (PICARD, Scholak et
  al., EMNLP 2021) or grammar-constrained decoding (Geng et al., EMNLP 2023) — so
  structural validity does not depend on the sampler behaving.

SFT specifics: fresh LoRA on `W_hybrid` (chat template intact, standard causal loss on
responses; windows ≤ ~8K tokens so the datalog.md truncation pathology cannot recur).
Optionally mix a small fraction of general chat data to hold instruction-following in
place (motivated by Qi et al., ICLR 2024). Warm-starting from `W_hybrid` rather than raw
base is the hypothesis that Plan C's merged representations transfer; the λ = 0 column
(raw base) is the control that tests it.

---

## 5. GRPO stage

`qwen_GRPO.ipynb` already implements everything needed and consumes the `grpo` config of
the same Hub repo, with **pure-Python verifiable rewards**: exact/approximate format
match, JSON structure validity, in-window range validity, and line-set overlap F1
against gold. GRPO (Shao et al., DeepSeekMath, arXiv:2402.03300) is the group-relative
PPO variant TRL implements; verifiable rewards make it reward-hacking-resistant here
because every reward is recomputable from the source window.

One change to the notebook is required: it currently loads
`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` as the policy. The hybrid pipeline swaps in the
Stage-2 SFT checkpoint (built on `W_hybrid`) instead. The reward functions, dataset, and
training loop are unchanged.

---

## 6. Pipeline summary and evaluation

```text
Stage 0  Plan C training (encoder + heads)                     — Qwen3_5_(4B).ipynb, W&B artifact
Stage 1  λ-scaled LoRA merge (TIES/DARE preprocessing)         — chat regression + extraction probe per λ
         └ fallback: add chat vector back (Huang et al. 2024)
Stage 2  Generative SFT on existing Plan B windowed gold data  — fresh LoRA on W_hybrid
         └ optional GRIT-style joint objective; heads re-calibration if kept
Stage 3  GRPO with verifiable rewards                          — qwen_GRPO.ipynb, policy = Stage-2 model
Stage 4  Export: merged 16-bit / GGUF chat model               — a standard HF LLM
         └ heads shipped as optional sidecar for guaranteed-verbatim batch extraction
```

**Evaluation gates** (Plan C.md style):

- chat: fixed Indonesian/English prompt suite, judged before/after each stage; a stage
  that destroys chat fails, regardless of extraction gains;
- extraction (LLM path): window line-overlap F1 and assembled-document character F1 on
  the held-out test split, reported against two bars — the λ = 0 control (base model,
  same SFT) and the Plan C semi-CRF sidecar;
- the hybrid claim — that Plan C's merged representations help the generative model —
  is **supported only if** the W_hybrid-initialized run beats the λ = 0 control with a
  paired-bootstrap 95% CI excluding zero; otherwise report that the merge contributed
  nothing and the value of Plan C remains the sidecar extractor;
- serializer invariant at every stage: emitted legal text equals its claimed source
  slice, 100% of rows.

**What the final build is:** a standard Hugging Face Qwen3.5 chat model (loadable with
`generate()`, exportable to GGUF) that carries Plan C's legal-structure knowledge in its
weights and expresses extraction as line-range JSON assembled deterministically — plus,
optionally, the Plan C heads as a zero-hallucination batch-extraction sidecar on the same
backbone.

---

## 7. References

**Model / task-vector merging**

1. Ilharco et al., *Editing Models with Task Arithmetic*. ICLR 2023.
   https://openreview.net/forum?id=6t0Kwf8-jrj
2. Yadav et al., *TIES-Merging: Resolving Interference When Merging Models*. NeurIPS 2023.
   https://openreview.net/forum?id=xtaX3WyCj1
3. Yu et al., *Language Models are Super Mario: Absorbing Abilities from Homologous
   Models as a Free Lunch* (DARE). ICML 2024.
   https://proceedings.mlr.press/v235/yu24p.html
4. Wortsman et al., *Model Soups: Averaging Weights of Multiple Fine-tuned Models
   Improves Accuracy without Increasing Inference Time*. ICML 2022.
5. Huang et al., *Chat Vector: A Simple Approach to Equip LLMs with Instruction
   Following and Model Alignment in New Languages*. ACL 2024.
   https://aclanthology.org/2024.acl-long.590/

**One backbone, generative + representation objectives**

6. Muennighoff et al., *Generative Representational Instruction Tuning* (GRIT). ICLR 2025.
   https://arxiv.org/abs/2402.09906
7. BehnamGhader et al., *LLM2Vec: Large Language Models Are Secretly Powerful Text
   Encoders*. COLM 2024. https://arxiv.org/abs/2404.05961

**Fine-tuning vs. pretrained behavior (risk analysis)**

8. Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform
   Out-of-Distribution*. ICLR 2022 (oral). https://arxiv.org/abs/2202.10054
9. Qi et al., *Fine-tuning Aligned Language Models Compromises Safety, Even When Users
   Do Not Intend To!* ICLR 2024.

**Index-emission / copy-not-generate output contract**

10. Vinyals et al., *Pointer Networks*. NeurIPS 2015.
    https://papers.neurips.cc/paper/5866-pointer-networks.pdf
11. Gu et al., *Incorporating Copying Mechanism in Sequence-to-Sequence Learning*
    (CopyNet). ACL 2016. https://aclanthology.org/P16-1154/
12. Scholak et al., *PICARD: Parsing Incrementally for Constrained Auto-Regressive
    Decoding from Language Models*. EMNLP 2021.
    https://aclanthology.org/2021.emnlp-main.779/
13. Geng et al., *Grammar-Constrained Decoding for Structured NLP Tasks without
    Finetuning*. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.674/

**GRPO**

14. Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
    Language Models* (introduces GRPO). arXiv:2402.03300.
    https://arxiv.org/abs/2402.03300

Legal-NLP and semi-CRF grounding for the Plan C recap is inherited from `Plan C.md` §2
(Sarawagi & Cohen, NIPS 2004; Filtered Semi-CRF, Findings of EMNLP 2023; ECIR 2023 joint
span segmentation; LegalSeg, Findings of NAACL 2025).
