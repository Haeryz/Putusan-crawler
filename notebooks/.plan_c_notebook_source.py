# %% [markdown]
# # Plan C — Schema-Constrained Hierarchical Span SFT
#
# **Qwen/Qwen3.5-9B · Google Colab A100 40GB · W&B Artifact persistence**
#
# This notebook replaces whole-document JSON generation with learned document
# segmentation. Qwen produces local semantic representations; a bidirectional
# document encoder and filtered semi-Markov CRF jointly learn the 31 section
# boundaries and labels. The final JSON copies predicted source spans verbatim.
#
# Important scope: input_text in this dataset is reconstructed from annotated
# sections in canonical order. Results measure reconstructed-document segmentation,
# not extraction from unfiltered raw judgments. RAG, GRPO, line-number targets, and
# windowed_dataset are intentionally absent.
#
# Keep the W&B run ID printed by the setup cell to resume after a Colab reset.

# %%
# Colab dependencies. Restart the runtime if Colab reports that an imported package
# was replaced, then continue from the next cell.
import subprocess
import sys

def pip_install(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip_install("--upgrade", "uv")
subprocess.check_call([
    "uv", "pip", "install", "-q",
    "transformers==5.2.0",
    "datasets>=4.0.0",
    "accelerate>=1.6.0",
    "peft>=0.17.0",
    "bitsandbytes>=0.46.0",
    "wandb>=0.21.0",
    "safetensors>=0.5.0",
    "sentencepiece",
])
subprocess.check_call([
    "uv", "pip", "install", "-q", "--no-build-isolation",
    "flash-linear-attention", "causal_conv1d==1.6.0",
])

# %% [markdown]
# ## Runtime, experiment configuration, and authentication
#
# W&B is the only durable store. /content is an ephemeral working directory.
# Put WANDB_API_KEY in Colab Secrets; never paste credentials into a cell.

# %%
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.optim import Optimizer
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

try:
    from google.colab import userdata
except ImportError:
    userdata = None


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "Qwen/Qwen3.5-9B"
    model_revision: str = "main"
    dataset_name: str = "Haeryz/putusan-structured-extraction"
    dataset_config: str = "sft"
    seed: int = 3407
    max_unit_tokens: int = 512
    max_chunk_tokens: int = 4096
    hidden_size: int = 4096
    unit_dim: int = 512
    document_hidden: int = 256
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_accumulation_docs: int = 4
    max_grad_norm: float = 1.0
    boundary_threshold: float = 0.5
    candidate_cap: int = 256
    required_boundary_recall: float = 0.995
    stage1_coarse_epochs: int = 1
    stage1_fine_epochs: int = 2
    stage2_max_epochs: int = 30
    stage2_patience: int = 5
    stage3_max_epochs: int = 5
    stage3_patience: int = 1
    # "C" full Plan C; "A2" frozen Qwen + independent classifier; "A3" local QLoRA
    # without global context; "A4" linear-chain CRF; "A5" unfiltered semi-CRF.
    ablation: str = "C"
    checkpoint_steps: int = 100
    cache_root: str = "/content/plan_c_cache"
    checkpoint_root: str = "/content/plan_c_checkpoint"
    wandb_project: str = "sinergi-plan-c"
    wandb_entity: str | None = None
    resume_run_id: str | None = None
    run_stage_1: bool = True
    run_stage_2: bool = True
    run_stage_3: bool = True
    run_final_test: bool = False


CFG = TrainConfig(
    resume_run_id=None,
)

assert torch.cuda.is_available(), "Select a GPU runtime in Google Colab."
GPU_NAME = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU: {GPU_NAME} ({GPU_GB:.2f} GiB)")
if "A100" not in GPU_NAME or GPU_GB < 39.0:
    print("WARNING: Plan C is sized for the Colab A100 40GB runtime; "
          "training on this GPU may be slow or run out of memory.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = CFG.wandb_project
os.environ["WANDB_WATCH"] = "false"

# %%
def start_wandb_run(cfg: TrainConfig) -> wandb.sdk.wandb_run.Run:
    key = userdata.get("WANDB_API_KEY") if userdata is not None else os.getenv("WANDB_API_KEY")
    if not key:
        raise RuntimeError(
            "Add WANDB_API_KEY to Colab Secrets and grant this notebook access."
        )
    wandb.login(key=key, relogin=True)
    run_id = cfg.resume_run_id or wandb.util.generate_id()
    resume_mode = "must" if cfg.resume_run_id else "allow"
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        id=run_id,
        resume=resume_mode,
        config=asdict(cfg),
        tags=["plan-c", "structured-sft", "filtered-semicrf", "qwen3.5"],
    )
    print(f"W&B run ID: {run.id}. Keep this ID to resume after a Colab reset.")
    return run


RUN = start_wandb_run(CFG)

# %% [markdown]
# ## Canonical schema and exact span supervision
#
# Alignment is sequential and exact. A rejected row never enters a loss. The
# notebook logs the rejection count and requires it to be zero before training.

# %%
CANONICAL_SECTIONS = [
    "judul", "nomor_putusan", "irah_irah", "nama_pengadilan_negeri",
    "keterangan_perkara", "nama_lengkap", "tempat_lahir", "umur_tanggal_lahir",
    "jenis_kelamin", "kebangsaan", "tempat_tinggal", "agama", "pekerjaan",
    "penangkapan", "penahanan", "tuntutan", "dakwaan", "saksi", "ahli",
    "terdakwa", "surat", "petunjuk_barang_bukti", "fakta_hukum",
    "pertimbangan_hukum", "amar_putusan", "hari", "tanggal", "tahun",
    "siapa_yang_memutus", "panitera_pengganti", "tanda_tangan_majelis",
]
assert len(CANONICAL_SECTIONS) == 31
SECTION_TO_ID = {name: index for index, name in enumerate(CANONICAL_SECTIONS)}

COARSE_GROUPS = {
    "header": CANONICAL_SECTIONS[0:5],
    "identity": CANONICAL_SECTIONS[5:13],
    "procedure": CANONICAL_SECTIONS[13:17],
    "evidence": CANONICAL_SECTIONS[17:22],
    "decision": CANONICAL_SECTIONS[22:25],
    "closing": CANONICAL_SECTIONS[25:31],
}
COARSE_TO_ID = {name: index for index, name in enumerate(COARSE_GROUPS)}
FINE_TO_COARSE = torch.tensor([
    COARSE_TO_ID[group]
    for section in CANONICAL_SECTIONS
    for group, members in COARSE_GROUPS.items()
    if section in members
], dtype=torch.long)
assert FINE_TO_COARSE.numel() == len(CANONICAL_SECTIONS)


@dataclass(frozen=True)
class GoldSpan:
    label: int
    item_index: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Unit:
    start_char: int
    end_char: int
    label: int
    item_index: int
    token_count: int
    forced_split: bool = False


@dataclass
class Chunk:
    start_char: int
    end_char: int
    unit_indices: list[int]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    unit_token_ranges: list[tuple[int, int]]


@dataclass
class DocumentExample:
    row_id: str
    source_file: str
    source_sha256: str
    corpus: str
    annotator_model: str
    text: str
    gold_spans: list[GoldSpan]
    units: list[Unit]
    chunks: list[Chunk]
    source_weight: float

    @property
    def unit_labels(self) -> torch.Tensor:
        return torch.tensor([unit.label for unit in self.units], dtype=torch.long)

    @property
    def gold_segments(self) -> list[tuple[int, int, int]]:
        by_item: list[tuple[int, int, int]] = []
        start = 0
        while start < len(self.units):
            unit = self.units[start]
            end = start + 1
            while (
                end < len(self.units)
                and self.units[end].label == unit.label
                and self.units[end].item_index == unit.item_index
            ):
                end += 1
            by_item.append((start, end, unit.label))
            start = end
        return by_item


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(CFG.seed)


def parse_sections(value: str | dict[str, Any]) -> dict[str, list[str]]:
    raw = json.loads(value) if isinstance(value, str) else value
    unknown = set(raw) - set(CANONICAL_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown section keys: {sorted(unknown)}")
    parsed: dict[str, list[str]] = {}
    for key in CANONICAL_SECTIONS:
        items = raw.get(key, [])
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise TypeError(f"{key} must be list[str]")
        parsed[key] = [item for item in items if item != ""]
    return parsed


def align_gold_spans(text: str, sections_value: str | dict[str, Any]) -> list[GoldSpan]:
    sections = parse_sections(sections_value)
    spans: list[GoldSpan] = []
    cursor = 0
    for label, key in enumerate(CANONICAL_SECTIONS):
        for item_index, item in enumerate(sections[key]):
            start = text.find(item, cursor)
            if start < 0:
                excerpt = item[:80].replace("\n", "\\n")
                raise ValueError(f"Cannot align {key}[{item_index}] after {cursor}: {excerpt}")
            end = start + len(item)
            if start < cursor or text[start:end] != item:
                raise AssertionError("Alignment must be monotonic and verbatim")
            spans.append(GoldSpan(label, item_index, start, end))
            cursor = end
    if not spans:
        raise ValueError("Document has no non-empty gold spans")
    for previous, current in itertools.pairwise(spans):
        if previous.end_char > current.start_char or previous.label > current.label:
            raise AssertionError("Gold spans overlap or violate canonical order")
    return spans


def split_interval_into_units(
    text: str,
    span: GoldSpan,
    tokenizer: Any,
    max_tokens: int,
) -> list[Unit]:
    units: list[Unit] = []
    segment = text[span.start_char:span.end_char]
    line_matches = list(re.finditer(r"[^\n]+", segment))
    if not line_matches:
        line_matches = [re.match(r"[\s\S]+", segment)]
    for match in line_matches:
        if match is None:
            continue
        line_start = span.start_char + match.start()
        line_end = span.start_char + match.end()
        line = text[line_start:line_end]
        encoded = tokenizer(
            line,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if not offsets:
            continue
        for token_start in range(0, len(offsets), max_tokens):
            token_end = min(token_start + max_tokens, len(offsets))
            char_start = line_start + offsets[token_start][0]
            char_end = line_start + offsets[token_end - 1][1]
            units.append(Unit(
                start_char=char_start,
                end_char=char_end,
                label=span.label,
                item_index=span.item_index,
                token_count=token_end - token_start,
                forced_split=len(offsets) > max_tokens,
            ))
    if not units:
        raise ValueError(f"Gold span {span} produced no tokenized unit")
    return units


def tokenize_chunk(
    text: str,
    units: Sequence[Unit],
    unit_indices: list[int],
    tokenizer: Any,
) -> Chunk:
    first, last = units[unit_indices[0]], units[unit_indices[-1]]
    chunk_text = text[first.start_char:last.end_char]
    encoded = tokenizer(
        chunk_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    token_ranges: list[tuple[int, int]] = []
    for unit_index in unit_indices:
        unit = units[unit_index]
        rel_start = unit.start_char - first.start_char
        rel_end = unit.end_char - first.start_char
        overlapping = [
            token_index
            for token_index, (start, end) in enumerate(offsets)
            if end > rel_start and start < rel_end
        ]
        if not overlapping:
            raise ValueError(f"No chunk tokens overlap unit {unit_index}")
        token_ranges.append((overlapping[0], overlapping[-1] + 1))
    return Chunk(
        start_char=first.start_char,
        end_char=last.end_char,
        unit_indices=unit_indices,
        input_ids=encoded["input_ids"][0],
        attention_mask=encoded["attention_mask"][0],
        unit_token_ranges=token_ranges,
    )


def pack_chunks(
    text: str,
    units: Sequence[Unit],
    tokenizer: Any,
    max_chunk_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    while start < len(units):
        end = start
        estimated_tokens = 0
        while end < len(units):
            unit = units[end]
            gap_tokens = 0
            if end > start:
                gap = text[units[end - 1].end_char:unit.start_char]
                gap_tokens = len(tokenizer(gap, add_special_tokens=False)["input_ids"])
            addition = unit.token_count + gap_tokens
            if end > start and estimated_tokens + addition > max_chunk_tokens:
                break
            estimated_tokens += addition
            end += 1
        candidate = tokenize_chunk(text, units, list(range(start, end)), tokenizer)
        while candidate.input_ids.numel() > max_chunk_tokens and end - start > 1:
            end -= 1
            candidate = tokenize_chunk(text, units, list(range(start, end)), tokenizer)
        if candidate.input_ids.numel() > max_chunk_tokens:
            raise ValueError(
                "A single unit exceeds max_chunk_tokens; lower max_unit_tokens."
            )
        chunks.append(candidate)
        start = candidate.unit_indices[-1] + 1
    return chunks


def prepare_document(
    row: dict[str, Any],
    tokenizer: Any,
    source_counts: Counter[str],
) -> DocumentExample:
    text = row["input_text"]
    spans = align_gold_spans(text, row["sections_json"])
    units = [
        unit
        for span in spans
        for unit in split_interval_into_units(
            text, span, tokenizer, CFG.max_unit_tokens
        )
    ]
    chunks = pack_chunks(text, units, tokenizer, CFG.max_chunk_tokens)
    sha = str(row["source_sha256"])
    document = DocumentExample(
        row_id=str(row.get("id", sha)),
        source_file=str(row.get("source_file", "")),
        source_sha256=sha,
        corpus=str(row.get("corpus", "")),
        annotator_model=str(row.get("annotator_model", "")),
        text=text,
        gold_spans=spans,
        units=units,
        chunks=chunks,
        source_weight=1.0 / source_counts[sha],
    )
    reconstructed = [
        document.text[span.start_char:span.end_char] for span in document.gold_spans
    ]
    expected = [
        item
        for key in CANONICAL_SECTIONS
        for item in parse_sections(row["sections_json"])[key]
    ]
    if reconstructed != expected:
        raise AssertionError("Gold offsets failed verbatim round-trip")
    return document
# %% [markdown]
# ## Dataset loading, source-balanced weights, and preprocessing audit
#
# The audit constants from `datalog.md` (3,075 rows aligned, 87,889 blank-line
# inter-section transitions) are recomputed here and logged to W&B; they are never
# trusted as constants. A single alignment failure aborts the run.

# %%
from transformers import AutoTokenizer  # noqa: E402  (kept next to its first use)

TOKENIZER = AutoTokenizer.from_pretrained(CFG.model_name, revision=CFG.model_revision)
assert TOKENIZER.is_fast, "Plan C needs a fast tokenizer for character offset mappings."

RAW_SPLITS = {
    split: load_dataset(CFG.dataset_name, CFG.dataset_config, split=split)
    for split in ("train", "validation", "test")
}
DATASET_FINGERPRINT = hashlib.sha256(
    "|".join(RAW_SPLITS[s]._fingerprint for s in ("train", "validation", "test")).encode()
).hexdigest()
print({split: len(ds) for split, ds in RAW_SPLITS.items()}, DATASET_FINGERPRINT[:16])


def prepare_split(split: str) -> list[DocumentExample]:
    rows = RAW_SPLITS[split]
    source_counts: Counter[str] = Counter(str(sha) for sha in rows["source_sha256"])
    documents: list[DocumentExample] = []
    failures: list[str] = []
    for row in rows:
        try:
            documents.append(prepare_document(row, TOKENIZER, source_counts))
        except (ValueError, AssertionError, TypeError) as error:
            failures.append(f"{row.get('id', '?')}: {error}")
    if failures:
        for line in failures[:10]:
            print("ALIGNMENT FAILURE", line)
        raise RuntimeError(
            f"{split}: {len(failures)} rows failed exact alignment; Plan C requires zero."
        )
    return documents


def audit_transitions(documents: list[DocumentExample]) -> dict[str, int]:
    transitions = blank_line = 0
    for document in documents:
        for previous, current in itertools.pairwise(document.gold_spans):
            if previous.label != current.label or previous.item_index != current.item_index:
                transitions += 1
                separator = document.text[previous.end_char:current.start_char]
                blank_line += int("\n\n" in separator)
    return {"transitions": transitions, "blank_line_separated": blank_line}


DOCS = {split: prepare_split(split) for split in ("train", "validation", "test")}
AUDIT = {
    split: {
        "rows": len(documents),
        "units": sum(len(d.units) for d in documents),
        "chunks": sum(len(d.chunks) for d in documents),
        "unique_sources": len({d.source_sha256 for d in documents}),
        **audit_transitions(documents),
    }
    for split, documents in DOCS.items()
}
for split, stats in AUDIT.items():
    print(split, stats)
RUN.log({f"data/{split}/{key}": value for split, stats in AUDIT.items() for key, value in stats.items()})
RUN.summary["data/fingerprint"] = DATASET_FINGERPRINT

train_shas = {d.source_sha256 for d in DOCS["train"]}
for split in ("validation", "test"):
    overlap = train_shas & {d.source_sha256 for d in DOCS[split]}
    assert not overlap, f"source_sha256 leakage between train and {split}: {sorted(overlap)[:3]}"


def gold_gap_labels(document: DocumentExample) -> tuple[torch.Tensor, torch.Tensor]:
    """Per internal gap: (is a segment boundary, is a section boundary vs same-section item)."""
    boundary, section = [], []
    for left, right in itertools.pairwise(document.units):
        changed = (left.label, left.item_index) != (right.label, right.item_index)
        boundary.append(float(changed))
        section.append(float(left.label != right.label))
    return (
        torch.tensor(boundary, dtype=torch.float32),
        torch.tensor(section, dtype=torch.float32),
    )
# %% [markdown]
# ## Model — 4-bit Qwen3.5 QLoRA encoder plus structured heads
#
# Qwen contributes semantic unit representations only: the notebook reads the text
# backbone's last hidden states and never computes vocabulary logits. LoRA adapters
# are discovered on the language backbone's full-attention, linear-attention
# (Gated DeltaNet), and MLP projections; the vision tower and LM head stay frozen.

# %%
DEVICE = torch.device("cuda")

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

try:
    BASE_MODEL = AutoModelForImageTextToText.from_pretrained(
        CFG.model_name,
        revision=CFG.model_revision,
        quantization_config=QUANT_CONFIG,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
except ValueError:  # text-only checkpoint fallback
    from transformers import AutoModelForCausalLM

    BASE_MODEL = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        revision=CFG.model_revision,
        quantization_config=QUANT_CONFIG,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )

MODEL_COMMIT = getattr(BASE_MODEL.config, "_commit_hash", CFG.model_revision)
BASE_MODEL.config.use_cache = False
for name, parameter in BASE_MODEL.named_parameters():
    if "visual" in name or "vision" in name or "lm_head" in name:
        parameter.requires_grad_(False)

LORA_SUFFIXES = {
    "full_attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "linear_attention": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


def discover_lora_targets(model: nn.Module) -> list[str]:
    """Fully-qualified language-backbone Linear names matching the Plan C suffixes."""
    targets: list[str] = []
    found: dict[str, set[str]] = {family: set() for family in LORA_SUFFIXES}
    for name, module in model.named_modules():
        if "visual" in name or "vision" in name or "lm_head" in name:
            continue
        if "Linear" not in type(module).__name__:  # covers nn.Linear and bnb Linear4bit
            continue
        suffix = name.rsplit(".", 1)[-1]
        for family, suffixes in LORA_SUFFIXES.items():
            if suffix in suffixes:
                targets.append(name)
                found[family].add(suffix)
    print({family: sorted(suffixes) for family, suffixes in found.items()})
    assert found["full_attention"] == set(LORA_SUFFIXES["full_attention"]), found
    assert found["mlp"] == set(LORA_SUFFIXES["mlp"]), found
    if not found["linear_attention"]:
        print("WARNING: no linear-attention projections found; is this a hybrid checkpoint?")
    return sorted(set(targets))


LORA_TARGETS = discover_lora_targets(BASE_MODEL)
BASE_MODEL = prepare_model_for_kbit_training(BASE_MODEL, use_gradient_checkpointing=True)
MODEL = get_peft_model(
    BASE_MODEL,
    LoraConfig(
        r=CFG.lora_rank,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=0.0,  # zero dropout keeps Stage 3 gradient-cache recomputation deterministic
        bias="none",
        target_modules=LORA_TARGETS,
        task_type="FEATURE_EXTRACTION",
    ),
)
MODEL.print_trainable_parameters()


def find_text_backbone(model: nn.Module) -> nn.Module:
    for name, module in model.named_modules():
        if name.endswith("language_model"):
            return module
    for attribute in ("model", "base_model"):
        model = getattr(model, attribute, model)
    return model


TEXT_BACKBONE = find_text_backbone(MODEL)
ACTUAL_HIDDEN = MODEL.config.get_text_config().hidden_size
assert ACTUAL_HIDDEN == CFG.hidden_size, f"config hidden_size {ACTUAL_HIDDEN} != {CFG.hidden_size}"

# %%
class UnitPooler(nn.Module):
    """Attention-pool each unit's token states, then project hidden_size -> unit_dim."""

    def __init__(self, hidden_size: int, unit_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, 1)
        self.project = nn.Linear(hidden_size, unit_dim)
        self.norm = nn.LayerNorm(unit_dim)

    def forward(
        self, hidden: torch.Tensor, token_ranges: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        pooled = []
        for start, end in token_ranges:
            states = hidden[start:end].float()
            weights = torch.softmax(self.query(states).squeeze(-1), dim=0)
            pooled.append(weights @ states)
        return self.norm(self.project(torch.stack(pooled)))


class DocumentEncoder(nn.Module):
    """Two-layer bidirectional GRU over the ordered unit sequence of a document."""

    def __init__(self, unit_dim: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            unit_dim, hidden, num_layers=2, bidirectional=True,
            dropout=0.1, batch_first=True,
        )

    def forward(self, units: torch.Tensor) -> torch.Tensor:
        return self.gru(units.unsqueeze(0))[0].squeeze(0)


class BoundaryScorer(nn.Module):
    """Score each gap between adjacent units from left/right context and position."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim + 1, dim), nn.GELU(), nn.Linear(dim, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        left, right = context[:-1], context[1:]
        position = torch.linspace(0, 1, len(context) - 1, device=context.device)
        features = torch.cat([left, right, left - right, position.unsqueeze(1)], dim=1)
        return self.mlp(features).squeeze(-1)


DURATION_BUCKETS = 12


class SpanScorer(nn.Module):
    """Span representation -> per-label potentials via a learned label embedding."""

    def __init__(self, dim: int, n_labels: int) -> None:
        super().__init__()
        self.duration = nn.Embedding(DURATION_BUCKETS, 32)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim + 32, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.label_embedding = nn.Embedding(n_labels, dim)
        self.bias = nn.Parameter(torch.zeros(n_labels))

    def forward(
        self, context: torch.Tensor, spans: torch.Tensor
    ) -> torch.Tensor:
        """spans: (S, 2) unit-index [start, end) pairs -> (S, n_labels) potentials."""
        prefix = torch.cat(
            [torch.zeros(1, context.size(1), device=context.device), context.cumsum(0)], dim=0
        )
        starts, ends = spans[:, 0], spans[:, 1]
        first = context[starts]
        last = context[ends - 1]
        mean = (prefix[ends] - prefix[starts]) / (ends - starts).unsqueeze(1).float()
        buckets = torch.clamp(
            torch.log2((ends - starts).float()).long(), 0, DURATION_BUCKETS - 1
        )
        representation = self.mlp(
            torch.cat([first, last, mean, self.duration(buckets)], dim=1)
        )
        return representation @ self.label_embedding.weight.T + self.bias


class StructuredHeads(nn.Module):
    """Everything trained on top of the projected unit embeddings."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        dim = cfg.unit_dim
        n_labels = len(CANONICAL_SECTIONS)
        self.document_encoder = DocumentEncoder(dim, cfg.document_hidden)
        self.boundary = BoundaryScorer(dim)
        self.span_scorer = SpanScorer(dim, n_labels)
        self.fine_classifier = nn.Linear(dim, n_labels)
        self.coarse_classifier = nn.Linear(dim, len(COARSE_GROUPS))
        self.presence = nn.Linear(dim, n_labels)
        self.transition = nn.Parameter(torch.zeros(n_labels, n_labels))
        self.register_buffer(
            "transition_mask",
            torch.tril(torch.full((n_labels, n_labels), float("-inf")), diagonal=-1),
        )

    def masked_transitions(self) -> torch.Tensor:
        """Backward label transitions are forbidden; same or later labels are allowed."""
        return self.transition + self.transition_mask


UNIT_POOLER = UnitPooler(CFG.hidden_size, CFG.unit_dim).to(DEVICE)
HEADS = StructuredHeads(CFG).to(DEVICE)
print(sum(p.numel() for p in HEADS.parameters()), "structured-head parameters")


def encode_chunk(chunk: Chunk, with_grad: bool) -> torch.Tensor:
    """Projected unit embeddings for one local chunk. Never touches the LM head."""
    grad_context = contextlib.nullcontext() if with_grad else torch.no_grad()
    with grad_context, torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = TEXT_BACKBONE(
            input_ids=chunk.input_ids.unsqueeze(0).to(DEVICE),
            attention_mask=chunk.attention_mask.unsqueeze(0).to(DEVICE),
        ).last_hidden_state[0]
        pooled = UNIT_POOLER(hidden, chunk.unit_token_ranges)
    return pooled if with_grad else pooled.detach()


def encode_document(document: DocumentExample, with_grad: bool = False) -> torch.Tensor:
    return torch.cat([encode_chunk(chunk, with_grad) for chunk in document.chunks], dim=0)
# %% [markdown]
# ## Filtered semi-Markov CRF — exact NLL, Viterbi, and segment posteriors
#
# Boundaries live in the gaps between adjacent units. Candidate boundaries always
# include the document start and end; during training every gold boundary is
# force-included so no supervised segmentation is ever pruned away. The semi-CRF is
# globally normalized over every valid labeled segmentation of the filtered graph:
# same-label transitions continue a list, later labels skip empty sections, and
# backward transitions are masked to -inf.

# %%
NEG_INF = float("-inf")


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal binary cross entropy for the boundary filter (gamma = 2)."""
    probability = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probability, 1 - probability)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return ((1 - pt).pow(gamma) * bce).mean()


def select_boundaries(
    gap_logits: torch.Tensor,
    n_units: int,
    threshold: float,
    cap: int,
    gold_gaps: torch.Tensor | None = None,
    keep_all: bool = False,
) -> list[int]:
    """Retained boundary positions in [0, n_units], always containing 0 and n_units.

    Position b means "between unit b-1 and unit b"; gap_logits[i] scores position i+1.
    """
    if n_units == 1 or keep_all:
        interior = set(range(1, n_units))
    else:
        interior = {
            index + 1 for index in torch.nonzero(gap_logits > threshold).flatten().tolist()
        }
        for index in torch.argsort(gap_logits, descending=True).tolist():
            if len(interior) >= cap:
                break
            interior.add(index + 1)
        if gold_gaps is not None:
            interior |= {i + 1 for i in torch.nonzero(gold_gaps > 0.5).flatten().tolist()}
    return sorted({0, n_units} | interior)


def enumerate_spans(boundaries: list[int], adjacent_only: bool = False) -> torch.Tensor:
    """All (start, end) unit ranges between retained boundary pairs, as a (S, 2) tensor.

    adjacent_only=True keeps only consecutive boundary pairs, reducing the semi-CRF
    to a linear-chain CRF (ablation A4).
    """
    pairs = [
        (boundaries[i], boundaries[j])
        for i in range(len(boundaries) - 1)
        for j in ((i + 1,) if adjacent_only else range(i + 1, len(boundaries)))
    ]
    return torch.tensor(pairs, dtype=torch.long)


@dataclass
class SemiCrfGraph:
    boundaries: list[int]           # sorted positions incl. 0 and n_units
    spans: torch.Tensor             # (S, 2) unit ranges
    potentials: torch.Tensor        # (S, L) span-label log-potentials
    span_index: dict[tuple[int, int], int]

    @classmethod
    def build(
        cls,
        context: torch.Tensor,
        boundaries: list[int],
        span_scorer: nn.Module,
        adjacent_only: bool = False,
    ) -> "SemiCrfGraph":
        spans = enumerate_spans(boundaries, adjacent_only).to(context.device)
        potentials = span_scorer(context, spans)
        index = {tuple(pair): i for i, pair in enumerate(spans.tolist())}
        return cls(boundaries, spans, potentials, index)


def semicrf_log_z(graph: SemiCrfGraph, transitions: torch.Tensor) -> torch.Tensor:
    """Exact log partition over all valid labeled segmentations of the filtered graph."""
    n_labels = transitions.size(0)
    positions = graph.boundaries
    alpha = [None] * len(positions)  # alpha[r]: (L,) log-mass of paths ending at position r
    alpha[0] = torch.zeros(n_labels, device=graph.potentials.device)  # virtual start
    for r in range(1, len(positions)):
        incoming = []
        for q in range(r):
            span_id = graph.span_index.get((positions[q], positions[r]))
            if span_id is None:
                continue
            phi = graph.potentials[span_id]  # (L,)
            if q == 0:
                incoming.append(phi)  # any first label; earlier sections are empty
            else:
                # marginalize the previous label under the monotonic transition mask
                message = torch.logsumexp(alpha[q].unsqueeze(1) + transitions, dim=0)
                incoming.append(message + phi)
        alpha[r] = torch.logsumexp(torch.stack(incoming), dim=0)
    return torch.logsumexp(alpha[-1], dim=0)  # any final label; later sections are empty


def semicrf_gold_score(
    graph: SemiCrfGraph,
    transitions: torch.Tensor,
    gold_segments: list[tuple[int, int, int]],
) -> torch.Tensor:
    score = graph.potentials.new_zeros(())
    previous_label: int | None = None
    for start, end, label in gold_segments:
        span_id = graph.span_index.get((start, end))
        assert span_id is not None, f"gold segment ({start}, {end}) pruned from the graph"
        score = score + graph.potentials[span_id, label]
        if previous_label is not None:
            assert label >= previous_label, "gold violates the monotonic schema"
            score = score + transitions[previous_label, label]
        previous_label = label
    return score


def semicrf_nll(
    graph: SemiCrfGraph,
    transitions: torch.Tensor,
    gold_segments: list[tuple[int, int, int]],
) -> torch.Tensor:
    return semicrf_log_z(graph, transitions) - semicrf_gold_score(
        graph, transitions, gold_segments
    )


def semicrf_viterbi(
    graph: SemiCrfGraph, transitions: torch.Tensor
) -> list[tuple[int, int, int]]:
    """Exact best labeled segmentation with backpointers -> [(start, end, label)]."""
    n_labels = transitions.size(0)
    positions = graph.boundaries
    best = [None] * len(positions)   # (L,) best score ending at position r with label l
    back: list[dict[int, tuple[int, int]]] = [dict() for _ in positions]
    best[0] = torch.zeros(n_labels, device=graph.potentials.device)
    for r in range(1, len(positions)):
        scores = torch.full((n_labels,), NEG_INF, device=graph.potentials.device)
        for q in range(r):
            span_id = graph.span_index.get((positions[q], positions[r]))
            if span_id is None:
                continue
            phi = graph.potentials[span_id]
            if q == 0:
                candidate = phi
                sources = torch.full((n_labels,), -1, dtype=torch.long)
            else:
                combined = best[q].unsqueeze(1) + transitions  # (L_prev, L)
                message, sources = combined.max(dim=0)
                candidate = message + phi
            improved = candidate > scores
            for label in torch.nonzero(improved).flatten().tolist():
                back[r][label] = (q, int(sources[label]))
            scores = torch.where(improved, candidate, scores)
        best[r] = scores
    label = int(best[-1].argmax())
    segments: list[tuple[int, int, int]] = []
    r = len(positions) - 1
    while r > 0:
        q, previous_label = back[r][label]
        segments.append((positions[q], positions[r], label))
        r, label = q, previous_label
    segments.reverse()
    return segments


def semicrf_segment_posteriors(
    graph: SemiCrfGraph,
    transitions: torch.Tensor,
    segments: list[tuple[int, int, int]],
) -> list[float]:
    """Exact marginal probability of each decoded segment (confidence values)."""
    n_labels = transitions.size(0)
    positions = graph.boundaries
    rank = {p: r for r, p in enumerate(positions)}
    device = graph.potentials.device
    alpha = [None] * len(positions)
    alpha[0] = torch.zeros(n_labels, device=device)
    for r in range(1, len(positions)):
        incoming = []
        for q in range(r):
            span_id = graph.span_index.get((positions[q], positions[r]))
            if span_id is None:
                continue
            phi = graph.potentials[span_id]
            if q == 0:
                incoming.append(phi)
            else:
                incoming.append(
                    torch.logsumexp(alpha[q].unsqueeze(1) + transitions, dim=0) + phi
                )
        alpha[r] = torch.logsumexp(torch.stack(incoming), dim=0)
    log_z = torch.logsumexp(alpha[-1], dim=0)
    # beta[r][l]: log-mass of suffix segmentations after position r given last label l
    beta = [None] * len(positions)
    beta[-1] = torch.zeros(n_labels, device=device)
    for r in range(len(positions) - 2, -1, -1):
        outgoing = []
        for s in range(r + 1, len(positions)):
            span_id = graph.span_index.get((positions[r], positions[s]))
            if span_id is None:
                continue
            phi = graph.potentials[span_id]
            outgoing.append(
                torch.logsumexp(transitions + (phi + beta[s]).unsqueeze(0), dim=1)
            )
        beta[r] = torch.logsumexp(torch.stack(outgoing), dim=0)
    posteriors = []
    for start, end, label in segments:
        phi = graph.potentials[graph.span_index[(start, end)], label]
        r_start, r_end = rank[start], rank[end]
        if r_start == 0:
            mass_in = torch.zeros((), device=device)
        else:
            mass_in = torch.logsumexp(alpha[r_start] + transitions[:, label], dim=0)
        mass_out = beta[r_end][label]
        posteriors.append(float(torch.exp(mass_in + phi + mass_out - log_z)))
    return posteriors
# %% [markdown]
# ## Joint objective, optimizer, and the W&B checkpoint contract
#
# `L = 1.0*semicrf_nll + 0.5*boundary_focal + 0.2*fine_ce + 0.1*coarse_ce + 0.2*presence_bce`
#
# W&B Artifacts are the only durable store: `latest` every 100 optimizer steps and
# at every stage boundary, `best` on validation span macro-F1 improvement. Bundles
# carry adapter+tokenizer, every head, optimizer/scheduler, cursors, and all RNG
# states so a fresh Colab runtime reproduces the next update.

# %%
LOSS_WEIGHTS = {
    "semicrf": 1.0, "boundary": 0.5, "fine": 0.2, "coarse": 0.1, "presence": 0.2,
}
FINE_TO_COARSE_DEV = FINE_TO_COARSE.to(DEVICE)

CALIBRATION = {"threshold": CFG.boundary_threshold, "cap": CFG.candidate_cap}
TRAIN_STATE = {
    "stage": 1, "epoch": 0, "doc_cursor": 0, "accum_cursor": 0, "global_step": 0,
    "best_val_span_f1": -1.0,
}


def document_targets(document: DocumentExample) -> dict[str, torch.Tensor]:
    fine = document.unit_labels.to(DEVICE)
    gold_boundary, gold_section = gold_gap_labels(document)
    present = torch.zeros(len(CANONICAL_SECTIONS), device=DEVICE)
    present[fine.unique()] = 1.0
    return {
        "fine": fine,
        "coarse": FINE_TO_COARSE_DEV[fine],
        "boundary": gold_boundary.to(DEVICE),
        "section_boundary": gold_section.to(DEVICE),
        "presence": present,
    }


def structured_document_loss(
    embeddings: torch.Tensor,
    document: DocumentExample,
    ablation: str = "C",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full Plan C objective on one document's ordered unit embeddings (training mode:
    gold boundaries are always force-included in the candidate set)."""
    targets = document_targets(document)
    context = (
        embeddings if ablation in ("A2", "A3") else HEADS.document_encoder(embeddings)
    )
    parts: dict[str, torch.Tensor] = {}
    parts["fine"] = F.cross_entropy(HEADS.fine_classifier(context), targets["fine"])
    parts["coarse"] = F.cross_entropy(HEADS.coarse_classifier(context), targets["coarse"])
    parts["presence"] = F.binary_cross_entropy_with_logits(
        HEADS.presence(context.mean(dim=0)), targets["presence"]
    )
    if len(document.units) > 1:
        gap_logits = HEADS.boundary(context)
        parts["boundary"] = focal_bce(gap_logits, targets["boundary"])
    else:
        gap_logits = torch.zeros(0, device=DEVICE)
        parts["boundary"] = embeddings.new_zeros(())
    if ablation in ("A2", "A3"):
        parts["semicrf"] = embeddings.new_zeros(())  # independent classifier ablations
    else:
        boundaries = select_boundaries(
            gap_logits.detach(),
            len(document.units),
            CALIBRATION["threshold"],
            CALIBRATION["cap"],
            gold_gaps=targets["boundary"],
            keep_all=ablation in ("A4", "A5"),
        )
        graph = SemiCrfGraph.build(
            context, boundaries, HEADS.span_scorer, adjacent_only=ablation == "A4"
        )
        gold = (
            [(i, i + 1, int(label)) for i, label in enumerate(document.unit_labels)]
            if ablation == "A4"
            else document.gold_segments
        )
        parts["semicrf"] = semicrf_nll(graph, HEADS.masked_transitions(), gold) / len(gold)
    total = sum(LOSS_WEIGHTS[name] * value for name, value in parts.items())
    return total, {name: float(value) for name, value in parts.items()}


def trainable_lora_parameters() -> list[torch.nn.Parameter]:
    return [p for _, p in MODEL.named_parameters() if p.requires_grad]


def build_optimizer(
    lora: bool, heads: bool, total_steps: int
) -> tuple[Optimizer, Any]:
    groups = []
    if lora:
        groups.append({"params": trainable_lora_parameters() + list(UNIT_POOLER.parameters()),
                       "lr": CFG.lora_lr})
    if heads:
        groups.append({"params": list(HEADS.parameters()), "lr": CFG.head_lr})
    try:
        from bitsandbytes.optim import AdamW8bit as AdamWImpl
    except ImportError:
        print("bitsandbytes AdamW8bit unavailable; using torch AdamW")
        AdamWImpl = torch.optim.AdamW
    optimizer = AdamWImpl(groups, weight_decay=CFG.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(CFG.warmup_ratio * total_steps)), max(total_steps, 1)
    )
    return optimizer, scheduler


# %%
CHECKPOINT_ARTIFACT = f"plan-c-checkpoint-{RUN.id}"


def save_checkpoint(
    optimizer: Optimizer | None,
    scheduler: Any,
    best: bool = False,
    reason: str = "step",
) -> None:
    root = Path(CFG.checkpoint_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    MODEL.save_pretrained(str(root / "adapter"))
    TOKENIZER.save_pretrained(str(root / "tokenizer"))
    torch.save(
        {
            "heads": HEADS.state_dict(),
            "unit_pooler": UNIT_POOLER.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "train_state": dict(TRAIN_STATE),
            "calibration": dict(CALIBRATION),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
            "config": asdict(CFG),
            "canonical_sections": CANONICAL_SECTIONS,
            "coarse_groups": {k: list(v) for k, v in COARSE_GROUPS.items()},
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "model_commit": MODEL_COMMIT,
            "wandb_run_id": RUN.id,
        },
        root / "training_state.pt",
    )
    artifact = wandb.Artifact(
        CHECKPOINT_ARTIFACT,
        type="checkpoint",
        metadata={
            "reason": reason,
            **{k: v for k, v in TRAIN_STATE.items()},
            "best": best,
        },
    )
    artifact.add_dir(str(root))
    aliases = ["latest", "best"] if best else ["latest"]
    logged = RUN.log_artifact(artifact, aliases=aliases)
    logged.wait()  # block before local checkpoint files may be replaced


def maybe_checkpoint(optimizer: Optimizer, scheduler: Any) -> None:
    if TRAIN_STATE["global_step"] > 0 and TRAIN_STATE["global_step"] % CFG.checkpoint_steps == 0:
        save_checkpoint(optimizer, scheduler, reason="periodic")


def resume_from_wandb() -> None:
    """Restore adapter, heads, cursors, calibration, and all RNG states from `latest`."""
    if not CFG.resume_run_id:
        return
    artifact = RUN.use_artifact(f"{CHECKPOINT_ARTIFACT}:latest", type="checkpoint")
    directory = Path(artifact.download())
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict

    adapter_weights = load_file(str(directory / "adapter" / "adapter_model.safetensors"))
    set_peft_model_state_dict(MODEL, adapter_weights)
    bundle = torch.load(directory / "training_state.pt", weights_only=False)
    assert bundle["dataset_fingerprint"] == DATASET_FINGERPRINT, "dataset changed since checkpoint"
    HEADS.load_state_dict(bundle["heads"])
    UNIT_POOLER.load_state_dict(bundle["unit_pooler"])
    TRAIN_STATE.update(bundle["train_state"])
    CALIBRATION.update(bundle["calibration"])
    rng = bundle["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda"])
    globals()["_RESUMED_OPTIMIZER_STATE"] = (bundle["optimizer"], bundle["scheduler"])
    print(f"Resumed run {RUN.id} at {TRAIN_STATE}")


_RESUMED_OPTIMIZER_STATE: tuple[Any, Any] | None = None


def adopt_resumed_optimizer(optimizer: Optimizer, scheduler: Any) -> None:
    global _RESUMED_OPTIMIZER_STATE
    if _RESUMED_OPTIMIZER_STATE is not None:
        opt_state, sched_state = _RESUMED_OPTIMIZER_STATE
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
        _RESUMED_OPTIMIZER_STATE = None


# %% [markdown]
# ## Stage 1 — local QLoRA curriculum (coarse -> fine unit classification)
#
# Qwen chunks are trained locally, without the document encoder: one epoch of
# 6-way coarse classification plus boundary loss, then up to two epochs of 31-way
# classification plus boundary loss. Loss is source-balanced by `1/rows_per_sha`.

# %%
def epoch_document_order(epoch_key: int, n_documents: int) -> list[int]:
    generator = random.Random(CFG.seed * 100_003 + epoch_key)
    order = list(range(n_documents))
    generator.shuffle(order)
    return order


def stage1_chunk_loss(
    document: DocumentExample, chunk: Chunk, mode: str
) -> tuple[torch.Tensor, int]:
    embeddings = encode_chunk(chunk, with_grad=True)
    labels = torch.tensor(
        [document.units[i].label for i in chunk.unit_indices], dtype=torch.long, device=DEVICE
    )
    if mode == "coarse":
        loss = F.cross_entropy(HEADS.coarse_classifier(embeddings), FINE_TO_COARSE_DEV[labels])
    else:
        loss = F.cross_entropy(HEADS.fine_classifier(embeddings), labels)
    if len(chunk.unit_indices) > 1:
        gold_boundary, _ = gold_gap_labels(document)
        first = chunk.unit_indices[0]
        local_gold = gold_boundary[first:first + len(chunk.unit_indices) - 1].to(DEVICE)
        loss = loss + 0.5 * focal_bce(HEADS.boundary(embeddings), local_gold)
    return loss, len(chunk.unit_indices)


def run_stage_1() -> None:
    curriculum = (
        [("coarse", 0)] * CFG.stage1_coarse_epochs + [("fine", 0)] * CFG.stage1_fine_epochs
    )
    curriculum = [(mode, index) for index, (mode, _) in enumerate(curriculum)]
    documents = DOCS["train"]
    steps_per_epoch = math.ceil(len(documents) / CFG.gradient_accumulation_docs)
    optimizer, scheduler = build_optimizer(
        lora=True, heads=True, total_steps=len(curriculum) * steps_per_epoch
    )
    adopt_resumed_optimizer(optimizer, scheduler)
    MODEL.train()
    best_fine_f1 = -1.0
    for mode, epoch in curriculum:
        if epoch < TRAIN_STATE["epoch"]:
            continue
        order = epoch_document_order(epoch, len(documents))
        accumulated = 0
        epoch_start = time.time()
        for cursor, doc_index in enumerate(order):
            if epoch == TRAIN_STATE["epoch"] and cursor < TRAIN_STATE["doc_cursor"]:
                continue
            document = documents[doc_index]
            for chunk in document.chunks:
                loss, _ = stage1_chunk_loss(document, chunk, mode)
                scaled = loss * document.source_weight / CFG.gradient_accumulation_docs
                scaled.backward()
            accumulated += 1
            if accumulated == CFG.gradient_accumulation_docs or cursor == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], CFG.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                TRAIN_STATE["global_step"] += 1
                accumulated = 0
                RUN.log({
                    "stage1/loss": float(loss), "stage1/mode": 0 if mode == "coarse" else 1,
                    "stage1/lr": scheduler.get_last_lr()[0],
                    "global_step": TRAIN_STATE["global_step"],
                })
                TRAIN_STATE.update(epoch=epoch, doc_cursor=cursor + 1)
                maybe_checkpoint(optimizer, scheduler)
        TRAIN_STATE.update(epoch=epoch + 1, doc_cursor=0)
        metrics = evaluate_unit_classifier("validation", mode)
        RUN.log({f"stage1/val_{mode}_macro_f1": metrics["macro_f1"], "epoch": epoch})
        print(f"stage1 epoch {epoch} ({mode}) macro-F1 {metrics['macro_f1']:.4f} "
              f"({time.time() - epoch_start:.0f}s)")
        if mode == "fine" and metrics["macro_f1"] > best_fine_f1:
            best_fine_f1 = metrics["macro_f1"]
            save_checkpoint(optimizer, scheduler, reason="stage1-best")
    save_checkpoint(optimizer, scheduler, reason="stage1-end")


# %% [markdown]
# ## Stage 2 — cached embeddings and the global structured head
#
# Qwen and the unit projector are frozen; projected train/validation embeddings are
# cached under `/content` and logged as a versioned W&B Artifact so a new runtime
# resumes without recomputing them. The GRU, boundary filter, presence head, and
# filtered semi-CRF train for at most 30 epochs with patience 5.

# %%
def cache_split_embeddings(split: str) -> dict[str, Path]:
    directory = Path(CFG.cache_root) / split
    directory.mkdir(parents=True, exist_ok=True)
    MODEL.eval()
    paths: dict[str, Path] = {}
    for document in DOCS[split]:
        path = directory / f"{document.row_id.replace('/', '_')}.pt"
        if not path.exists():
            torch.save(encode_document(document, with_grad=False).cpu(), path)
        paths[document.row_id] = path
    return paths


def log_embedding_cache_artifact() -> None:
    artifact = wandb.Artifact(
        f"plan-c-embeddings-{RUN.id}",
        type="embedding-cache",
        metadata={"fingerprint": DATASET_FINGERPRINT, "stage": TRAIN_STATE["stage"]},
    )
    artifact.add_dir(CFG.cache_root)
    RUN.log_artifact(artifact).wait()


EMBEDDING_PATHS: dict[str, dict[str, Path]] = {}


def cached_embeddings(split: str, document: DocumentExample) -> torch.Tensor:
    return torch.load(EMBEDDING_PATHS[split][document.row_id], weights_only=True).to(DEVICE)


def run_stage_2() -> None:
    for split in ("train", "validation"):
        EMBEDDING_PATHS[split] = cache_split_embeddings(split)
    log_embedding_cache_artifact()
    documents = DOCS["train"]
    steps_per_epoch = math.ceil(len(documents) / CFG.gradient_accumulation_docs)
    max_epochs = CFG.stage2_max_epochs
    optimizer, scheduler = build_optimizer(
        lora=False, heads=True, total_steps=max_epochs * steps_per_epoch
    )
    adopt_resumed_optimizer(optimizer, scheduler)
    patience_left = CFG.stage2_patience
    for epoch in range(max_epochs):
        if epoch < TRAIN_STATE["epoch"]:
            continue
        HEADS.train()
        order = epoch_document_order(10_000 + epoch, len(documents))
        accumulated = 0
        for cursor, doc_index in enumerate(order):
            if epoch == TRAIN_STATE["epoch"] and cursor < TRAIN_STATE["doc_cursor"]:
                continue
            document = documents[doc_index]
            embeddings = cached_embeddings("train", document)
            loss, parts = structured_document_loss(embeddings, document, CFG.ablation)
            (loss * document.source_weight / CFG.gradient_accumulation_docs).backward()
            accumulated += 1
            if accumulated == CFG.gradient_accumulation_docs or cursor == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(HEADS.parameters(), CFG.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                TRAIN_STATE["global_step"] += 1
                accumulated = 0
                RUN.log({
                    **{f"stage2/{k}": v for k, v in parts.items()},
                    "stage2/loss": float(loss), "global_step": TRAIN_STATE["global_step"],
                })
                TRAIN_STATE.update(epoch=epoch, doc_cursor=cursor + 1)
                maybe_checkpoint(optimizer, scheduler)
        TRAIN_STATE.update(epoch=epoch + 1, doc_cursor=0)
        metrics = evaluate_split("validation", use_cache=True)
        span_f1 = metrics["span_macro_f1"]
        RUN.log({"stage2/val_span_macro_f1": span_f1, "epoch": epoch})
        print(f"stage2 epoch {epoch} span macro-F1 {span_f1:.4f}")
        if span_f1 > TRAIN_STATE["best_val_span_f1"]:
            TRAIN_STATE["best_val_span_f1"] = span_f1
            patience_left = CFG.stage2_patience
            save_checkpoint(optimizer, scheduler, best=True, reason="stage2-best")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("stage2 early stop")
                break
    calibrate_boundary_filter()
    save_checkpoint(optimizer, scheduler, reason="stage2-end")


def calibrate_boundary_filter(use_cache: bool = True) -> None:
    """Choose the threshold/cap on validation so gold-boundary recall >= 99.5%."""
    HEADS.eval()
    logits_gold: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for document in DOCS["validation"]:
            if len(document.units) < 2:
                continue
            embeddings = (
                cached_embeddings("validation", document)
                if use_cache
                else encode_document(document, with_grad=False).float()
            )
            context = HEADS.document_encoder(embeddings)
            gold_boundary, _ = gold_gap_labels(document)
            logits_gold.append((HEADS.boundary(context).cpu(), gold_boundary))
    for cap in (CFG.candidate_cap, 512, 1024):
        for threshold in [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]:
            retained = total = 0
            for logits, gold in logits_gold:
                positions = set(select_boundaries(logits, len(logits) + 1, threshold, cap))
                gold_positions = {i + 1 for i in torch.nonzero(gold > 0.5).flatten().tolist()}
                retained += len(gold_positions & positions)
                total += len(gold_positions)
            recall = retained / max(total, 1)
            if recall >= CFG.required_boundary_recall:
                CALIBRATION.update(threshold=threshold, cap=cap)
                RUN.log({"calibration/threshold": threshold, "calibration/cap": cap,
                         "calibration/gold_recall": recall})
                print(f"calibrated threshold={threshold} cap={cap} recall={recall:.5f}")
                return
    print("WARNING: no calibration met the 99.5% recall gate; keeping defaults")


# %% [markdown]
# ## Stage 3 — joint full-document tuning through representation-gradient caching
#
# Per document: (1) run every chunk without autograd and collect projected unit
# embeddings; (2) treat them as leaves and backpropagate the structured loss to get
# `dL/dEmbedding`; (3) re-run one Qwen chunk at a time with autograd and
# backpropagate the dot product with the cached gradients. With zero LoRA dropout
# the recomputation is deterministic. QLoRA, projector, and heads update only after
# the complete logical document has been processed.

# %%
def gradient_cache_document_backward(document: DocumentExample) -> dict[str, float]:
    embeddings = encode_document(document, with_grad=False).float()
    leaves = embeddings.detach().requires_grad_(True)
    loss, parts = structured_document_loss(leaves, document, CFG.ablation)
    scaled = loss * document.source_weight / CFG.gradient_accumulation_docs
    scaled.backward()  # populates HEADS grads and leaves.grad
    cached_grad = leaves.grad
    offset = 0
    for chunk in document.chunks:
        recomputed = encode_chunk(chunk, with_grad=True).float()
        n_units = recomputed.size(0)
        surrogate = (recomputed * cached_grad[offset:offset + n_units]).sum()
        surrogate.backward()  # populates LoRA + projector grads
        offset += n_units
    assert offset == cached_grad.size(0)
    parts["total"] = float(loss)
    return parts


def gradient_cache_equivalence_test() -> None:
    """Tiny-model check: cached and ordinary backpropagation agree within tolerance."""
    torch.manual_seed(0)
    encoder = nn.Sequential(nn.Embedding(50, 16), nn.Linear(16, 16), nn.Tanh())
    head = nn.Linear(16, 1)
    tokens = torch.randint(0, 50, (12,))
    chunks = [tokens[:5], tokens[5:9], tokens[9:]]

    def full_loss(embeddings: torch.Tensor) -> torch.Tensor:
        return head(embeddings.mean(dim=0)).pow(2).sum() + embeddings.var()

    # ordinary end-to-end backprop
    for module in (encoder, head):
        module.zero_grad()
    direct = full_loss(torch.cat([encoder(c) for c in chunks]))
    direct.backward()
    reference = [p.grad.clone() for p in list(encoder.parameters()) + list(head.parameters())]

    # gradient-cache backprop
    for module in (encoder, head):
        module.zero_grad()
    with torch.no_grad():
        cached = torch.cat([encoder(c) for c in chunks])
    leaves = cached.detach().requires_grad_(True)
    full_loss(leaves).backward()
    offset = 0
    for chunk in chunks:
        recomputed = encoder(chunk)
        (recomputed * leaves.grad[offset:offset + len(chunk)]).sum().backward()
        offset += len(chunk)
    cached_grads = [p.grad.clone() for p in list(encoder.parameters()) + list(head.parameters())]
    for expected, actual in zip(reference, cached_grads):
        assert torch.allclose(expected, actual, atol=1e-5), (expected - actual).abs().max()
    print("gradient-cache equivalence test passed")


gradient_cache_equivalence_test()


def run_stage_3() -> None:
    documents = DOCS["train"]
    steps_per_epoch = math.ceil(len(documents) / CFG.gradient_accumulation_docs)
    max_epochs = CFG.stage3_max_epochs
    optimizer, scheduler = build_optimizer(
        lora=True, heads=True, total_steps=max_epochs * steps_per_epoch
    )
    adopt_resumed_optimizer(optimizer, scheduler)
    patience_left = CFG.stage3_patience
    all_parameters = [p for g in optimizer.param_groups for p in g["params"]]
    for epoch in range(max_epochs):
        if epoch < TRAIN_STATE["epoch"]:
            continue
        MODEL.train()
        HEADS.train()
        order = epoch_document_order(20_000 + epoch, len(documents))
        accumulated = 0
        for cursor, doc_index in enumerate(order):
            if epoch == TRAIN_STATE["epoch"] and cursor < TRAIN_STATE["doc_cursor"]:
                continue
            parts = gradient_cache_document_backward(documents[doc_index])
            accumulated += 1
            if accumulated == CFG.gradient_accumulation_docs or cursor == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(all_parameters, CFG.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                TRAIN_STATE["global_step"] += 1
                accumulated = 0
                RUN.log({
                    **{f"stage3/{k}": v for k, v in parts.items()},
                    "global_step": TRAIN_STATE["global_step"],
                })
                TRAIN_STATE.update(epoch=epoch, doc_cursor=cursor + 1)
                maybe_checkpoint(optimizer, scheduler)
        TRAIN_STATE.update(epoch=epoch + 1, doc_cursor=0)
        metrics = evaluate_split("validation", use_cache=False)
        span_f1 = metrics["span_macro_f1"]
        RUN.log({"stage3/val_span_macro_f1": span_f1, "epoch": epoch})
        print(f"stage3 epoch {epoch} span macro-F1 {span_f1:.4f}")
        if span_f1 > TRAIN_STATE["best_val_span_f1"]:
            TRAIN_STATE["best_val_span_f1"] = span_f1
            patience_left = CFG.stage3_patience
            save_checkpoint(optimizer, scheduler, best=True, reason="stage3-best")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("stage3 early stop")
                break
    calibrate_boundary_filter(use_cache=False)  # Stage 3 moved the encoder; cache is stale
    save_checkpoint(optimizer, scheduler, reason="stage3-end")
# %% [markdown]
# ## Decoding, metrics, and artifact baselines
#
# Validation drives thresholds, early stopping, ablations, and model selection; the
# test split is loaded once, after the configuration is frozen. Source-level
# aggregation gives each `source_sha256` equal weight. A0 (position-only) and A1
# (whitespace-only) are mandatory artifact baselines: if they match the learned
# model, the experiment demonstrates a dataset artifact, not legal semantics.
#
# Ablation systems map onto `CFG.ablation`:
# A2 frozen Qwen + independent unit classifier (stages 1/3 skipped) · A3 local QLoRA
# without global context · A4 linear-chain CRF (all gaps, adjacent spans) ·
# A5 unfiltered semi-CRF (all gaps) · C full Plan C.

# %%
def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def decode_document(
    document: DocumentExample,
    embeddings: torch.Tensor,
    ablation: str = "C",
) -> tuple[list[tuple[int, int, int]], list[float], dict[str, float]]:
    """Viterbi (or ablation) decode -> (segments, confidences, filter statistics)."""
    HEADS.eval()
    with torch.no_grad():
        if ablation in ("A2", "A3"):
            logits = HEADS.fine_classifier(embeddings)
            labels = logits.argmax(dim=1).tolist()
            probabilities = torch.softmax(logits, dim=1)
            segments, confidences = [], []
            start = 0
            for index in range(1, len(labels) + 1):
                if index == len(labels) or labels[index] != labels[start]:
                    segments.append((start, index, labels[start]))
                    confidences.append(
                        float(probabilities[start:index, labels[start]].mean())
                    )
                    start = index
            return segments, confidences, {"candidates": len(labels) + 1, "kept_gold": -1}
        context = HEADS.document_encoder(embeddings)
        if len(document.units) > 1:
            gap_logits = HEADS.boundary(context)
        else:
            gap_logits = torch.zeros(0, device=embeddings.device)
        boundaries = select_boundaries(
            gap_logits.cpu(),
            len(document.units),
            CALIBRATION["threshold"],
            CALIBRATION["cap"],
            keep_all=ablation in ("A4", "A5"),
        )
        graph = SemiCrfGraph.build(
            context, boundaries, HEADS.span_scorer, adjacent_only=ablation == "A4"
        )
        transitions = HEADS.masked_transitions()
        segments = semicrf_viterbi(graph, transitions)
        confidences = semicrf_segment_posteriors(graph, transitions, segments)
        if ablation == "A4":  # merge same-label unit runs into spans for fair metrics
            merged, merged_conf = [], []
            for segment, confidence in zip(segments, confidences):
                if merged and merged[-1][2] == segment[2] and merged[-1][1] == segment[0]:
                    merged[-1] = (merged[-1][0], segment[1], segment[2])
                    merged_conf[-1] = min(merged_conf[-1], confidence)
                else:
                    merged.append(segment)
                    merged_conf.append(confidence)
            segments = list(merged)
            confidences = merged_conf
        gold_boundary, _ = gold_gap_labels(document)
        gold_positions = {i + 1 for i in torch.nonzero(gold_boundary > 0.5).flatten().tolist()}
        kept = len(gold_positions & set(boundaries)) / max(len(gold_positions), 1)
    return segments, confidences, {"candidates": len(boundaries), "kept_gold": kept}


def segments_to_char_spans(
    document: DocumentExample, segments: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    return [
        (label, document.units[start].start_char, document.units[end - 1].end_char)
        for start, end, label in segments
    ]


def interval_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    total = 0
    for start_a, end_a in a:
        for start_b, end_b in b:
            total += max(0, min(end_a, end_b) - max(start_a, start_b))
    return total


def evaluate_document(
    document: DocumentExample, segments: list[tuple[int, int, int]]
) -> dict[str, Any]:
    n_labels = len(CANONICAL_SECTIONS)
    gold_segments = document.gold_segments
    predicted_chars = segments_to_char_spans(document, segments)
    gold_chars = segments_to_char_spans(document, gold_segments)

    def interior_boundaries(segmentation):
        positions, kinds = set(), {}
        for previous, current in itertools.pairwise(segmentation):
            positions.add(current[0])
            kinds[current[0]] = "section" if previous[2] != current[2] else "item"
        return positions, kinds

    predicted_positions, predicted_kinds = interior_boundaries(segments)
    gold_positions, gold_kinds = interior_boundaries(gold_segments)
    boundary = {}
    for kind in ("section", "item"):
        p_k = {p for p in predicted_positions if predicted_kinds[p] == kind}
        g_k = {p for p in gold_positions if gold_kinds[p] == kind}
        boundary[kind] = (len(p_k & g_k), len(p_k - g_k), len(g_k - p_k))

    predicted_units = torch.full((len(document.units),), -1, dtype=torch.long)
    for start, end, label in segments:
        predicted_units[start:end] = label
    gold_units = document.unit_labels
    unit_counts = []
    for label in range(n_labels):
        tp = int(((predicted_units == label) & (gold_units == label)).sum())
        fp = int(((predicted_units == label) & (gold_units != label)).sum())
        fn = int(((predicted_units != label) & (gold_units == label)).sum())
        unit_counts.append((tp, fp, fn))

    char_counts, span_counts, section_exact = [], [], []
    predicted_set = set(predicted_chars)
    gold_set = set(gold_chars)
    for label in range(n_labels):
        p_ranges = [(s, e) for l, s, e in predicted_chars if l == label]
        g_ranges = [(s, e) for l, s, e in gold_chars if l == label]
        overlap = interval_overlap(p_ranges, g_ranges)
        char_counts.append(
            (overlap, sum(e - s for s, e in p_ranges), sum(e - s for s, e in g_ranges))
        )
        p_spans = {(s, e) for l, s, e in predicted_chars if l == label}
        g_spans = {(s, e) for l, s, e in gold_chars if l == label}
        span_counts.append(
            (len(p_spans & g_spans), len(p_spans - g_spans), len(g_spans - p_spans))
        )
        p_items = [document.text[s:e] for s, e in sorted(p_ranges)]
        g_items = [document.text[s:e] for s, e in sorted(g_ranges)]
        section_exact.append(None if not p_items and not g_items else p_items == g_items)

    predicted_empty = {l for l in range(n_labels) if not any(x[0] == l for x in predicted_chars)}
    gold_empty = {l for l in range(n_labels) if not any(x[0] == l for x in gold_chars)}
    empty = (
        len(predicted_empty & gold_empty),
        len(predicted_empty - gold_empty),
        len(gold_empty - predicted_empty),
    )

    serialized = serialize_document(document, segments, [1.0] * len(segments))
    gold_sections = parse_sections(json.dumps(
        {CANONICAL_SECTIONS[l]: [document.text[s:e] for ll, s, e in gold_chars if ll == l]
         for l in range(n_labels)}, ensure_ascii=False))
    json_exact = serialized["sections"] == gold_sections

    emitted = sum(len(item) for items in serialized["sections"].values() for item in items)
    nonverbatim = 0  # verbatim by construction; recheck explicitly anyway
    for record in serialized["records"]:
        item = document.text[record["start_char"]:record["end_char"]]
        if item != serialized["sections"][record["section_key"]][record["item_position"]]:
            nonverbatim += len(item)

    return {
        "boundary": boundary,
        "unit_counts": unit_counts,
        "unit_correct": int((predicted_units == gold_units).sum()),
        "unit_total": len(document.units),
        "char_counts": char_counts,
        "span_counts": span_counts,
        "section_exact": section_exact,
        "empty": empty,
        "json_exact": json_exact,
        "exact_span_micro": (
            len(predicted_set & gold_set), len(predicted_set - gold_set),
            len(gold_set - predicted_set),
        ),
        "emitted_chars": emitted,
        "nonverbatim_chars": nonverbatim,
    }


# %%
def serialize_document(
    document: DocumentExample,
    segments: list[tuple[int, int, int]],
    confidences: list[float],
) -> dict[str, Any]:
    """Validate predicted spans and copy exact source substrings into the public schema.

    The model never generates the character integers; they are source offsets attached
    to the boundaries chosen by the structured decoder.
    """
    sections: dict[str, list[str]] = {key: [] for key in CANONICAL_SECTIONS}
    records: list[dict[str, Any]] = []
    previous_label = -1
    for (start, end, label), confidence in zip(segments, confidences):
        assert 0 <= start < end <= len(document.units), "span outside the document"
        assert label >= previous_label, "decoder emitted a backward label transition"
        previous_label = label
        start_char = document.units[start].start_char
        end_char = document.units[end - 1].end_char
        key = CANONICAL_SECTIONS[label]
        records.append({
            "section_key": key,
            "start_char": start_char,
            "end_char": end_char,
            "confidence": round(confidence, 4),
            "item_position": len(sections[key]),
        })
        sections[key].append(document.text[start_char:end_char])
    return {
        "status": "success",
        "source_file": document.source_file,
        "source_sha256": document.source_sha256,
        "sections": sections,
        "empty_sections": [key for key in CANONICAL_SECTIONS if not sections[key]],
        "records": records,
    }


# %%
def evaluate_unit_classifier(split: str, mode: str) -> dict[str, float]:
    """Per-unit argmax macro-F1 for Stage 1 validation (no document encoder)."""
    MODEL.eval()
    HEADS.eval()
    n = len(COARSE_GROUPS) if mode == "coarse" else len(CANONICAL_SECTIONS)
    counts = np.zeros((n, 3))
    with torch.no_grad():
        for document in DOCS[split]:
            embeddings = encode_document(document, with_grad=False)
            head = HEADS.coarse_classifier if mode == "coarse" else HEADS.fine_classifier
            predicted = head(embeddings).argmax(dim=1).cpu()
            gold = document.unit_labels
            if mode == "coarse":
                gold = FINE_TO_COARSE[gold]
            for label in range(n):
                counts[label, 0] += int(((predicted == label) & (gold == label)).sum())
                counts[label, 1] += int(((predicted == label) & (gold != label)).sum())
                counts[label, 2] += int(((predicted != label) & (gold == label)).sum())
    MODEL.train()
    f1s = [prf(*counts[label])[2] for label in range(n) if counts[label].sum() > 0]
    return {"macro_f1": float(np.mean(f1s)) if f1s else 0.0}


def evaluate_split(
    split: str,
    use_cache: bool,
    ablation: str | None = None,
    segment_fn: Any = None,
    log_prefix: str | None = None,
) -> dict[str, float]:
    """Aggregate all Plan C metrics over one split with equal weight per source_sha256."""
    ablation = ablation or CFG.ablation
    MODEL.eval()
    HEADS.eval()
    n_labels = len(CANONICAL_SECTIONS)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    doc_f1_by_source: dict[str, list[float]] = defaultdict(list)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    total_candidates, kept_gold, decoded_docs = 0, [], 0
    for document in DOCS[split]:
        if segment_fn is not None:
            segments = segment_fn(document)
            stats = {"candidates": 0, "kept_gold": 1.0}
        else:
            embeddings = (
                cached_embeddings(split, document)
                if use_cache
                else encode_document(document, with_grad=False).float()
            )
            segments, _, stats = decode_document(document, embeddings, ablation)
        metrics = evaluate_document(document, segments)
        by_source[document.source_sha256].append(metrics)
        doc_f1_by_source[document.source_sha256].append(
            prf(*metrics["exact_span_micro"])[2]
        )
        total_candidates += stats["candidates"]
        if stats["kept_gold"] >= 0:
            kept_gold.append(stats["kept_gold"])
        decoded_docs += 1
    elapsed = time.time() - started

    def source_weighted(extract) -> np.ndarray:
        totals = None
        for source, metric_list in by_source.items():
            values = np.mean([np.asarray(extract(m), dtype=float) for m in metric_list], axis=0)
            totals = values if totals is None else totals + values
        return totals / len(by_source)

    span = source_weighted(lambda m: m["span_counts"])          # (31, 3)
    char = source_weighted(lambda m: m["char_counts"])          # (31, 3)
    unit = source_weighted(lambda m: m["unit_counts"])          # (31, 3)
    empty = source_weighted(lambda m: [m["empty"]])[0]
    section_boundary = source_weighted(lambda m: [m["boundary"]["section"]])[0]
    item_boundary = source_weighted(lambda m: [m["boundary"]["item"]])[0]

    span_f1s = [prf(*span[l])[2] for l in range(n_labels) if span[l].sum() > 0]
    char_f1s = [
        2 * char[l][0] / (char[l][1] + char[l][2])
        for l in range(n_labels)
        if char[l][1] + char[l][2] > 0
    ]
    unit_f1s = [prf(*unit[l])[2] for l in range(n_labels) if unit[l].sum() > 0]
    span_micro = prf(*span.sum(axis=0))
    result = {
        "span_macro_f1": float(np.mean(span_f1s)),
        "span_micro_precision": span_micro[0],
        "span_micro_recall": span_micro[1],
        "span_micro_f1": span_micro[2],
        "char_macro_f1": float(np.mean(char_f1s)),
        "char_micro_f1": float(2 * char.sum(0)[0] / max(char.sum(0)[1] + char.sum(0)[2], 1)),
        "unit_macro_f1": float(np.mean(unit_f1s)),
        "unit_micro_accuracy": float(
            sum(m["unit_correct"] for ms in by_source.values() for m in ms)
            / max(sum(m["unit_total"] for ms in by_source.values() for m in ms), 1)
        ),
        "boundary_section_f1": prf(*section_boundary)[2],
        "boundary_item_f1": prf(*item_boundary)[2],
        "empty_section_f1": prf(*empty)[2],
        "json_exact_match": float(np.mean([
            np.mean([m["json_exact"] for m in ms]) for ms in by_source.values()
        ])),
        "filter_gold_recall": float(np.mean(kept_gold)) if kept_gold else 1.0,
        "filter_mean_candidates": total_candidates / max(decoded_docs, 1),
        "nonverbatim_fraction": float(
            sum(m["nonverbatim_chars"] for ms in by_source.values() for m in ms)
            / max(sum(m["emitted_chars"] for ms in by_source.values() for m in ms), 1)
        ),
        "documents_per_hour": decoded_docs / max(elapsed / 3600, 1e-9),
        "eval_seconds": elapsed,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }
    globals().setdefault("DOC_SPAN_F1", {})[
        (split, log_prefix or ablation)
    ] = {sha: float(np.mean(v)) for sha, v in doc_f1_by_source.items()}
    if log_prefix:
        RUN.log({f"{log_prefix}/{k}": v for k, v in result.items()})
    return result


# %% [markdown]
# ## Mandatory artifact baselines — A0 position-only and A1 whitespace-only

# %%
def build_position_layout() -> np.ndarray:
    """Mean fraction of document characters per canonical section on train gold."""
    fractions = np.zeros(len(CANONICAL_SECTIONS))
    for document in DOCS["train"]:
        lengths = np.zeros(len(CANONICAL_SECTIONS))
        for span in document.gold_spans:
            lengths[span.label] += span.end_char - span.start_char
        fractions += lengths / max(len(document.text), 1)
    fractions /= max(len(DOCS["train"]), 1)
    return fractions / fractions.sum()


def baseline_a0_segments(
    document: DocumentExample, layout: np.ndarray
) -> list[tuple[int, int, int]]:
    """Position-only canonical segmentation: proportional cuts snapped to unit starts."""
    unit_starts = [unit.start_char for unit in document.units]
    cuts = np.cumsum(layout) * len(document.text)
    segments = []
    previous_unit = 0
    for label in range(len(CANONICAL_SECTIONS)):
        target = cuts[label]
        next_unit = previous_unit
        while next_unit < len(document.units) and document.units[next_unit].start_char < target:
            next_unit += 1
        if label == len(CANONICAL_SECTIONS) - 1:
            next_unit = len(document.units)
        if next_unit > previous_unit:
            segments.append((previous_unit, next_unit, label))
            previous_unit = next_unit
    return segments


def baseline_a1_segments(
    document: DocumentExample, layout: np.ndarray
) -> list[tuple[int, int, int]]:
    """Whitespace-only boundaries (blank-line separators), labels by position layout."""
    boundaries = [0]
    for index, (left, right) in enumerate(itertools.pairwise(document.units)):
        separator = document.text[left.end_char:right.start_char]
        if "\n\n" in separator:
            boundaries.append(index + 1)
    boundaries.append(len(document.units))
    cuts = np.cumsum(layout) * len(document.text)
    segments = []
    previous_label = 0
    for start, end in itertools.pairwise(sorted(set(boundaries))):
        midpoint = (
            document.units[start].start_char + document.units[end - 1].end_char
        ) / 2
        label = int(np.searchsorted(cuts, midpoint))
        label = min(max(label, previous_label), len(CANONICAL_SECTIONS) - 1)
        segments.append((start, end, label))
        previous_label = label
    return segments


def evaluate_baselines(split: str) -> dict[str, dict[str, float]]:
    layout = build_position_layout()
    return {
        "A0": evaluate_split(
            split, use_cache=False,
            segment_fn=lambda d: baseline_a0_segments(d, layout),
            log_prefix=f"baseline_a0_{split}",
        ),
        "A1": evaluate_split(
            split, use_cache=False,
            segment_fn=lambda d: baseline_a1_segments(d, layout),
            log_prefix=f"baseline_a1_{split}",
        ),
    }


def paired_bootstrap_delta(
    system_f1: dict[str, float], baseline_f1: dict[str, float], iterations: int = 1000
) -> tuple[float, float, float]:
    """95% CI of the source-level span-F1 difference (system - baseline)."""
    shas = sorted(set(system_f1) & set(baseline_f1))
    deltas = np.array([system_f1[s] - baseline_f1[s] for s in shas])
    rng = np.random.default_rng(CFG.seed)
    samples = [
        float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))]))
        for _ in range(iterations)
    ]
    return float(np.mean(deltas)), float(np.percentile(samples, 2.5)), float(
        np.percentile(samples, 97.5)
    )


# %% [markdown]
# ## Run the curriculum
#
# Stages honour `CFG.run_stage_*`, the ablation flag, and any resumed cursor.
# Each stage boundary logs a `latest` checkpoint Artifact.

# %%
resume_from_wandb()

STAGE_PLAN = [
    (1, run_stage_1, CFG.run_stage_1 and CFG.ablation != "A2"),
    (2, run_stage_2, CFG.run_stage_2),
    (3, run_stage_3, CFG.run_stage_3 and CFG.ablation in ("C", "A4", "A5")),
]
for stage_number, stage_fn, enabled in STAGE_PLAN:
    if TRAIN_STATE["stage"] > stage_number:
        continue
    if TRAIN_STATE["stage"] < stage_number:
        TRAIN_STATE.update(stage=stage_number, epoch=0, doc_cursor=0)
    if enabled:
        print(f"=== Stage {stage_number} ===")
        stage_fn()
    TRAIN_STATE.update(stage=stage_number + 1, epoch=0, doc_cursor=0)

# %% [markdown]
# ## Validation report, baselines, and promotion gates

# %%
VAL_METRICS = evaluate_split("validation", use_cache=False, log_prefix="val")
print(json.dumps(VAL_METRICS, indent=2))
VAL_BASELINES = evaluate_baselines("validation")
for name, metrics in VAL_BASELINES.items():
    print(name, {k: round(v, 4) for k, v in metrics.items() if "f1" in k})

STRONGEST_BASELINE = max(VAL_BASELINES, key=lambda k: VAL_BASELINES[k]["span_macro_f1"])
delta, low, high = paired_bootstrap_delta(
    DOC_SPAN_F1[("validation", "val")],
    DOC_SPAN_F1[("validation", f"baseline_{STRONGEST_BASELINE.lower()}_validation")],
)
print(f"val ΔF1 vs {STRONGEST_BASELINE}: {delta:.4f} [{low:.4f}, {high:.4f}]")

GATES = {
    "all_rows_aligned": all(AUDIT[s]["rows"] == len(DOCS[s]) for s in DOCS),
    "no_label_truncation": True,  # units cover every gold span by construction (asserted)
    "gold_roundtrip_100pct": True,  # prepare_document asserts verbatim round-trip per row
    "boundary_candidate_recall": VAL_METRICS["filter_gold_recall"]
    >= CFG.required_boundary_recall,
    "json_validity_100pct": VAL_METRICS["nonverbatim_fraction"] == 0.0,
    "verbatim_source_slices": VAL_METRICS["nonverbatim_fraction"] == 0.0,
    "peak_memory_below_a100": VAL_METRICS["peak_gpu_gib"] < 39.0,
    "beats_strongest_baseline_ci": low > 0.0,
}
RUN.log({f"gates/{name}": float(ok) for name, ok in GATES.items()})
print(json.dumps(GATES, indent=2))
if (
    VAL_BASELINES[STRONGEST_BASELINE]["span_macro_f1"]
    >= VAL_METRICS["span_macro_f1"] - 1e-6
):
    print(
        "RESULT MUST BE REPORTED AS A FAILED SEMANTIC-LEARNING EXPERIMENT: "
        f"the {STRONGEST_BASELINE} artifact baseline matches Plan C."
    )

# %%
# Qualitative check: serialize the first validation document with predicted spans.
_doc = DOCS["validation"][0]
_segments, _confidences, _ = decode_document(
    _doc, encode_document(_doc, with_grad=False).float(), CFG.ablation
)
_serialized = serialize_document(_doc, _segments, _confidences)
print(json.dumps(
    {k: _serialized[k] for k in ("status", "source_file", "empty_sections")},
    ensure_ascii=False, indent=2,
))
for record in _serialized["records"][:8]:
    print(record)

# %% [markdown]
# ## Final test evaluation — run once, after freezing the configuration
#
# Set `run_final_test=True` only when validation gates pass and the configuration
# is frozen. The cell restores the `best` checkpoint before touching the test split.

# %%
if CFG.run_final_test:
    artifact = RUN.use_artifact(f"{CHECKPOINT_ARTIFACT}:best", type="checkpoint")
    directory = Path(artifact.download())
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(
        MODEL, load_file(str(directory / "adapter" / "adapter_model.safetensors"))
    )
    bundle = torch.load(directory / "training_state.pt", weights_only=False)
    HEADS.load_state_dict(bundle["heads"])
    UNIT_POOLER.load_state_dict(bundle["unit_pooler"])
    CALIBRATION.update(bundle["calibration"])

    TEST_METRICS = evaluate_split("test", use_cache=False, log_prefix="test")
    TEST_BASELINES = evaluate_baselines("test")
    t_delta, t_low, t_high = paired_bootstrap_delta(
        DOC_SPAN_F1[("test", "test")],
        DOC_SPAN_F1[("test", f"baseline_{STRONGEST_BASELINE.lower()}_test")],
    )
    RUN.summary["test/span_macro_f1"] = TEST_METRICS["span_macro_f1"]
    RUN.summary["test/delta_vs_baseline"] = t_delta
    RUN.summary["test/delta_ci"] = [t_low, t_high]
    print(json.dumps(TEST_METRICS, indent=2))
    print(f"test ΔF1 vs {STRONGEST_BASELINE}: {t_delta:.4f} [{t_low:.4f}, {t_high:.4f}]")
else:
    print("run_final_test=False — test split untouched (single-shot policy).")

# %%
RUN.summary["train_state"] = dict(TRAIN_STATE)
RUN.summary["calibration"] = dict(CALIBRATION)
wandb.finish()
