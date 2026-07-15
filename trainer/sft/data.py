"""Dataset loading, chat formatting, and context-length selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

from .config import DataConfig


@dataclass(frozen=True)
class LengthProfile:
    """Summary used to choose the trainer's context window."""

    p50: int
    p90: int
    p95: int
    maximum: int
    max_length: int
    coverage: float


def get_text_tokenizer(tokenizer_or_processor: Any) -> Any:
    """Return the inner tokenizer when Unsloth supplies a processor."""

    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def load_splits(config: DataConfig) -> tuple[Any, Any]:
    """Load training and validation splits from Hugging Face."""

    from datasets import load_dataset

    train = load_dataset(config.repository, config.subset, split=config.train_split)
    validation = load_dataset(
        config.repository, config.subset, split=config.validation_split
    )
    return train, validation


def format_messages(
    examples: dict[str, Sequence[Any]], tokenizer: Any
) -> dict[str, list[str]]:
    """Render batched messages with the model chat template."""

    return {
        "text": [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in examples["messages"]
        ]
    }


def format_dataset(dataset: Any, tokenizer_or_processor: Any) -> Any:
    """Add the text field consumed by SFTTrainer."""

    tokenizer = get_text_tokenizer(tokenizer_or_processor)
    return dataset.map(
        lambda examples: format_messages(examples, tokenizer), batched=True
    )


def measure_token_lengths(
    texts: Sequence[str], tokenizer: Any, batch_size: int = 256
) -> list[int]:
    """Measure formatted examples with batched tokenizer calls."""

    # The fast tokenizer's Rust threads sit idle unless this is set: importing
    # transformers disables them so forked dataloader workers cannot deadlock.
    # Nothing is forked here, and the batches below are what they parallelize.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    total = len(texts)
    lengths: list[int] = []
    started = time.monotonic()
    for start in range(0, total, batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], add_special_tokens=False
        )["input_ids"]
        lengths.extend(len(token_ids) for token_ids in encoded)

        elapsed = time.monotonic() - started
        rate = len(lengths) / elapsed if elapsed else 0.0
        remaining = (
            f", ~{(total - len(lengths)) / rate / 60:.1f} min left"
            if rate > 0 and len(lengths) < total
            else ""
        )
        print(
            f"  {len(lengths)}/{total} examples "
            f"({100 * len(lengths) / total:5.1f}%) at {rate:.0f}/s{remaining}",
            flush=True,
        )
    return lengths


def choose_max_length(
    lengths: Iterable[int],
    context_cap: int,
    percentile: int = 95,
    multiple: int = 256,
) -> LengthProfile:
    """Round the selected percentile upward without exceeding the model cap."""

    values = sorted(int(length) for length in lengths)
    if not values:
        raise ValueError("Cannot choose max_length from an empty dataset")
    if context_cap <= 0 or multiple <= 0:
        raise ValueError("context_cap and multiple must be positive")

    def linear_percentile(percent: int) -> int:
        if not 0 <= percent <= 100:
            raise ValueError("percentile must be between 0 and 100")
        position = (len(values) - 1) * percent / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        interpolated = values[lower] + (
            values[upper] - values[lower]
        ) * (position - lower)
        return int(interpolated)

    selected = linear_percentile(percentile)
    max_length = min(
        ((selected + multiple - 1) // multiple) * multiple, context_cap
    )
    profile = LengthProfile(
        p50=linear_percentile(50),
        p90=linear_percentile(90),
        p95=linear_percentile(95),
        maximum=values[-1],
        max_length=int(max_length),
        coverage=sum(value <= max_length for value in values) / len(values),
    )
    if selected > max_length:
        raise ValueError(
            f"The {percentile}th percentile ({selected}) exceeds context cap "
            f"{context_cap}"
        )
    return profile


def length_cache_path(config: DataConfig) -> Path:
    """Return the file holding one cached token count per training row."""

    return Path(config.cache_dir) / "token_lengths.npy"


def load_cached_lengths(dataset: Any, config: DataConfig) -> list[int] | None:
    """Return cached counts, or None when they are absent or stale."""

    import numpy as np

    cache_path = length_cache_path(config)
    if not cache_path.exists():
        return None
    try:
        cached = np.load(cache_path)
    except (OSError, ValueError):
        return None
    if len(cached) != len(dataset):
        return None
    return [int(value) for value in cached]


def load_or_measure_lengths(
    dataset: Any, tokenizer_or_processor: Any, config: DataConfig
) -> list[int]:
    """Use a local cache, measuring again when its row count is stale."""

    import numpy as np

    cache_path = length_cache_path(config)

    def load_valid_cache() -> list[int] | None:
        return load_cached_lengths(dataset, config)

    cached_lengths = load_valid_cache()
    if cached_lengths is not None:
        return cached_lengths

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and rank != 0:
        deadline = (
            time.monotonic() + config.distributed_cache_timeout_seconds
        )
        while time.monotonic() < deadline:
            cached_lengths = load_valid_cache()
            if cached_lengths is not None:
                return cached_lengths
            time.sleep(1)
        raise TimeoutError(
            f"Rank {rank} timed out waiting for rank 0 to create {cache_path}"
        )

    lengths = measure_token_lengths(
        dataset["text"],
        get_text_tokenizer(tokenizer_or_processor),
        config.tokenization_batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f"{cache_path.stem}.rank0{cache_path.suffix}"
    )
    np.save(temporary_path, np.asarray(lengths, dtype=np.int64))
    os.replace(temporary_path, cache_path)
    return lengths
