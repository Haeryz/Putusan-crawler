"""Dataset loading, chat formatting, and context-length selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

from .config import DataConfig, ModelConfig


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
    examples: dict[str, Sequence[Any]],
    tokenizer_or_processor: Any,
    model_config: ModelConfig,
) -> dict[str, list[str]]:
    """Render text-only messages with the selected model's chat template."""

    texts: list[str] = []
    for conversation in examples["messages"]:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"{model_config.profile_name} SFT accepts text-only message "
                    "content; multimodal content blocks are not allowed"
                )
        text = tokenizer_or_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        if model_config.strip_bos_from_formatted_text:
            text = text.removeprefix("<bos>")
        texts.append(text)
    return {"text": texts}


def format_dataset(
    dataset: Any,
    tokenizer_or_processor: Any,
    model_config: ModelConfig,
) -> Any:
    """Add the text field consumed by SFTTrainer."""

    return dataset.map(
        lambda examples: format_messages(
            examples, tokenizer_or_processor, model_config
        ),
        batched=True,
    )


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(token_id) for token_id in encoded]


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def truncate_text_preserving_response(
    text: str,
    tokenizer_or_processor: Any,
    model_config: ModelConfig,
    max_length: int,
) -> str:
    """Middle-truncate a chat while retaining both masking markers and target."""

    tokenizer = get_text_tokenizer(tokenizer_or_processor)
    if len(_encode_text(tokenizer, text)) <= max_length:
        return text

    instruction_at = text.find(model_config.instruction_part)
    response_at = text.rfind(model_config.response_part)
    if instruction_at < 0 or response_at < 0 or response_at <= instruction_at:
        raise ValueError(
            f"Cannot safely truncate {model_config.profile_name} row because "
            "its instruction or response marker is missing"
        )

    instruction_end = instruction_at + len(model_config.instruction_part)
    prefix = text[:instruction_end]
    prompt_body = text[instruction_end:response_at]
    response = text[response_at:]
    prefix_ids = _encode_text(tokenizer, prefix)
    prompt_ids = _encode_text(tokenizer, prompt_body)
    response_ids = _encode_text(tokenizer, response)

    # Leave a small boundary-token margin because separately decoded pieces can
    # tokenize a few tokens differently after concatenation.
    token_budget = max(1, max_length - 32)
    prompt_budget = token_budget - len(prefix_ids) - len(response_ids)
    if prompt_budget >= 0:
        head_count = prompt_budget // 3
        tail_count = prompt_budget - head_count
        kept_prompt = (
            _decode_tokens(tokenizer, prompt_ids[:head_count])
            + _decode_tokens(
                tokenizer, prompt_ids[-tail_count:] if tail_count else []
            )
        )
        truncated = prefix + kept_prompt + response
    else:
        # Extremely large targets are rare. Preserve both exact markers and the
        # beginning of the supervised answer rather than returning an all--100 row.
        response_payload = text[response_at + len(model_config.response_part) :]
        marker_ids = _encode_text(tokenizer, model_config.response_part)
        payload_budget = max(
            0, token_budget - len(prefix_ids) - len(marker_ids)
        )
        payload_ids = _encode_text(tokenizer, response_payload)
        truncated = (
            prefix
            + model_config.response_part
            + _decode_tokens(tokenizer, payload_ids[:payload_budget])
        )

    if len(_encode_text(tokenizer, truncated)) > max_length:
        raise RuntimeError("Marker-preserving truncation exceeded max_length")
    return truncated


def truncate_dataset_preserving_responses(
    dataset: Any,
    tokenizer_or_processor: Any,
    model_config: ModelConfig,
    max_length: int,
) -> Any:
    """Apply marker-preserving middle truncation to a formatted dataset."""

    return dataset.map(
        lambda examples: {
            "text": [
                truncate_text_preserving_response(
                    text, tokenizer_or_processor, model_config, max_length
                )
                for text in examples["text"]
            ]
        },
        batched=True,
        desc="Preserving chat markers while truncating long rows",
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
    """Round the selected percentile upward and cap oversized rows."""

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
