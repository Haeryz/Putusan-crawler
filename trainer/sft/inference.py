"""Structured-extraction generation helpers."""

from __future__ import annotations

from typing import Any, Sequence

from .data import get_text_tokenizer
from .transformer import backfill_architectures


def generate(
    model: Any,
    tokenizer_or_processor: Any,
    messages: Sequence[dict[str, str]],
    *,
    max_new_tokens: int = 2_048,
    temperature: float = 0.3,
    min_p: float = 0.1,
    stream: bool = True,
) -> Any:
    """Generate requested JSON from system and user messages."""

    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    backfill_architectures(model)
    tokenizer = get_text_tokenizer(tokenizer_or_processor)
    inputs = tokenizer.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    kwargs: dict[str, Any] = {}
    if stream:
        from transformers import TextStreamer

        kwargs["streamer"] = TextStreamer(tokenizer, skip_prompt=True)
    return model.generate(
        **inputs,
        **kwargs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
        pad_token_id=tokenizer.eos_token_id,
    )

