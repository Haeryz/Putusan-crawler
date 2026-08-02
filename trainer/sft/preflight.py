"""Fail-fast environment, Hub-access, dataset, and model smoke checks."""

from __future__ import annotations

import argparse
import gc
from importlib import metadata
import os
from typing import Any, Sequence

from .config import MODEL_PROFILES, TRAINING_ORDER, ModelConfig
from .data import format_messages, get_text_tokenizer


MINIMUM_VERSIONS = {
    "transformers": "5.5.0",
    "trl": "0.28.0",
    "unsloth": "2026.4.2",
    "huggingface_hub": "1.5.0",
    "datasets": "4.3.0",
}


def validate_versions() -> dict[str, str]:
    """Require the dependency floor used by the Gemma 4 text notebook."""

    from packaging.version import Version

    installed: dict[str, str] = {}
    for package, minimum in MINIMUM_VERSIONS.items():
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Required package {package} is not installed") from error
        installed[package] = found
        if Version(found) < Version(minimum):
            raise RuntimeError(
                f"{package}>={minimum} is required; found {found}"
            )
    return installed


def validate_environment() -> None:
    """Check credentials and persistent cache configuration."""

    missing = [
        name
        for name in ("HF_TOKEN", "HF_HOME", "WANDB_API_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def infer_modalities(config_dict: dict[str, Any]) -> set[str]:
    """Infer input towers from a Transformers config dictionary."""

    modalities = {"text"}
    if "vision_config" in config_dict:
        modalities.update(("image", "video"))
    if "audio_config" in config_dict:
        modalities.add("audio")
    return modalities


def validate_hub_model(profile: ModelConfig) -> None:
    """Validate repository access, architecture, towers, and chat markers."""

    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    token = os.environ["HF_TOKEN"]
    HfApi(token=token).model_info(profile.model_name)
    hub_config = AutoConfig.from_pretrained(profile.model_name, token=token)
    architectures = tuple(getattr(hub_config, "architectures", ()) or ())
    if profile.architecture not in architectures:
        raise RuntimeError(
            f"{profile.model_name} architecture changed: {architectures!r}"
        )
    actual_modalities = infer_modalities(hub_config.to_dict())
    expected_modalities = set(profile.input_modalities)
    if actual_modalities != expected_modalities:
        raise RuntimeError(
            f"{profile.model_name} modality mismatch: expected "
            f"{sorted(expected_modalities)}, found {sorted(actual_modalities)}"
        )

    if profile.input_modalities == ("text",):
        formatter = AutoTokenizer.from_pretrained(
            profile.model_name, token=token
        )
    else:
        formatter = AutoProcessor.from_pretrained(
            profile.model_name, token=token
        )
    rendered = formatter.apply_chat_template(
        [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if profile.strip_bos_from_formatted_text:
        rendered = rendered.removeprefix("<bos>")
    if (
        profile.instruction_part not in rendered
        or profile.response_part not in rendered
    ):
        raise RuntimeError(
            f"{profile.model_name} chat template no longer matches its "
            "response-mask markers"
        )


def validate_dataset_sample(
    repository: str = "Haeryz/putusan-structured-extraction",
    subset: str = "sft",
) -> None:
    """Stream one real row and require string-only chat messages."""

    from datasets import load_dataset

    dataset = load_dataset(
        repository,
        subset,
        split="train",
        streaming=True,
    )
    row = next(iter(dataset))
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("Dataset sample has no messages conversation")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise RuntimeError("Dataset sample contains non-text message content")


def validate_wandb_access() -> None:
    """Verify that the configured W&B key can reach the API."""

    import wandb

    if not wandb.login(
        key=os.environ["WANDB_API_KEY"], relogin=False, verify=True
    ):
        raise RuntimeError("Weights & Biases authentication failed")


def deep_model_smoke_test(profile: ModelConfig) -> None:
    """Load, attach LoRA, tokenize text, and run one short forward pass."""

    import torch

    from .transformer import attach_lora, load_base_model

    model, formatter = load_base_model(profile)
    model = attach_lora(model, profile)
    rendered = format_messages(
        {
            "messages": [[
                {"role": "user", "content": "Ringkas putusan ini."},
                {"role": "assistant", "content": "Putusan diringkas."},
            ]]
        },
        formatter,
        profile,
    )["text"][0]
    tokenizer = get_text_tokenizer(formatter)
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded, labels=encoded["input_ids"])
    if not torch.isfinite(output.loss):
        raise RuntimeError(f"{profile.model_name} smoke loss is not finite")
    del output, encoded, tokenizer, formatter, model
    gc.collect()
    torch.cuda.empty_cache()


def run_preflight(
    deep: bool = False,
    dataset: str = "Haeryz/putusan-structured-extraction",
    dataset_config: str = "sft",
    model_keys: Sequence[str] = TRAINING_ORDER,
) -> None:
    """Run checks for the outstanding training profiles."""

    validate_environment()
    versions = validate_versions()
    print("Dependency versions:", ", ".join(
        f"{name}={version}" for name, version in versions.items()
    ))
    for key in model_keys:
        profile = MODEL_PROFILES[key]
        print(f"Checking Hugging Face metadata: {profile.model_name}")
        validate_hub_model(profile)
    validate_dataset_sample(dataset, dataset_config)
    validate_wandb_access()
    if deep:
        from .transformer import validate_hardware

        validate_hardware(MODEL_PROFILES[model_keys[0]])
        for key in model_keys:
            profile = MODEL_PROFILES[key]
            print(f"Deep model smoke test: {profile.model_name}")
            deep_model_smoke_test(profile)
    print("SFT preflight passed.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also load each 4-bit model, attach LoRA, and run a forward pass",
    )
    parser.add_argument(
        "--dataset", default="Haeryz/putusan-structured-extraction"
    )
    parser.add_argument("--dataset-config", default="sft")
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_PROFILES),
        help="Profile to check; repeat it (default: Gemma and DeepSeek)",
    )
    args = parser.parse_args(argv)
    run_preflight(
        deep=args.deep,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        model_keys=tuple(args.model or TRAINING_ORDER),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
