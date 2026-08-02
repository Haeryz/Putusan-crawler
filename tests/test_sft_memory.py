from __future__ import annotations

import pytest

from trainer.sft.config import MODEL_PROFILES, run_config_for_model
from trainer.sft.memory import assert_training_memory_fits, estimate_training_memory


@pytest.mark.parametrize(
    ("model_key", "expected_peak_upper_bound"),
    [("gemma", 70.2), ("deepseek", 70.2)],
)
def test_section_profile_fits_conservative_a100_80gb_budget(
    model_key: str, expected_peak_upper_bound: float
) -> None:
    model = MODEL_PROFILES[model_key]
    training = run_config_for_model(model).training

    estimate = assert_training_memory_fits(
        model, training, max_length=8_192, available_vram_gib=78.0
    )

    assert estimate.total_gib < expected_peak_upper_bound
    assert estimate.total_gib < estimate.usable_vram_gib


def test_memory_guard_rejects_unsafe_override_before_training() -> None:
    model = MODEL_PROFILES["gemma"]
    training = run_config_for_model(model).training

    with pytest.raises(RuntimeError, match="lower --per-device-batch-size"):
        assert_training_memory_fits(
            model, training, max_length=49_152, available_vram_gib=78.0
        )


@pytest.mark.parametrize(("model_key", "unsafe_batch"), [("gemma", 18), ("deepseek", 25)])
def test_profile_default_is_largest_safe_integer_microbatch(
    model_key: str, unsafe_batch: int
) -> None:
    model = MODEL_PROFILES[model_key]
    training = run_config_for_model(model).training
    overridden = type(training)(
        **{**training.__dict__, "per_device_train_batch_size": unsafe_batch}
    )

    with pytest.raises(RuntimeError, match="Estimated peak"):
        assert_training_memory_fits(model, overridden, 8_192, 78.0)


def test_estimator_scales_activation_memory_with_microbatch() -> None:
    model = MODEL_PROFILES["deepseek"]
    training = run_config_for_model(model).training
    half = type(training)(
        **{
            **training.__dict__,
            "per_device_train_batch_size": 12,
        }
    )

    full_estimate = estimate_training_memory(model, training, 8_192, 78.0)
    half_estimate = estimate_training_memory(model, half, 8_192, 78.0)

    assert full_estimate.checkpointed_activations_gib == pytest.approx(
        2 * half_estimate.checkpointed_activations_gib
    )
