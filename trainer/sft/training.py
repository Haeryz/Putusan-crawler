"""TRL trainer construction and response-only supervision."""

from __future__ import annotations

import os
from typing import Any

from .config import TrainingConfig
from .data import get_text_tokenizer


def build_trainer(
    model: Any,
    tokenizer_or_processor: Any,
    train_dataset: Any,
    eval_dataset: Any,
    max_length: int,
    config: TrainingConfig,
) -> Any:
    """Create the notebook-equivalent trainer and mask prompt tokens."""

    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    FastLanguageModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=get_text_tokenizer(tokenizer_or_processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            optim="adamw_8bit",
            weight_decay=config.weight_decay,
            lr_scheduler_type="linear",
            seed=config.seed,
            output_dir=str(config.output_dir),
            report_to=config.report_to,
            max_length=max_length,
            packing=False,
            bf16=True,
            fp16=False,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            per_device_eval_batch_size=1,
            prediction_loss_only=True,
            tf32=True,
            ddp_find_unused_parameters=False,
            ddp_broadcast_buffers=False,
            dataloader_num_workers=config.dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=config.dataloader_num_workers > 0,
            dataloader_prefetch_factor=(
                config.dataloader_prefetch_factor
                if config.dataloader_num_workers > 0
                else None
            ),
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    train_retention = len(trainer.train_dataset) / len(train_dataset)
    eval_retention = len(trainer.eval_dataset) / len(eval_dataset)
    if train_retention < config.minimum_response_retention:
        raise RuntimeError(
            f"Only {train_retention:.2%} of training rows retained responses"
        )
    print(
        "response-supervised rows retained: "
        f"train={train_retention:.2%}, eval={eval_retention:.2%}"
    )
    return trainer
