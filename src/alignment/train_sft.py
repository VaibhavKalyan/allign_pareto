"""
Stage 1 of M2: supervised fine-tuning of Qwen3-0.6B-Base into a helpful-only chat model.
A LoRA adapter is trained on human-written helpful replies (sft_data.py; loss on the reply only), then merged into the
base weights by models.load_sft_model. This model is lambda = 0: it follows instructions but has had no safety
training, so stage 2 (DPO) isolates the effect of safety alignment.
"""
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.alignment.models import (
    cast_trainable_to_fp32, lora_config, render_prompt, supported_kwargs, trainer_precision_kwargs,
)


def build_sft_dataset(examples: pd.DataFrame, tokenizer, max_length: int, enable_thinking: bool = False):
    """(prompt, completion) strings; examples over max_length are dropped rather than truncated."""
    from datasets import Dataset
    eos = tokenizer.eos_token
    rows, dropped = [], 0
    for prompt, reply in zip(examples["prompt"], examples["reply"]):
        prompt_text = render_prompt(prompt, tokenizer, enable_thinking)
        if len(tokenizer(prompt_text + reply + eos, add_special_tokens=False)["input_ids"]) > max_length:
            dropped += 1
            continue
        rows.append({"prompt": prompt_text, "completion": reply})
    return Dataset.from_list(rows), dropped


def make_sft_trainer(model, tokenizer, train_ds, eval_ds, cfg: Dict, output_dir: Path, max_steps: Optional[int] = None):
    from trl import SFTConfig, SFTTrainer

    scfg = cfg["sft"]
    args = SFTConfig(**supported_kwargs(SFTConfig, {
        "output_dir": str(output_dir),
        "seed": int(cfg.get("seed", 42)),
        "learning_rate": float(scfg["learning_rate"]),
        "num_train_epochs": float(scfg["num_train_epochs"]),
        "max_steps": max_steps if max_steps else -1,
        "per_device_train_batch_size": int(scfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(scfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(scfg["gradient_accumulation_steps"]),
        "warmup_ratio": float(scfg["warmup_ratio"]),
        "lr_scheduler_type": "cosine",
        "max_length": int(scfg["max_length"]),
        "completion_only_loss": True,
        "packing": False,
        "eos_token": tokenizer.eos_token,
        "logging_steps": int(scfg["logging_steps"]),
        "eval_strategy": "steps",
        "eval_steps": int(scfg["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(scfg["save_steps"]),
        "save_total_limit": 2,
        "report_to": "none",
        **trainer_precision_kwargs(),
    }))
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tokenizer, peft_config=lora_config(cfg["lora"]))
    cast_trainable_to_fp32(trainer.model)
    return trainer
