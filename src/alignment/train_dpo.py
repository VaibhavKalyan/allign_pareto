"""
Stage 2 of M2: DPO safety alignment (spec §18-19).
Starts from the helpful-only SFT model (models.load_sft_model) and trains a new LoRA adapter with TRL's DPOTrainer
on hh-rlhf harmless-base pairs. The reference policy is the SFT model itself (the adapter disabled), so the adapter
holds only the safety-alignment update; adapters.set_peft_lora_lambda scales it to any lambda.
"""
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.alignment.models import (
    cast_trainable_to_fp32, lora_config, render_prompt, supported_kwargs, trainer_precision_kwargs,
)


def build_dpo_dataset(pairs: pd.DataFrame, tokenizer, max_prompt_length: int, max_length: int, enable_thinking: bool = False):
    """Rendered (prompt, chosen, rejected) strings; pairs over the length limits are dropped rather than truncated."""
    from datasets import Dataset
    eos = tokenizer.eos_token
    rows, dropped = [], 0
    for prompt, chosen, rejected in zip(pairs["prompt"], pairs["chosen"], pairs["rejected"]):
        prompt_text = render_prompt(prompt, tokenizer, enable_thinking)
        n_prompt = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        n_reply = max(len(tokenizer(r + eos, add_special_tokens=False)["input_ids"]) for r in (chosen, rejected))
        if n_prompt > max_prompt_length or n_prompt + n_reply > max_length:
            dropped += 1
            continue
        rows.append({"prompt": prompt_text, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(rows), dropped


def make_dpo_trainer(model, tokenizer, train_ds, eval_ds, cfg: Dict, output_dir: Path, max_steps: Optional[int] = None):
    from trl import DPOConfig, DPOTrainer

    dcfg = cfg["dpo"]
    args = DPOConfig(**supported_kwargs(DPOConfig, {
        "output_dir": str(output_dir),
        "seed": int(cfg.get("seed", 42)),
        "beta": float(dcfg["beta"]),
        "loss_type": dcfg["loss_type"],
        "learning_rate": float(dcfg["learning_rate"]),
        "num_train_epochs": float(dcfg["num_train_epochs"]),
        "max_steps": max_steps if max_steps else -1,
        "per_device_train_batch_size": int(dcfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(dcfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(dcfg["gradient_accumulation_steps"]),
        "warmup_ratio": float(dcfg["warmup_ratio"]),
        "max_length": int(dcfg["max_length"]),
        "logging_steps": int(dcfg["logging_steps"]),
        "eval_strategy": "steps",
        "eval_steps": int(dcfg["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(dcfg["save_steps"]),
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        **trainer_precision_kwargs(),
    }))
    trainer = DPOTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tokenizer, peft_config=lora_config(cfg["lora"]))
    cast_trainable_to_fp32(trainer.model)
    return trainer


def pairs_to_parquet_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    """Message lists stored as JSON strings so the pairs can be saved as Parquet."""
    return pairs.assign(prompt=pairs["prompt"].map(json.dumps))
