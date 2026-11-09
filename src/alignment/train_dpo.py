"""
DPO alignment training for ALIGN-PARETO (M2, spec §18-19).
Qwen3-0.6B (4-bit QLoRA on a CUDA GPU) + LoRA adapter trained with TRL's DPOTrainer on hh-rlhf pairs.
The saved adapter is the full alignment update; `adapters.set_peft_lora_lambda` scales it to any lambda.
"""
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(mcfg: Dict):
    """Base model in 4-bit NF4 on a CUDA GPU when `use_4bit_qlora` is set, else full precision float32."""
    import torch
    from transformers import AutoModelForCausalLM
    kwargs = {}
    if mcfg.get("use_4bit_qlora", True) and has_cuda():
        from transformers import BitsAndBytesConfig
        compute_dtype = getattr(torch, mcfg["bnb_4bit_compute_dtype"])
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mcfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(mcfg["bnb_4bit_use_double_quant"]),
        )
        # Non-quantized layers (embeddings, norms) would otherwise load in the checkpoint's bfloat16
        kwargs["dtype"] = compute_dtype
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.float32
    return AutoModelForCausalLM.from_pretrained(mcfg["base_model_name"], **kwargs)


def is_quantized(model) -> bool:
    return bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model, "quantization_method", None))


def render_prompt(messages: List[Dict[str, str]], tokenizer, enable_thinking: bool = False) -> str:
    """Chat-template text up to the assistant's reply, exactly as M3 will prompt the model."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


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


def _supported(cls, kwargs: Dict) -> Dict:
    """Keep only the arguments this TRL version accepts (names have moved between releases)."""
    params = inspect.signature(cls).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def make_trainer(model, tokenizer, train_ds, eval_ds, cfg: Dict, output_dir: Path, max_steps: Optional[int] = None):
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer

    dcfg, lcfg = cfg["dpo"], cfg["lora"]
    cuda = has_cuda()
    if is_quantized(model):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(
        r=int(lcfg["r"]), lora_alpha=int(lcfg["lora_alpha"]), lora_dropout=float(lcfg["lora_dropout"]),
        target_modules=list(lcfg["target_modules"]), task_type=lcfg["task_type"], bias="none",
    )
    args = DPOConfig(**_supported(DPOConfig, {
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
        "max_prompt_length": int(dcfg["max_prompt_length"]),
        "logging_steps": int(dcfg["logging_steps"]),
        "eval_strategy": "steps",
        "eval_steps": int(dcfg["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(dcfg["save_steps"]),
        "save_total_limit": 2,
        "gradient_checkpointing": cuda,
        "fp16": cuda,
        "report_to": "none",
        "remove_unused_columns": False,
    }))
    trainer_kwargs = {"model": model, "args": args, "train_dataset": train_ds, "eval_dataset": eval_ds, "peft_config": peft_config}
    trainer_kwargs["processing_class" if "processing_class" in inspect.signature(DPOTrainer).parameters else "tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    # LoRA weights inherit the base dtype; fp16 mixed precision needs float32 trainable weights
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return trainer


def last_checkpoint(output_dir: Path) -> Optional[str]:
    ckpts = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


def train_log_frame(trainer) -> pd.DataFrame:
    return pd.DataFrame(trainer.state.log_history)


def save_adapter(trainer, tokenizer, adapter_dir: Path) -> Path:
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir


def load_adapter_model(mcfg: Dict, adapter_dir: Path):
    """Fresh base model with the saved adapter attached (what M3/M4 load)."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(load_base_model(mcfg), str(adapter_dir))
    model.eval()
    return model


def generate(model, tokenizer, prompts: List[str], max_new_tokens: int, enable_thinking: bool = False) -> List[str]:
    """Greedy replies (deterministic, so lambda=0 vs lambda=1 differences come from the adapter only)."""
    import torch
    device = next(model.parameters()).device
    replies = []
    for p in prompts:
        text = render_prompt([{"role": "user", "content": p}], tokenizer, enable_thinking)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        replies.append(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return replies


def last_token_logits(model, tokenizer, text: str):
    import torch
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float().cpu()


def pairs_to_parquet_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    """Message lists stored as JSON strings so the pairs can be saved as Parquet."""
    return pairs.assign(prompt=pairs["prompt"].map(json.dumps))
