"""
Model loading and generation shared by M2 (training) and M3/M4 (lambda sweep).
  Qwen3-0.6B-Base (pretrained only) + SFT LoRA adapter, merged -> helpful-only chat model (lambda = 0)
The aligned endpoint (lambda = 1) is Qwen's Qwen3-0.6B; src/alignment/interpolate.py blends the two.
Weights stay in float32 (a 0.6B model fits a T4); matrix multiplies run in fp16 under autocast during training.
"""
import inspect
from pathlib import Path
from typing import Dict, List, Optional

END_OF_TEXT = "<|endoftext|>"
END_OF_REPLY = "<|im_end|>"


def has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def load_tokenizer(mcfg: Dict):
    """
    Tokenizer of the chat model (same vocabulary as the base model, plus the chat template). Its EOS, which SFT and
    DPO append to every training reply, is set to `end_of_reply_token`: the base model's <|im_end|> row is untrained
    (norm ~0.35 vs ~1.0) and LoRA cannot train it, so a model taught to end with <|im_end|> never stops.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(mcfg["tokenizer_name"])
    tokenizer.eos_token = mcfg.get("end_of_reply_token", tokenizer.eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = END_OF_TEXT
    return tokenizer


def stop_token_ids(tokenizer) -> List[int]:
    """Generation stops at <|im_end|> (Qwen's chat models) or <|endoftext|> (our SFT model), whichever comes first."""
    return sorted({tokenizer.convert_tokens_to_ids(t) for t in (END_OF_REPLY, END_OF_TEXT)} | {tokenizer.eos_token_id})


def load_base_model(mcfg: Dict):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(mcfg["base_model_name"], dtype=getattr(torch, mcfg.get("dtype", "float32")))
    return model.to("cuda") if has_cuda() else model


def load_sft_model(mcfg: Dict, sft_adapter_dir: Path):
    """Base model with the SFT adapter merged into its weights: the helpful-only chat model."""
    from peft import PeftModel
    return PeftModel.from_pretrained(load_base_model(mcfg), str(sft_adapter_dir)).merge_and_unload()



def lora_config(lcfg: Dict):
    from peft import LoraConfig
    return LoraConfig(
        r=int(lcfg["r"]), lora_alpha=int(lcfg["lora_alpha"]), lora_dropout=float(lcfg["lora_dropout"]),
        target_modules=list(lcfg["target_modules"]), task_type=lcfg["task_type"], bias="none",
    )


def render_prompt(messages: List[Dict[str, str]], tokenizer, enable_thinking: bool = False) -> str:
    """Chat-template text up to the assistant's reply; identical in SFT, DPO and M3 generation."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def supported_kwargs(cls, kwargs: Dict) -> Dict:
    """Keep only the arguments this TRL version accepts (names move between releases)."""
    params = inspect.signature(cls).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def trainer_precision_kwargs() -> Dict:
    cuda = has_cuda()
    return {
        "fp16": cuda,
        "gradient_checkpointing": cuda,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }


def cast_trainable_to_fp32(model) -> None:
    """fp16 mixed precision can only unscale float32 gradients."""
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()


def last_checkpoint(output_dir: Path) -> Optional[str]:
    ckpts = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


def save_adapter(trainer, tokenizer, adapter_dir: Path) -> Path:
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir


def generate(model, tokenizer, prompts: List[str], max_new_tokens: int, enable_thinking: bool = False,
             temperature: float = 0.0, top_p: float = 1.0, seed: int = 0) -> List[str]:
    """
    One reply per prompt: greedy when temperature is 0, else sampled with the RNG reset to `seed` before every prompt,
    so two models given the same seed draw the same random numbers and differences come from the weights.
    """
    import torch
    device = next(model.parameters()).device
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {"do_sample": False}
    replies = []
    for p in prompts:
        text = render_prompt([{"role": "user", "content": p}], tokenizer, enable_thinking)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling,
                                 eos_token_id=stop_token_ids(tokenizer), pad_token_id=tokenizer.pad_token_id)
        replies.append(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return replies


def last_token_logits(model, tokenizer, text: str):
    import torch
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float().cpu()
