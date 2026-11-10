"""
Batched sampling for the M3 sweep: N samples per prompt, one seed per sample (configs/evaluation.yaml -> sweep).
Prompts are sorted by length so a batch carries little padding. The RNG is reset to a value fixed by (seed, batch),
and batches are identical for every lambda, so differences between lambdas come from the weights, not the dice.
"""
from typing import Dict, List

from src.alignment.models import render_prompt, stop_token_ids


def sample_responses(model, tokenizer, prompts: List[str], seeds: List[int], max_new_tokens: int, temperature: float,
                     top_p: float, enable_thinking: bool = False, batch_size: int = 32) -> List[Dict]:
    """
    One row per (prompt index, sample): response text, n_tokens (generated, without the stop token) and stopped
    (True if the model ended the reply itself, False if it ran into max_new_tokens).
    """
    import torch
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    stops = set(stop_token_ids(tokenizer))
    texts = [render_prompt([{"role": "user", "content": p}], tokenizer, enable_thinking) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(tokenizer(texts[i], add_special_tokens=False)["input_ids"]))
    rows = []
    for sample_idx, seed in enumerate(seeds):
        for b, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            torch.manual_seed(int(seed) * 100_003 + b)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                     top_p=top_p, eos_token_id=sorted(stops), pad_token_id=tokenizer.pad_token_id)
            for row, i in enumerate(idx):
                ids = out[row, enc["input_ids"].shape[1]:].tolist()
                end = next((k for k, t in enumerate(ids) if t in stops), None)
                kept = ids if end is None else ids[:end]
                rows.append({"prompt_idx": i, "sample_idx": sample_idx, "seed": int(seed),
                             "response": tokenizer.decode(kept, skip_special_tokens=True).strip(),
                             "n_tokens": len(kept), "stopped": end is not None})
    return rows
