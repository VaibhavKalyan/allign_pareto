"""
The lambda dial for M3/M4: weight interpolation between our unaligned SFT model and Qwen's aligned Qwen3-0.6B.
  W(lambda) = W_sft + lambda * (W_aligned - W_sft)      lambda = 0: SFT model, lambda = 1: Qwen3-0.6B
Both models start from Qwen3-0.6B-Base (same architecture and vocabulary), which is what lets the in-between models
stay coherent (model averaging, as in "Mitigating the Alignment Tax of RLHF"). The lambda direction holds all of
Qwen's post-training (chat quality, reasoning, safety), not safety alone; report results that way.
"""
from pathlib import Path
from typing import Dict

from src.alignment.models import has_cuda, load_sft_model


def load_aligned_model(mcfg: Dict, aligned_model_name: str):
    """Qwen's aligned chat model (e.g. Qwen/Qwen3-0.6B), same dtype and device as our models."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(aligned_model_name, dtype=getattr(torch, mcfg.get("dtype", "float32")))
    return model.to("cuda") if has_cuda() else model


class InterpolatedModel:
    """
    One model whose weights are set to W(lambda) in place, so the sweep loads both endpoints once and switches lambda
    in about a second. Both endpoints are kept on the model's device (3 x 2.4 GB for 0.6B in float32 on a T4).
    """

    def __init__(self, mcfg: Dict, sft_adapter_dir: Path, aligned_model_name: str):
        self.model = load_sft_model(mcfg, sft_adapter_dir)
        self.model.eval()
        aligned = load_aligned_model(mcfg, aligned_model_name)
        sft_params, aligned_params = dict(self.model.named_parameters()), dict(aligned.named_parameters())
        if sft_params.keys() != aligned_params.keys():
            raise ValueError(f"{aligned_model_name} does not have the SFT model's parameters: "
                             f"{sorted(sft_params.keys() ^ aligned_params.keys())[:5]}")
        bad = [k for k in sft_params if sft_params[k].shape != aligned_params[k].shape]
        if bad:
            raise ValueError(f"Shape mismatch between the SFT model and {aligned_model_name}: {bad[:5]}")
        self.w_sft = {k: p.detach().clone() for k, p in sft_params.items()}
        self.w_aligned = {k: aligned_params[k].detach().to(p.device, p.dtype).clone() for k, p in sft_params.items()}
        del aligned
        self.lam = 0.0

    def set_lambda(self, lam: float):
        """Set the weights to W(lam). torch.lerp is exact at both ends: lam = 0 and lam = 1 give the endpoint models."""
        import torch
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(torch.lerp(self.w_sft[name], self.w_aligned[name], float(lam)))
        self.lam = float(lam)
        return self.model
