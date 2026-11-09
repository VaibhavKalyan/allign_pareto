"""
LoRA adapter strength control for ALIGN-PARETO (M2-M4).
Implements W_lambda = W_base + lambda * (alpha / r) * (B @ A) by rescaling the PEFT
`scaling` factor only, so the stored LoRA weights are never modified and lambda can
be changed any number of times without compounding.
"""


def set_peft_lora_lambda(model, lambda_val: float) -> int:
    """Set the adapter strength of every LoRA layer in a PEFT model. Returns the number of layers scaled."""
    scaled_count = 0
    for _, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "scaling"):
            if isinstance(module.scaling, dict):
                for adapter_name in module.scaling:
                    r = module.r[adapter_name] if isinstance(module.r, dict) else module.r
                    alpha = module.lora_alpha[adapter_name] if isinstance(module.lora_alpha, dict) else module.lora_alpha
                    module.scaling[adapter_name] = float(lambda_val) * (float(alpha) / float(r))
                    scaled_count += 1
            else:
                module.scaling = float(lambda_val) * (float(module.lora_alpha) / float(module.r))
                scaled_count += 1
    if scaled_count == 0:
        raise ValueError("No LoRA layers found; is this a PEFT model with an adapter loaded?")
    return scaled_count
