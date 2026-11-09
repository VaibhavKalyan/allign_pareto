"""Moved out of src/alignment/models.py: loads the SFT model with the DPO adapter attached."""
from pathlib import Path
from typing import Dict

from src.alignment.models import load_sft_model

def load_aligned_model(mcfg: Dict, sft_adapter_dir: Path, dpo_adapter_dir: Path):
    """SFT model with the DPO adapter attached; set its strength with adapters.set_peft_lora_lambda."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(load_sft_model(mcfg, sft_adapter_dir), str(dpo_adapter_dir))
    model.eval()
    return model
