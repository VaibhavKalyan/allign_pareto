"""Reproducibility and deterministic seeding utilities for ALIGN-PARETO."""
import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> dict:
    """Seed Python random, NumPy, environment hashes, and PyTorch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    status = {"seed": seed, "random": True, "numpy": True, "torch": False, "cuda": False}
    try:
        import torch
        torch.manual_seed(seed)
        status["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            status["cuda"] = True
            status["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return status


REPRO_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "sentence-transformers",
    "datasketch", "huggingface_hub", "pyarrow", "matplotlib", "seaborn", "peft", "trl", "bitsandbytes",
]


def environment_info(packages=REPRO_PACKAGES) -> dict:
    """Python, platform, package versions, and GPU info to save next to every experiment (spec §31)."""
    import platform
    import importlib.metadata as md
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    info = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions, "gpu": None}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": props.name, "vram_gb": round(props.total_memory / 1e9, 1), "count": torch.cuda.device_count()}
    except ImportError:
        pass
    return info


if __name__ == "__main__":
    from pathlib import Path
    info = set_global_seed(42)
    print("[Step 0a: Seed Utility] Deterministic global seed locked:", info)
