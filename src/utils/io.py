"""I/O helper utilities for YAML, JSON, Parquet, and Pickle serialization."""
import sys
from pathlib import Path

# Prevent shadowing Python standard library `io` and `logging` when executed directly as `python src/utils/io.py`
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)

import json
import pickle
from typing import Any, Dict, Union
import pandas as pd
import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    return p


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_pickle(obj: Any, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return p


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(root / "configs/classifier.yaml")
    print("[Step 0c: I/O Utility] Successfully loaded configs/classifier.yaml:")
    print("  experiment_name:", cfg.get("experiment_name"), "| seed:", cfg.get("seed"))
