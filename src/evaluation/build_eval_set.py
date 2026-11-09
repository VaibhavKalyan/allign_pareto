"""
Shared evaluation prompt set for M3/M4 and Group B's benchmark (spec §22, §32).
Draws a fixed number of prompts per creative/safety quadrant from the M1 validation and test splits, so the
classifier never trained on any of them. Run from project root: `python3 src/evaluation/build_eval_set.py`.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import load_yaml, save_json

QUADRANTS = ["0_0", "0_1", "1_0", "1_1"]
OUTPUT_COLUMNS = [
    "prompt_id", "prompt", "split", "quadrant", "creative_label", "safety_label",
    "source_dataset", "source_label", "adversarial",
]


def eligible_prompts(df: pd.DataFrame, max_chars: int, exclude_has_context: bool) -> pd.DataFrame:
    """Drop prompts that cannot be answered on their own (Dolly rows whose passage is not in the prompt) or are too long."""
    keep = df["prompt"].str.len() <= max_chars
    if exclude_has_context:
        keep &= ~df["has_context"].fillna(False).astype(bool)
    return df[keep]


def sample_quadrant(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At most one prompt per family, split evenly across sources; a short source is topped up from the others."""
    df = df.sample(frac=1.0, random_state=seed).drop_duplicates("family_id")
    sources = sorted(df["source_dataset"].unique())
    picked = []
    for i, src in enumerate(sources):
        quota = n // len(sources) + (1 if i < n % len(sources) else 0)
        picked.append(df[df["source_dataset"] == src].head(quota))
    picked = pd.concat(picked)
    shortfall = n - len(picked)
    if shortfall > 0:
        picked = pd.concat([picked, df.drop(picked.index).head(shortfall)])
    if len(picked) < n:
        raise ValueError(f"Only {len(picked)} eligible prompts for a quadrant that needs {n}")
    return picked


def build_eval_set(root_dir: Path = Path(".")) -> pd.DataFrame:
    cfg = load_yaml(root_dir / "configs/evaluation.yaml")
    ecfg = cfg["eval_set"]
    seed = int(cfg.get("seed", 42))
    processed = root_dir / ecfg["m1_processed_dir"]

    parts = []
    for split_file in ecfg["source_splits"]:
        df = pd.read_parquet(processed / f"{split_file}.parquet")
        df = eligible_prompts(df, int(ecfg["max_prompt_chars"]), bool(ecfg["exclude_has_context"]))
        for q in QUADRANTS:
            parts.append(sample_quadrant(df[df["quadrant"] == q], int(ecfg["per_quadrant_per_split"]), seed))

    out = pd.concat(parts).rename(columns={"id": "prompt_id"})[OUTPUT_COLUMNS]
    out = out.sort_values(["split", "quadrant", "prompt_id"]).reset_index(drop=True)
    assert out["prompt_id"].is_unique, "A prompt was picked twice"

    out_path = root_dir / ecfg["output_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    stats = {
        "n_prompts": len(out),
        "seed": seed,
        "config": ecfg,
        "counts": out.groupby(["split", "quadrant", "source_dataset"]).size().rename("n").reset_index().to_dict("records"),
    }
    save_json(stats, out_path.with_name("eval_prompts_stats.json"))
    print(f"Saved {len(out)} prompts to {out_path}")
    return out


if __name__ == "__main__":
    build_eval_set(Path(__file__).resolve().parents[2])
