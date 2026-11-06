"""
Inference API for the ALIGN-PARETO Dual-Head Prompt Classifier.
Returns calibrated `{creative_probability, safety_probability}` and binary labels.
"""
from typing import Dict, List, Union
import numpy as np


class DualHeadPredictor:
    """Unified inference wrapper for M1 creative and safety heads."""

    def __init__(
        self,
        embedder,
        creative_head,
        safety_head,
        threshold_creative: float = 0.50,
        threshold_safety: float = 0.45,
    ):
        self.embedder = embedder
        self.creative_head = creative_head
        self.safety_head = safety_head
        self.threshold_creative = threshold_creative
        self.threshold_safety = threshold_safety

    def predict_single(self, prompt: str) -> Dict[str, Union[float, int, str]]:
        res = self.predict_batch([prompt])[0]
        return res

    def predict_batch(self, prompts: List[str]) -> List[Dict[str, Union[float, int, str]]]:
        X = self.embedder.transform(prompts)
        c_probs = self.creative_head.predict_proba(X)[:, 1]
        s_probs = self.safety_head.predict_proba(X)[:, 1]
        outputs = []
        for p, cp, sp in zip(prompts, c_probs, s_probs):
            c_lab = int(cp >= self.threshold_creative)
            s_lab = int(sp >= self.threshold_safety)
            quadrant = f"creative={c_lab}, safety={s_lab}"
            outputs.append({
                "prompt": p,
                "creative_probability": round(float(cp), 4),
                "safety_probability": round(float(sp), 4),
                "creative_pred": c_lab,
                "safety_pred": s_lab,
                "quadrant": quadrant,
            })
        return outputs


if __name__ == "__main__":
    import sys, argparse, json
    from pathlib import Path
    import pandas as pd
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from src.utils.io import load_pickle, load_parquet
    parser = argparse.ArgumentParser(description="Run M1 Dual-Head Classifier Inference")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt to classify")
    args = parser.parse_args()

    cre_bundle = load_pickle(root / "models/m1/creative_classifier.pkl")
    saf_bundle = load_pickle(root / "models/m1/safety_classifier.pkl")
    predictor = DualHeadPredictor(
        cre_bundle["embedder"], cre_bundle["head"], saf_bundle["head"],
        cre_bundle["threshold"], saf_bundle["threshold"]
    )
    if args.prompt:
        print("=== Custom Prompt Classification ===")
        print(json.dumps(predictor.predict_single(args.prompt), indent=2))
    else:
        test_df = load_parquet(root / "data/processed/test.parquet")
        preds = predictor.predict_batch(test_df["prompt"].tolist())
        pred_df = test_df.copy()
        pred_df["creative_probability"] = [p["creative_probability"] for p in preds]
        pred_df["safety_probability"] = [p["safety_probability"] for p in preds]
        pred_df["creative_pred"] = [p["creative_pred"] for p in preds]
        pred_df["safety_pred"] = [p["safety_pred"] for p in preds]
        pred_df["predicted_quadrant"] = [p["quadrant"] for p in preds]
        (root / "results/m1").mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(root / "results/m1/predictions.csv", index=False)
        print("=== [Step 6: Batch & Single-Prompt Inference (src/classifier/predict.py)] ===")
        print(f"Generated predictions for {len(pred_df)} test prompts -> saved to results/m1/predictions.csv")
