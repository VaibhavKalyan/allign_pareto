"""
Milestone 4 (M4): prompt-adaptive lambda controller (spec §20), on CPU from the M3 results.
  1. classify   M1 classifier on the 400 eval prompts            -> results/m4/eval_prompt_predictions.csv
  2. calibrate  on validation: one lambda per predicted quadrant, most creative policy that is at least as safe
                as fixed lambda = 1 (src/alignment/controller.py); the oracle does the same with true labels
  3. evaluate   fixed lambdas, adaptive (M1 predictions) and oracle (true labels) on validation and test, with
                bootstrap 95% intervals; test is used only here     -> comparison.csv, bootstrap.csv, plots/
Run from project root: `python3 src/run_m4.py` (needs results/m3/per_prompt.csv and models/m1/).
"""
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alignment.controller import SweepTable, bootstrap, calibrate
from src.utils.io import load_pickle, load_yaml, save_json
from src.utils.logging import StructuredLogger
from src.utils.seed import environment_info, set_global_seed


def classify(root: Path, ev: pd.DataFrame, tag: str, out_path: Path) -> pd.DataFrame:
    """Creative/safety probabilities and predicted quadrant from the M1 classifier (its own tuned thresholds)."""
    if out_path.exists():
        return pd.read_csv(out_path)
    from src.classifier.predict import DualHeadPredictor
    cre, saf = load_pickle(root / "models" / tag / "creative_classifier.pkl"), load_pickle(root / "models" / tag / "safety_classifier.pkl")
    pred = DualHeadPredictor(cre["embedder"], cre["head"], saf["head"], cre["threshold"], saf["threshold"]).predict_batch(ev["prompt"].tolist())
    out = ev[["prompt_id", "split", "quadrant", "creative_label", "safety_label"]].copy()
    for k in ["creative_probability", "safety_probability", "creative_pred", "safety_pred"]:
        out[k] = [p[k] for p in pred]
    out["predicted_quadrant"] = out["creative_pred"].astype(str) + "_" + out["safety_pred"].astype(str)
    out.to_csv(out_path, index=False)
    return out


def plot(comparison: pd.DataFrame, boot: pd.DataFrame, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fixed = comparison[comparison["policy"].str.startswith("fixed")].sort_values("mean_lambda")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fixed["safety"], fixed["creativity"], "o-", color="0.6", label="fixed λ (M3 curve)")
    for _, r in fixed.iterrows():
        ax.annotate(f"λ={r['mean_lambda']:g}", (r["safety"], r["creativity"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    for name, marker, color in [("adaptive", "*", "C3"), ("oracle", "D", "C2")]:
        r = comparison[comparison["policy"] == name].iloc[0]
        b = boot[boot["policy"] == name].set_index("metric")
        ax.errorbar(r["safety"], r["creativity"],
                    xerr=[[r["safety"] - b.loc["safety", "ci_low"]], [b.loc["safety", "ci_high"] - r["safety"]]],
                    yerr=[[r["creativity"] - b.loc["creativity", "ci_low"]], [b.loc["creativity", "ci_high"] - r["creativity"]]],
                    fmt=marker, color=color, ms=12 if marker == "*" else 8, capsize=3,
                    label=f"{name} ({'M1 predictions' if name == 'adaptive' else 'true labels'}), 95% CI")
    ax.set(xlabel="safety = 1 − harmful response rate (harmful prompts)", ylabel="semantic diversity (creative, safe prompts: 1_0)", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_m4(root: Path = Path(".")) -> dict:
    ecfg = load_yaml(root / "configs/evaluation.yaml")
    ccfg = ecfg["controller"]
    set_global_seed(int(ecfg.get("seed", 42)))
    lambdas = [float(x) for x in ecfg["sweep"]["lambdas"]]
    ref_lam = float(ccfg["safety_reference_lambda"])
    res = root / "results/m4"
    res.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger("m4_pipeline", res / "execution.jsonl")

    ev = pd.read_csv(root / ecfg["eval_set"]["output_path"])
    preds = classify(root, ev, ccfg["classifier_tag"], res / "eval_prompt_predictions.csv")
    acc = {s: {"creative": round(float((g["creative_pred"] == g["creative_label"]).mean()), 3),
               "safety": round(float((g["safety_pred"] == g["safety_label"]).mean()), 3),
               "quadrant": round(float((g["predicted_quadrant"] == g["quadrant"]).mean()), 3)} for s, g in preds.groupby("split")}
    logger.info("Classified eval prompts", accuracy=acc)

    per_prompt = pd.read_csv(root / "results/m3/per_prompt.csv")
    tables = {s: SweepTable(per_prompt[per_prompt["split"] == s], lambdas) for s in ["validation", "test"]}
    groups = {"adaptive": preds.set_index("prompt_id")["predicted_quadrant"], "oracle": preds.set_index("prompt_id")["quadrant"]}

    # Calibrate on validation only
    val = tables["validation"]
    floor = val.score(val.fixed_index(ref_lam))["safety"]
    policies = {name: calibrate(val, g, floor) for name, g in groups.items()}
    logger.info("Calibrated", safety_floor=round(floor, 4), **{k: v["lambda_by_group"] for k, v in policies.items()})

    rows, boots = [], []
    for split, t in tables.items():
        idx = {f"fixed_{lam:g}": t.fixed_index(lam) for lam in lambdas}
        idx.update({name: t.policy_index(groups[name], p["lambda_by_group"]) for name, p in policies.items()})
        rows += [{"split": split, "policy": name, **t.score(li)} for name, li in idx.items()]
        b = bootstrap(t, idx, reference=f"fixed_{ref_lam:g}", n_boot=int(ccfg["bootstrap_samples"]), seed=int(ecfg.get("seed", 42)))
        boots.append(b.assign(split=split))
    comparison, boot = pd.DataFrame(rows), pd.concat(boots, ignore_index=True)
    comparison.to_csv(res / "comparison.csv", index=False)
    boot.to_csv(res / "bootstrap.csv", index=False)
    for split in tables:
        plot(comparison[comparison["split"] == split], boot[boot["split"] == split], res / f"plots/m4_{split}.png",
             f"Adaptive λ vs fixed λ ({split})")

    ref = f"fixed_{ref_lam:g}"
    test = boot[(boot["split"] == "test") & boot["policy"].isin(["adaptive", "oracle"])]
    summary = {"safety_floor_validation": round(floor, 4), "safety_reference": ref, "classifier_accuracy": acc,
               "policies": {k: v["lambda_by_group"] for k, v in policies.items()},
               "test_vs_reference": {f"{r.policy}_{r.metric}": {"value": round(r.value, 4), f"diff_vs_{ref}": round(getattr(r, f"diff_vs_{ref}"), 4),
                                                                  "diff_95ci": [round(r.diff_ci_low, 4), round(r.diff_ci_high, 4)]}
                                     for r in test.itertuples()}}
    save_json(summary, res / "summary.json")
    save_json({"evaluation_config": ecfg, "environment": environment_info()}, res / "experiment_config.json")
    logger.info("Completed M4")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_m4(Path(__file__).resolve().parents[1]), indent=2))
