"""
Milestone 1 (M1): dataset curation, deduplication, leakage-safe split, and dual-head classifier.
Run from project root: `python3 src/run_m1.py` (add `--quick` for a small local smoke test).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.seed import set_global_seed
from src.utils.logging import StructuredLogger
from src.utils.io import load_yaml, save_json, save_parquet, save_pickle
from src.data.download import load_raw_datasets
from src.data.mapping import (
    DOLLY_CREATIVE, DOLLY_NON_CREATIVE, map_dolly_to_schema, map_wildguard_to_schema, drop_unassigned,
    sample_quadrants, creative_rule_audit_sample, verify_label_independence,
)
from src.data.cleaning import clean_dataset
from src.data.dedup import exact_deduplicate, near_deduplicate_minhash, assign_prompt_families, leakage_safe_family_split
from src.classifier.train import train_dual_head_classifier
from src.classifier.predict import DualHeadPredictor
from src.classifier.evaluate import compute_binary_metrics, generate_all_m1_plots, tune_threshold

LABEL_MAPS = {
    "creative_label": {"0": "non_creative", "1": "creative"},
    "safety_label": {"0": "benign", "1": "harmful"},
    "dolly_creative_categories": sorted(DOLLY_CREATIVE),
    "dolly_non_creative_categories": sorted(DOLLY_NON_CREATIVE),
    "dolly_safety_label": "0 for all rows (no harm annotation; provisional, see DATASET_DECISION.md)",
    "wildguard_safety_label": "prompt_harm_label: harmful -> 1, unharmful -> 0, missing/conflicting -> excluded",
    "wildguard_creative_label": "strict rule: explicit creative-artifact request -> 1; fictional framing only -> excluded; else 0",
}


def _strip_curves(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "_curves"}


def run_m1(root_dir: Path = Path("."), quick: bool = False) -> dict:
    cfg = load_yaml(root_dir / "configs/classifier.yaml")
    dcfg = cfg["data"]
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    tag = "m1_quick" if quick else "m1"
    results_dir = root_dir / "results" / tag
    models_dir = root_dir / "models" / tag
    processed_dir = root_dir / "data/processed" / tag
    logger = StructuredLogger("m1_pipeline", results_dir / "execution.jsonl")
    logger.info("Starting M1 pipeline", seed=seed, quick=quick)

    # 1. Raw public datasets (no synthetic fallback)
    dolly_raw, wg_raw, wg_test_raw, raw_meta = load_raw_datasets(root_dir / "data/raw")

    # 2. Labels + provenance; WildGuardTest is kept aside as an external evaluation set
    combined = pd.concat([map_dolly_to_schema(dolly_raw), map_wildguard_to_schema(wg_raw)], ignore_index=True)
    wg_test = map_wildguard_to_schema(wg_test_raw, id_prefix="wgtest")

    # 3-4. Clean (dropping any WildGuardTest prompt), exact-dedup, drop rows with unknown labels
    cleaned, clean_stats = clean_dataset(combined, min_char_len=int(dcfg["min_char_len"]), benchmark_prompts=wg_test["prompt"])
    deduped, n_exact = exact_deduplicate(cleaned)
    audit = creative_rule_audit_sample(deduped, n_per_rule=int(dcfg["creative_rule_audit_per_rule"]), seed=seed)
    labelled, unassigned_stats = drop_unassigned(deduped)
    pool_stats = verify_label_independence(labelled)

    # 5. Pilot dataset: per-quadrant targets, balanced across sources
    targets = dcfg["target_quadrant_counts"]
    if quick:
        targets = {k: int(cfg["quick_test"]["per_quadrant"]) for k in targets}
    sampled, sampling_report = sample_quadrants(labelled, targets, seed=seed)

    # 6. Near-duplicates, prompt families, leakage-safe split
    k, num_perm = int(dcfg["shingle_k"]), int(dcfg["minhash_num_perm"])
    near_deduped, n_near = near_deduplicate_minhash(sampled, threshold=float(dcfg["near_dup_jaccard_threshold"]), k=k, num_perm=num_perm, seed=seed)
    with_families, family_stats = assign_prompt_families(near_deduped, threshold=float(dcfg["family_jaccard_threshold"]), k=k, num_perm=num_perm, seed=seed)
    indep_stats = verify_label_independence(with_families)
    final_df, split_stats = leakage_safe_family_split(
        with_families, train_ratio=float(dcfg["train_ratio"]), val_ratio=float(dcfg["val_ratio"]), seed=seed
    )

    save_parquet(final_df, processed_dir / "classifier_dataset.parquet")
    train_df = final_df[final_df["split"] == "train"].reset_index(drop=True)
    val_df = final_df[final_df["split"] == "validation"].reset_index(drop=True)
    test_df = final_df[final_df["split"] == "test"].reset_index(drop=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        save_parquet(part, processed_dir / f"{name}.parquet")
    audit.to_csv(results_dir / "creative_rule_audit.csv", index=False)

    dataset_stats = {
        "raw_metadata": raw_meta,
        "cleaning_stats": clean_stats,
        "exact_duplicates_removed": n_exact,
        "unassigned_label_stats": unassigned_stats,
        "labelled_pool_before_sampling": pool_stats,
        "sampling_report": sampling_report,
        "near_duplicates_removed": n_near,
        "family_stats": family_stats,
        "independence_check": indep_stats,
        "split_stats": split_stats,
        "source_by_quadrant": {
            f"{q}/{s}": int(n) for (q, s), n in final_df.groupby(["quadrant", "source_dataset"]).size().items()
        },
    }
    save_json(dataset_stats, results_dir / "dataset_statistics.json")
    save_json(LABEL_MAPS, models_dir / "label_maps.json")

    # 7. Train heads, tune thresholds on validation
    embedder, creative_head, safety_head, train_meta, emb = train_dual_head_classifier(train_df, val_df, cfg)
    icfg = cfg["inference"]
    grid = np.arange(icfg["threshold_grid_min"], icfg["threshold_grid_max"] + 1e-9, icfg["threshold_grid_step"])
    thr_c, val_f1_c = tune_threshold(val_df["creative_label"], creative_head.predict_proba(emb["val"])[:, 1], grid)
    thr_s, val_f1_s = tune_threshold(val_df["safety_label"], safety_head.predict_proba(emb["val"])[:, 1], grid)
    train_meta["thresholds"] = {"creative": thr_c, "safety": thr_s, "val_f1_creative": val_f1_c, "val_f1_safety": val_f1_s}

    save_pickle({"embedder": embedder, "head": creative_head, "threshold": thr_c, "meta": train_meta}, models_dir / "creative_classifier.pkl")
    save_pickle({"embedder": embedder, "head": safety_head, "threshold": thr_s, "meta": train_meta}, models_dir / "safety_classifier.pkl")

    # 8. Test-split evaluation
    predictor = DualHeadPredictor(embedder, creative_head, safety_head, thr_c, thr_s)
    test_preds = predictor.predict_batch(test_df["prompt"].tolist())
    pred_df = test_df.copy()
    for key in ["creative_probability", "safety_probability", "creative_pred", "safety_pred"]:
        pred_df[key] = [p[key] for p in test_preds]
    pred_df.to_csv(results_dir / "predictions.csv", index=False)

    cre_metrics = compute_binary_metrics(test_df["creative_label"].to_numpy(), pred_df["creative_probability"].to_numpy(), threshold=thr_c)
    saf_metrics = compute_binary_metrics(test_df["safety_label"].to_numpy(), pred_df["safety_probability"].to_numpy(), threshold=thr_s)
    generate_all_m1_plots(final_df, cre_metrics, saf_metrics, results_dir / "plots")

    # 9. External safety evaluation on WildGuardTest (never used for training or thresholds)
    wg_eval = wg_test.dropna(subset=["safety_label"]).reset_index(drop=True)
    if quick:
        wg_eval = wg_eval.sample(min(200, len(wg_eval)), random_state=seed).reset_index(drop=True)
    wg_probs = np.array([p["safety_probability"] for p in predictor.predict_batch(wg_eval["prompt"].tolist())])
    wg_metrics = compute_binary_metrics(wg_eval["safety_label"].astype(int).to_numpy(), wg_probs, threshold=thr_s)

    # 10. Error analysis (spec §17)
    errors = {
        "safety_false_negatives": (pred_df["safety_label"] == 1) & (pred_df["safety_pred"] == 0),
        "safety_false_positives": (pred_df["safety_label"] == 0) & (pred_df["safety_pred"] == 1),
        "creative_false_negatives": (pred_df["creative_label"] == 1) & (pred_df["creative_pred"] == 0),
        "creative_false_positives": (pred_df["creative_label"] == 0) & (pred_df["creative_pred"] == 1),
    }
    cols = ["id", "source_dataset", "label_rule", "source_subcategory", "creative_label", "safety_label",
            "creative_probability", "safety_probability", "prompt"]
    for name, mask in errors.items():
        pred_df.loc[mask, cols].to_csv(results_dir / f"errors_{name}.csv", index=False)

    per_quadrant = {}
    for q, g in pred_df.groupby("quadrant"):
        per_quadrant[q] = {
            "n": int(len(g)),
            "creative_accuracy": round(float((g["creative_pred"] == g["creative_label"]).mean()), 4),
            "safety_accuracy": round(float((g["safety_pred"] == g["safety_label"]).mean()), 4),
        }

    metrics_out = {
        "quick_test": quick,
        "creative_head": _strip_curves(cre_metrics),
        "safety_head": _strip_curves(saf_metrics),
        "safety_head_wildguardtest": {**_strip_curves(wg_metrics), "n": int(len(wg_eval))},
        "per_quadrant_test_accuracy": per_quadrant,
        "error_counts": {k: int(v.sum()) for k, v in errors.items()},
        "aggregate": {
            "macro_f1": round(0.5 * (cre_metrics["f1"] + saf_metrics["f1"]), 4),
            "exact_match_ratio": round(float(
                ((pred_df["creative_label"] == pred_df["creative_pred"]) & (pred_df["safety_label"] == pred_df["safety_pred"])).mean()
            ), 4),
        },
    }
    save_json(metrics_out, results_dir / "metrics.json")
    save_json({"m1_config": cfg, "training_meta": train_meta}, results_dir / "experiment_config.json")
    logger.info("Completed M1 pipeline", macro_f1=metrics_out["aggregate"]["macro_f1"])
    return metrics_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ALIGN-PARETO M1")
    parser.add_argument("--quick", action="store_true", help="small local smoke test; results go to results/m1_quick")
    args = parser.parse_args()
    res = run_m1(Path(__file__).resolve().parents[1], quick=args.quick)
    print("M1 Complete:", json.dumps(res["aggregate"], indent=2))
