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
from src.utils.seed import environment_info, set_global_seed
from src.utils.logging import StructuredLogger
from src.utils.io import load_yaml, save_json, save_parquet, save_pickle
from src.data.download import load_raw_datasets
from src.data.mapping import (
    DOLLY_CREATIVE, DOLLY_NON_CREATIVE, map_dolly_to_schema, map_wildguard_to_schema, drop_unassigned,
    sample_quadrants, creative_rule_audit_sample, verify_label_independence,
)
from src.data.cleaning import clean_dataset
from src.data.dedup import (
    exact_deduplicate, near_deduplicate_minhash, assign_prompt_families, leakage_safe_family_split, drop_near_duplicates_of,
)
from src.classifier.train import train_dual_head_classifier
from src.classifier.predict import DualHeadPredictor
from src.classifier.evaluate import (
    compute_binary_metrics, generate_all_m1_plots, tune_threshold, tune_threshold_for_recall, subgroup_metrics,
)

LABEL_MAPS = {
    "creative_label": {"0": "non_creative", "1": "creative"},
    "safety_label": {"0": "benign", "1": "harmful"},
    "dolly_creative_categories": sorted(DOLLY_CREATIVE),
    "dolly_non_creative_categories": sorted(DOLLY_NON_CREATIVE),
    "dolly_creative_writing_questions": "creative_writing rows that are plain questions with no creative cue -> excluded (noisy source label)",
    "dolly_safety_label": "0 for all rows (no harm annotation; provisional, see DATASET_DECISION.md)",
    "wildguard_safety_label": "prompt_harm_label: harmful -> 1, unharmful -> 0, missing/conflicting -> excluded",
    "wildguard_creative_label": (
        "strict rule: explicit creative-artifact or brainstorming request -> 1; fictional framing only, or "
        "'narrative' without fiction context -> excluded; else 0"
    ),
}


def _strip_curves(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "_curves"}


def _prf(m: dict) -> dict:
    return {k: m[k] for k in ["threshold", "precision", "recall", "f1"]} | {"false_negatives": m["confusion_matrix"]["fn"]}


FICTION_RULE = "uncertain_fictional_framing_only"


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
    # Prompts with fictional framing only have no creative label, but they are exactly the creative +
    # sensitive prompts the controller must handle: keep them aside as an extra evaluation slice.
    fiction_pool = deduped[(deduped["label_rule"] == FICTION_RULE) & deduped["safety_label"].notna()]
    labelled, unassigned_stats = drop_unassigned(deduped)

    # Near-variants of WildGuardTest prompts must not reach training (exact matches were removed above)
    k, num_perm = int(dcfg["shingle_k"]), int(dcfg["minhash_num_perm"])
    heldout_thr = float(dcfg.get("heldout_near_dup_jaccard_threshold", dcfg["family_jaccard_threshold"]))
    labelled, n_heldout_near = drop_near_duplicates_of(labelled, wg_test["prompt"].tolist(), heldout_thr, k, num_perm, seed)
    fiction_pool, n_fiction_heldout = drop_near_duplicates_of(fiction_pool, wg_test["prompt"].tolist(), heldout_thr, k, num_perm, seed)
    pool_stats = verify_label_independence(labelled)

    # 5. Pilot dataset: per-quadrant targets, balanced across sources
    targets = dcfg["target_quadrant_counts"]
    if quick:
        targets = {k: int(cfg["quick_test"]["per_quadrant"]) for k in targets}
    sampled, sampling_report = sample_quadrants(labelled, targets, seed=seed)

    # 6. Near-duplicates, prompt families, leakage-safe split
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
        "wildguardtest_near_duplicates_removed": {"threshold": heldout_thr, "labelled_pool": n_heldout_near, "fictional_framing_pool": n_fiction_heldout},
        "labelled_pool_before_sampling": pool_stats,
        "sampling_report": sampling_report,
        "near_duplicates_removed": n_near,
        "family_stats": family_stats,
        "independence_check": indep_stats,
        "split_stats": split_stats,
        "source_by_quadrant": {
            f"{q}/{s}": int(n) for (q, s), n in final_df.groupby(["quadrant", "source_dataset"]).size().items()
        },
        "source_by_split": {
            f"{sp}/{s}": int(n) for (sp, s), n in final_df.groupby(["split", "source_dataset"]).size().items()
        },
    }
    save_json(dataset_stats, results_dir / "dataset_statistics.json")
    save_json(LABEL_MAPS, models_dir / "label_maps.json")

    # 7. Train heads, tune thresholds on validation
    embedder, creative_head, safety_head, train_meta, emb = train_dual_head_classifier(train_df, val_df, cfg)
    icfg = cfg["inference"]
    grid = np.round(np.arange(icfg["threshold_grid_min"], icfg["threshold_grid_max"] + 1e-9, icfg["threshold_grid_step"]), 4)
    val_p_s = safety_head.predict_proba(emb["val"])[:, 1]
    thr_c, val_f1_c = tune_threshold(val_df["creative_label"], creative_head.predict_proba(emb["val"])[:, 1], grid)
    thr_s_f1, val_f1_s = tune_threshold(val_df["safety_label"], val_p_s, grid)
    s_objective = icfg.get("safety_threshold_objective", "f1")
    if s_objective == "target_recall":
        target = float(icfg["safety_target_recall"])
        thr_rec, _, _ = tune_threshold_for_recall(val_df["safety_label"], val_p_s, grid, target)
        # Never less recall than the max-F1 threshold would give
        thr_s = min(thr_rec, thr_s_f1)
        pred_v = val_p_s >= thr_s
        y_v = val_df["safety_label"].to_numpy(dtype=int)
        val_rec_s = round(float((pred_v & (y_v == 1)).sum() / max(1, (y_v == 1).sum())), 4)
        val_prec_s = round(float((pred_v & (y_v == 1)).sum() / max(1, pred_v.sum())), 4)
    elif s_objective == "f1":
        thr_s, target, val_rec_s, val_prec_s = thr_s_f1, None, None, None
    else:
        raise ValueError(f"Unknown safety_threshold_objective {s_objective!r} (use 'target_recall' or 'f1')")
    train_meta["thresholds"] = {
        "creative": thr_c, "val_f1_creative": val_f1_c,
        "safety": thr_s, "safety_objective": s_objective, "safety_target_recall": target,
        "val_recall_safety": val_rec_s, "val_precision_safety": val_prec_s,
        "safety_max_f1_threshold": thr_s_f1, "val_f1_safety_at_max_f1_threshold": val_f1_s,
    }

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
    wg_y = wg_eval["safety_label"].astype(int).to_numpy()
    wg_metrics = compute_binary_metrics(wg_y, wg_probs, threshold=thr_s)
    wg_eval = wg_eval.assign(safety_probability=wg_probs, safety_pred=(wg_probs >= thr_s).astype(int))

    # Same numbers at the max-F1 safety threshold, so the cost of the recall-first choice is visible
    test_s_prob = pred_df["safety_probability"].to_numpy()
    threshold_comparison = {
        name: {
            "test": _prf(compute_binary_metrics(test_df["safety_label"].to_numpy(), test_s_prob, threshold=t)),
            "wildguardtest": _prf(compute_binary_metrics(wg_y, wg_probs, threshold=t)),
        }
        for name, t in [("chosen", thr_s), ("max_f1", thr_s_f1)]
    }

    # 9b. Fictional-framing slice: never trained on, safety labels known, creative labels left for humans.
    # Prompts near-identical to a training prompt are dropped so this stays a held-out slice.
    fiction_eval, n_fiction_train_near = drop_near_duplicates_of(
        fiction_pool, train_df["prompt"].tolist(), heldout_thr, k, num_perm, seed
    )
    if quick:
        fiction_eval = fiction_eval.sample(min(200, len(fiction_eval)), random_state=seed).reset_index(drop=True)
    fiction_metrics, fiction_summary = None, {"n": 0}
    if len(fiction_eval):
        f_preds = predictor.predict_batch(fiction_eval["prompt"].tolist())
        fiction_eval = fiction_eval.assign(
            **{key: [p[key] for p in f_preds] for key in ["creative_probability", "safety_probability", "creative_pred", "safety_pred"]}
        )
        f_y = fiction_eval["safety_label"].astype(int).to_numpy()
        if 0 < f_y.sum() < len(f_y):
            fiction_metrics = _strip_curves(compute_binary_metrics(f_y, fiction_eval["safety_probability"].to_numpy(), threshold=thr_s))
        fiction_summary = {
            "n": int(len(fiction_eval)),
            "n_harmful": int(f_y.sum()),
            "removed_near_duplicates_of_train": n_fiction_train_near,
            "share_predicted_creative": round(float(fiction_eval["creative_pred"].mean()), 4),
            "mean_creative_probability": round(float(fiction_eval["creative_probability"].mean()), 4),
        }
        fiction_out = fiction_eval[["id", "adversarial", "source_subcategory", "safety_label", "creative_probability",
                                    "safety_probability", "creative_pred", "safety_pred", "prompt"]].copy()
        fiction_out["human_creative_label"] = ""
        fiction_out.to_csv(results_dir / "fictional_framing_predictions.csv", index=False)

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
        "safety_head_fictional_framing_slice": {"summary": fiction_summary, "metrics": fiction_metrics},
        "safety_threshold_comparison": threshold_comparison,
        # Per-source and per-adversarial breakdowns: Dolly rows are all safety=0 and easy, so the blended
        # test numbers overstate how well the safety head separates harmful from benign WildGuardMix prompts
        "by_source_test": {
            "creative_head": subgroup_metrics(pred_df, "creative_label", "creative_probability", "creative_pred", "source_dataset"),
            "safety_head": subgroup_metrics(pred_df, "safety_label", "safety_probability", "safety_pred", "source_dataset"),
        },
        "by_adversarial": {
            "test_wildguardmix_rows": subgroup_metrics(
                pred_df[pred_df["source_dataset"] == "wildguardmix"], "safety_label", "safety_probability", "safety_pred", "adversarial"
            ),
            "wildguardtest": subgroup_metrics(wg_eval, "safety_label", "safety_probability", "safety_pred", "adversarial"),
        },
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
    save_json(
        {"m1_config": cfg, "training_meta": train_meta, "dataset_versions": raw_meta, "environment": environment_info()},
        results_dir / "experiment_config.json",
    )
    logger.info("Completed M1 pipeline", macro_f1=metrics_out["aggregate"]["macro_f1"])
    return metrics_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ALIGN-PARETO M1")
    parser.add_argument("--quick", action="store_true", help="small local smoke test; results go to results/m1_quick")
    args = parser.parse_args()
    res = run_m1(Path(__file__).resolve().parents[1], quick=args.quick)
    print("M1 Complete:", json.dumps(res["aggregate"], indent=2))
