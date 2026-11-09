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
from src.data.download import load_raw_datasets, load_xstest
from src.data.mapping import (
    DOLLY_CREATIVE, DOLLY_NON_CREATIVE, map_dolly_to_schema, map_wildguard_to_schema, drop_unassigned,
    sample_quadrants, creative_rule_audit_sample, verify_label_independence,
)
from src.data.cleaning import clean_dataset, normalize_prompt_text
from src.data.dedup import (
    exact_deduplicate, near_deduplicate_minhash, assign_prompt_families, leakage_safe_family_split, drop_near_duplicates_of,
)
from src.classifier.train import select_embedder_and_C, train_dual_head_classifier
from src.classifier.predict import DualHeadPredictor
from src.classifier.evaluate import (
    bootstrap_ci, bootstrap_ci_rate, compute_binary_metrics, generate_all_m1_plots, tune_threshold, tune_threshold_for_recall, subgroup_metrics,
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


def _build_split(labelled: pd.DataFrame, dcfg: dict, targets: dict, split_seed: int, minhash_seed: int,
                 safety_only: pd.DataFrame = None):
    """
    Steps 5-6: per-quadrant sampling, near-dup removal, prompt families, family-safe split.
    `safety_only` rows (known safety label, no creative label) join after sampling and go through the same
    near-dup / family / split steps, so they never leak across splits; only the safety head trains on them.
    """
    k, num_perm = int(dcfg["shingle_k"]), int(dcfg["minhash_num_perm"])
    sampled, sampling_report = sample_quadrants(labelled, targets, seed=split_seed)
    if safety_only is not None and len(safety_only):
        sampled = pd.concat([sampled, safety_only], ignore_index=True)
    near_deduped, n_near = near_deduplicate_minhash(
        sampled, threshold=float(dcfg["near_dup_jaccard_threshold"]), k=k, num_perm=num_perm, seed=minhash_seed
    )
    with_families, family_stats = assign_prompt_families(
        near_deduped, threshold=float(dcfg["family_jaccard_threshold"]), k=k, num_perm=num_perm, seed=minhash_seed
    )
    indep_stats = verify_label_independence(with_families[with_families["creative_label"].notna()])
    final_df, split_stats = leakage_safe_family_split(
        with_families, train_ratio=float(dcfg["train_ratio"]), val_ratio=float(dcfg["val_ratio"]), seed=split_seed
    )
    info = {"sampling_report": sampling_report,
            "safety_only_rows_added": int(len(safety_only)) if safety_only is not None else 0,
            "near_duplicates_removed": n_near, "family_stats": family_stats,
            "independence_check": indep_stats, "split_stats": split_stats}
    return final_df, info


def _choose_thresholds(val_df: pd.DataFrame, creative_head, safety_head, X_val: np.ndarray, icfg: dict):
    """Validation-only thresholds: creative = max F1; safety = recall target (never below max-F1 recall) or max F1."""
    grid = np.round(np.arange(icfg["threshold_grid_min"], icfg["threshold_grid_max"] + 1e-9, icfg["threshold_grid_step"]), 4)
    # Creative threshold on rows with a creative label; safety threshold on every validation row
    m_c = val_df["creative_label"].notna().to_numpy()
    val_p_s = safety_head.predict_proba(X_val)[:, 1]
    thr_c, val_f1_c = tune_threshold(val_df["creative_label"][m_c].astype(int), creative_head.predict_proba(X_val[m_c])[:, 1], grid)
    thr_s_f1, val_f1_s = tune_threshold(val_df["safety_label"].astype(int), val_p_s, grid)
    s_objective = icfg.get("safety_threshold_objective", "f1")
    if s_objective == "target_recall":
        target = float(icfg["safety_target_recall"])
        thr_rec, _, _ = tune_threshold_for_recall(val_df["safety_label"].astype(int), val_p_s, grid, target)
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
    meta = {
        "creative": thr_c, "val_f1_creative": val_f1_c,
        "safety": thr_s, "safety_objective": s_objective, "safety_target_recall": target,
        "val_recall_safety": val_rec_s, "val_precision_safety": val_prec_s,
        "safety_max_f1_threshold": thr_s_f1, "val_f1_safety_at_max_f1_threshold": val_f1_s,
    }
    return thr_c, thr_s, thr_s_f1, meta


def _key_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    m = compute_binary_metrics(y, p, threshold=threshold)
    out = {k: m[k] for k in ["precision", "recall", "f1", "roc_auc"]}
    cm = m["confusion_matrix"]
    out["false_positive_rate"] = round(cm["fp"] / max(1, cm["fp"] + cm["tn"]), 4)
    return out


def _spread(rows: list) -> dict:
    """Mean, std, min, max of each metric across seed runs."""
    out = {}
    for group in rows[0]["metrics"]:
        for metric in rows[0]["metrics"][group]:
            v = np.array([r["metrics"][group][metric] for r in rows], dtype=float)
            out[f"{group}/{metric}"] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
                                        "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
    return out


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
    xstest, raw_meta["xstest"] = load_xstest(root_dir / "data/raw")

    # 2. Labels + provenance; WildGuardTest is kept aside as an external evaluation set
    combined = pd.concat([map_dolly_to_schema(dolly_raw), map_wildguard_to_schema(wg_raw)], ignore_index=True)
    wg_test = map_wildguard_to_schema(wg_test_raw, id_prefix="wgtest")
    # Held-out evaluation prompts (WildGuardTest + XSTest) must never reach training or threshold tuning
    heldout_prompts = wg_test["prompt"].tolist() + xstest["prompt"].astype(str).tolist()

    # 3-4. Clean (dropping any held-out prompt), exact-dedup, drop rows with unknown labels
    cleaned, clean_stats = clean_dataset(combined, min_char_len=int(dcfg["min_char_len"]), benchmark_prompts=heldout_prompts)
    deduped, n_exact = exact_deduplicate(cleaned)
    audit = creative_rule_audit_sample(deduped, n_per_rule=int(dcfg["creative_rule_audit_per_rule"]), seed=seed)
    # Prompts with fictional framing only have no creative label, but they are exactly the creative +
    # sensitive prompts the controller must handle: keep them aside as an extra evaluation slice.
    fiction_pool = deduped[(deduped["label_rule"] == FICTION_RULE) & deduped["safety_label"].notna()]
    labelled, unassigned_stats = drop_unassigned(deduped)

    # Near-variants of held-out prompts must not reach training (exact matches were removed above)
    k, num_perm = int(dcfg["shingle_k"]), int(dcfg["minhash_num_perm"])
    heldout_thr = float(dcfg.get("heldout_near_dup_jaccard_threshold", dcfg["family_jaccard_threshold"]))
    labelled, n_heldout_near = drop_near_duplicates_of(labelled, heldout_prompts, heldout_thr, k, num_perm, seed)
    fiction_pool, n_fiction_heldout = drop_near_duplicates_of(fiction_pool, heldout_prompts, heldout_thr, k, num_perm, seed)
    pool_stats = verify_label_independence(labelled)

    # 5. Pilot dataset: per-quadrant targets, balanced across sources
    targets = dcfg["target_quadrant_counts"]
    if quick:
        targets = {k: int(cfg["quick_test"]["per_quadrant"]) for k in targets}
    # 6. Near-duplicates, prompt families, leakage-safe split
    # Role-play prompts have a known safety label: optionally let the safety head learn from them
    use_fiction = bool(dcfg.get("fictional_framing_for_safety_head", False))
    fiction_for_split = fiction_pool.assign(creative_label=np.nan) if use_fiction else None
    if use_fiction and quick:
        fiction_for_split = fiction_for_split.sample(min(300, len(fiction_for_split)), random_state=seed)
    final_df, split_info = _build_split(labelled, dcfg, targets, split_seed=seed, minhash_seed=seed, safety_only=fiction_for_split)

    save_parquet(final_df, processed_dir / "classifier_dataset.parquet")
    train_df = final_df[final_df["split"] == "train"].reset_index(drop=True)
    val_df = final_df[final_df["split"] == "validation"].reset_index(drop=True)
    test_all = final_df[final_df["split"] == "test"].reset_index(drop=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_all)]:
        save_parquet(part, processed_dir / f"{name}.parquet")
    # Main test metrics use rows with both labels, as before; held-out role-play rows form the fiction slice
    test_df = test_all[test_all["creative_label"].notna()].reset_index(drop=True)
    test_df = test_df.assign(creative_label=test_df["creative_label"].astype(int), safety_label=test_df["safety_label"].astype(int))
    audit.to_csv(results_dir / "creative_rule_audit.csv", index=False)

    dataset_stats = {
        "raw_metadata": raw_meta,
        "cleaning_stats": clean_stats,
        "exact_duplicates_removed": n_exact,
        "unassigned_label_stats": unassigned_stats,
        "heldout_near_duplicates_removed": {
            "heldout_sets": ["wildguardtest", "xstest"], "threshold": heldout_thr,
            "labelled_pool": n_heldout_near, "fictional_framing_pool": n_fiction_heldout,
        },
        "labelled_pool_before_sampling": pool_stats,
        **split_info,
        "source_by_quadrant": {
            f"{q}/{s}": int(n) for (q, s), n in final_df.groupby(["quadrant", "source_dataset"]).size().items()
        },
        "source_by_split": {
            f"{sp}/{s}": int(n) for (sp, s), n in final_df.groupby(["split", "source_dataset"]).size().items()
        },
    }
    save_json(dataset_stats, results_dir / "dataset_statistics.json")
    save_json(LABEL_MAPS, models_dir / "label_maps.json")

    # 7. Choose encoder + per-head C on validation, train heads, tune thresholds on validation
    max_cand = int(cfg["quick_test"].get("max_embedding_candidates", 0)) if quick else None
    embedder, best_C, selection_table = select_embedder_and_C(train_df, val_df, cfg, max_candidates=max_cand)
    save_json(selection_table, results_dir / "model_selection.json")
    logger.info("Selected encoder", embedding_model=embedder.model_name, C=best_C)
    chosen_row = next(r for r in selection_table if r["selected"])
    edge = [h for h, at_edge in chosen_row["best_C_at_grid_edge"].items() if at_edge]
    if edge:
        logger.warning("Best C is at the edge of C_grid; widen the grid", heads=edge, C=best_C)
    embedder, creative_head, safety_head, train_meta, emb = train_dual_head_classifier(train_df, val_df, cfg, embedder, best_C)
    icfg = cfg["inference"]
    thr_c, thr_s, thr_s_f1, train_meta["thresholds"] = _choose_thresholds(val_df, creative_head, safety_head, emb["val"], icfg)

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

    # 9a. XSTest (external): safe prompts that only sound unsafe show over-flagging; never used for tuning
    xs_eval = xstest.copy()
    xs_probs = np.array([p["safety_probability"] for p in predictor.predict_batch(xs_eval["prompt"].astype(str).tolist())])
    xs_y = xs_eval["safety_label"].to_numpy(dtype=int)
    xs_metrics = compute_binary_metrics(xs_y, xs_probs, threshold=thr_s)
    xs_eval = xs_eval.assign(safety_probability=xs_probs, safety_pred=(xs_probs >= thr_s).astype(int))
    xs_eval.to_csv(results_dir / "xstest_predictions.csv", index=False)
    xs_cm = xs_metrics["confusion_matrix"]
    xstest_out = {
        **_strip_curves(xs_metrics), "n": int(len(xs_eval)),
        "safe_prompts_flagged_rate": round(xs_cm["fp"] / max(1, xs_cm["fp"] + xs_cm["tn"]), 4),
        "unsafe_prompts_missed_rate": round(xs_cm["fn"] / max(1, xs_cm["fn"] + xs_cm["tp"]), 4),
        "by_type": subgroup_metrics(xs_eval, "safety_label", "safety_probability", "safety_pred", "type"),
    }
    threshold_comparison["chosen"]["xstest"] = _prf(compute_binary_metrics(xs_y, xs_probs, threshold=thr_s))
    threshold_comparison["max_f1"]["xstest"] = _prf(compute_binary_metrics(xs_y, xs_probs, threshold=thr_s_f1))

    # 9b. Fictional-framing slice: never trained on, safety labels known, creative labels left for humans.
    # Prompts near-identical to a training prompt are dropped so this stays a held-out slice.
    if use_fiction:
        # Held-out = the role-play rows the family-safe split put in test (the rest trained the safety head)
        fiction_eval = test_all[test_all["creative_label"].isna()].reset_index(drop=True)
        n_fiction_train_near = 0
    else:
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
        fiction_summary["used_for_safety_head_training"] = use_fiction
    # Audit file for the human creative labels: every role-play prompt (the creative head never trains on them).
    # `split` says which ones the safety head trained on; only split == "test" rows are held out for safety.
    if len(fiction_pool):
        audit_rows = fiction_pool if not quick else fiction_pool.sample(min(200, len(fiction_pool)), random_state=seed)
        a_preds = predictor.predict_batch(audit_rows["prompt"].tolist())
        split_of = final_df.set_index("prompt_sha256")["split"] if use_fiction else None
        fiction_out = audit_rows[["id", "adversarial", "source_subcategory", "safety_label", "prompt"]].assign(
            **{key: [p[key] for p in a_preds] for key in ["creative_probability", "safety_probability", "creative_pred", "safety_pred"]},
            split=audit_rows["prompt_sha256"].map(split_of).fillna("not_sampled").values if use_fiction else "not_used",
            human_creative_label="",
        )
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

    # 10b. False alarms on harmless prompts: how often the safety head flags them, by creativity / source / framing
    harmless = pred_df[pred_df["safety_label"] == 0].assign(
        creative=lambda d: np.where(d["creative_label"] == 1, "creative", "non_creative"),
        group=lambda d: d["source_dataset"] + np.where(d["adversarial"].astype(bool), "/adversarial", "/plain"),
    )
    false_alarms = {
        name: {
            "n": int(len(g)),
            "flagged_rate": round(float(g["safety_pred"].mean()), 4),
            "flagged_rate_95ci": bootstrap_ci_rate(g["safety_pred"].to_numpy(), n_boot=int(cfg.get("robustness", {}).get("bootstrap_resamples", 1000)), seed=seed),
            "by_source_and_framing": {k: {"n": int(len(v)), "flagged_rate": round(float(v["safety_pred"].mean()), 4)} for k, v in g.groupby("group")},
        }
        for name, g in harmless.groupby("creative")
    }

    # 11. Uncertainty: bootstrap intervals over prompts (spec §16), and spread across re-sampled seeds
    rcfg = cfg.get("robustness", {})
    n_boot = int(rcfg.get("bootstrap_resamples", 1000))
    ci = {
        "creative_head_test": bootstrap_ci(test_df["creative_label"].to_numpy(), pred_df["creative_probability"].to_numpy(), thr_c, n_boot, seed),
        "safety_head_test": bootstrap_ci(test_df["safety_label"].to_numpy(), pred_df["safety_probability"].to_numpy(), thr_s, n_boot, seed),
        "safety_head_wildguardtest": bootstrap_ci(wg_y, wg_probs, thr_s, n_boot, seed),
        "safety_head_xstest": bootstrap_ci(xs_y, xs_probs, thr_s, n_boot, seed),
    }
    if fiction_metrics is not None:
        ci["safety_head_fictional_framing_slice"] = bootstrap_ci(
            fiction_eval["safety_label"].to_numpy(dtype=int), fiction_eval["safety_probability"].to_numpy(), thr_s, n_boot, seed
        )

    def _seed_row(split_seed, tst, p_c, p_s, t_c, t_s, wg_p, xs_p):
        return {"seed": split_seed, "thresholds": {"creative": t_c, "safety": t_s}, "metrics": {
            "creative_test": _key_metrics(tst["creative_label"].to_numpy(), p_c, t_c),
            "safety_test": _key_metrics(tst["safety_label"].to_numpy(), p_s, t_s),
            "safety_wildguardtest": _key_metrics(wg_y, wg_p, t_s),
            "safety_xstest": _key_metrics(xs_y, xs_p, t_s),
        }}

    seed_rows = [_seed_row(seed, test_df, pred_df["creative_probability"].to_numpy(), pred_df["safety_probability"].to_numpy(),
                           thr_c, thr_s, wg_probs, xs_probs)]
    extra_seeds = [int(x) for x in rcfg.get("extra_seeds", [])]
    if quick:
        extra_seeds = extra_seeds[: int(cfg["quick_test"].get("max_extra_seeds", 1))]
    for s_seed in extra_seeds:
        # Same encoder and C; new sample, split, heads, and validation thresholds
        f_df, _ = _build_split(labelled, dcfg, targets, split_seed=s_seed, minhash_seed=seed, safety_only=fiction_for_split)
        tr, va, te = (f_df[f_df["split"] == sp].reset_index(drop=True) for sp in ["train", "validation", "test"])
        te = te[te["creative_label"].notna()].reset_index(drop=True)
        te = te.assign(creative_label=te["creative_label"].astype(int), safety_label=te["safety_label"].astype(int))
        _, ch, sh, _, e = train_dual_head_classifier(tr, va, cfg, embedder, best_C)
        t_c, t_s, _, _ = _choose_thresholds(va, ch, sh, e["val"], icfg)
        X_te = embedder.transform(te["prompt"].tolist())
        X_wg = embedder.transform([normalize_prompt_text(p) for p in wg_eval["prompt"]])
        X_xs = embedder.transform([normalize_prompt_text(p) for p in xs_eval["prompt"].astype(str)])
        seed_rows.append(_seed_row(s_seed, te, ch.predict_proba(X_te)[:, 1], sh.predict_proba(X_te)[:, 1], t_c, t_s,
                                   sh.predict_proba(X_wg)[:, 1], sh.predict_proba(X_xs)[:, 1]))
        logger.info("Finished extra seed", seed=s_seed)
    seed_variation = {"seeds": seed_rows, "summary": _spread(seed_rows) if len(seed_rows) > 1 else None}
    save_json(seed_variation, results_dir / "seed_variation.json")

    metrics_out = {
        "quick_test": quick,
        "model_selection": {
            "selected_embedding_model": embedder.model_name, "selected_C": best_C,
            "candidates": [{k: r[k] for k in ["embedding_model", "best_C", "best_C_at_grid_edge", "score_mean_best_val_roc_auc", "selected"]}
                           for r in selection_table],
        },
        "creative_head": _strip_curves(cre_metrics),
        "safety_head": _strip_curves(saf_metrics),
        "safety_head_wildguardtest": {**_strip_curves(wg_metrics), "n": int(len(wg_eval))},
        "safety_false_alarms_on_harmless_test_prompts": false_alarms,
        "safety_head_xstest": xstest_out,
        "safety_head_fictional_framing_slice": {"summary": fiction_summary, "metrics": fiction_metrics},
        "bootstrap_95ci": ci,
        "seed_variation_summary": seed_variation["summary"],
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
