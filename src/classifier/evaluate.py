"""
Comprehensive Evaluation & Visualization Suite for ALIGN-PARETO Milestone 1 (M1).
Computes per-head Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Brier Score,
Expected Calibration Error (ECE), Multilabel Exact-Match Accuracy, Macro-F1,
and Error Analysis (including critical Safety False Negatives).
Generates all 9 publication-grade diagnostic plots.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# numpy < 2.0 (older Kaggle images) only has trapz
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# Set clean publication styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8


def compute_roc_curve_and_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute exact ROC curve (FPR, TPR) and trapezoidal AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    desc_idx = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_idx]
    distinct_idx = np.where(np.diff(y_prob[desc_idx]))[0]
    threshold_idxs = np.r_[distinct_idx, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / max(1, fps[-1])
    tpr = tps / max(1, tps[-1])
    auc = float(_trapezoid(tpr, fpr))
    return fpr, tpr, auc


def compute_pr_curve_and_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Precision-Recall curve and Average Precision (PR-AUC)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    desc_idx = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_idx]

    tps = np.cumsum(y_true_sorted)
    fps = np.arange(1, len(y_true) + 1) - tps

    precision = tps / np.maximum(1, tps + fps)
    recall = tps / max(1, int(np.sum(y_true)))

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    pr_auc = float(_trapezoid(precision, recall))
    return recall, precision, pr_auc


def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 8) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute reliability diagram bins and Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    mean_pred, frac_pos = [], []
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            mp = float(np.mean(y_prob[mask]))
            fp = float(np.mean(y_true[mask]))
            mean_pred.append(mp)
            frac_pos.append(fp)
            ece += (np.sum(mask) / n) * abs(mp - fp)
    return np.array(mean_pred), np.array(frac_pos), float(ece)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, len(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))

    fpr, tpr, roc_auc = compute_roc_curve_and_auc(y_true, y_prob)
    rec_curve, prec_curve, pr_auc = compute_pr_curve_and_auc(y_true, y_prob)
    cal_pred, cal_true, ece = compute_calibration_curve(y_true, y_prob)

    return {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "_curves": {
            "fpr": fpr,
            "tpr": tpr,
            "rec_curve": rec_curve,
            "prec_curve": prec_curve,
            "cal_pred": cal_pred,
            "cal_true": cal_true,
        },
    }


def generate_all_m1_plots(
    full_df: pd.DataFrame,
    cre_metrics: Dict,
    saf_metrics: Dict,
    plots_dir: Path,
) -> List[str]:
    """Generate all 9 required M1 diagnostic plots in `results/plots/`."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # 1 & 2: Confusion Matrices
    for label_name, m, cmap in [
        ("creative", cre_metrics, "Blues"),
        ("safety", saf_metrics, "OrRd"),
    ]:
        cm = m["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(5.5, 4.6), dpi=200)
        sns.heatmap(
            mat,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            xticklabels=[f"Pred 0 (Non-{label_name.title()})", f"Pred 1 ({label_name.title()})"],
            yticklabels=[f"True 0 (Non-{label_name.title()})", f"True 1 ({label_name.title()})"],
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(f"{label_name.title()} Head — Test Confusion Matrix (F1 = {m['f1']:.3f})", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylabel("Ground Truth Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        fig.tight_layout()
        p = plots_dir / f"confusion_matrix_{label_name}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        saved_files.append(str(p))

    # 3 & 4: ROC Curves
    for label_name, m, color in [
        ("creative", cre_metrics, "#2563EB"),
        ("safety", saf_metrics, "#DC2626"),
    ]:
        fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
        c = m["_curves"]
        ax.plot(c["fpr"], c["tpr"], color=color, lw=2.5, label=f"{label_name.title()} Head (AUC = {m['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.2, linestyle="--", label="Chance Baseline (0.500)")
        ax.fill_between(c["fpr"], c["tpr"], alpha=0.12, color=color)
        ax.set_title(f"ROC Curve — {label_name.title()} Classifier Head", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=10)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=10)
        ax.legend(loc="lower right", frameon=True, fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        p = plots_dir / f"roc_{label_name}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        saved_files.append(str(p))

    # 5 & 6: PR Curves
    for label_name, m, color in [
        ("creative", cre_metrics, "#1D4ED8"),
        ("safety", saf_metrics, "#B91C1C"),
    ]:
        fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
        c = m["_curves"]
        ax.plot(c["rec_curve"], c["prec_curve"], color=color, lw=2.5, label=f"{label_name.title()} Head (PR-AUC = {m['pr_auc']:.4f})")
        ax.fill_between(c["rec_curve"], c["prec_curve"], alpha=0.12, color=color)
        ax.set_title(f"Precision-Recall Curve — {label_name.title()} Classifier Head", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left", frameon=True, fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        p = plots_dir / f"pr_curve_{label_name}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        saved_files.append(str(p))

    # 7 & 8: Calibration Reliability Diagrams
    for label_name, m, color in [
        ("creative", cre_metrics, "#0284C7"),
        ("safety", saf_metrics, "#EA580C"),
    ]:
        fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
        c = m["_curves"]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", lw=1.5, label="Perfectly Calibrated")
        ax.plot(
            c["cal_pred"],
            c["cal_true"],
            marker="o",
            markersize=7,
            color=color,
            lw=2.2,
            label=f"{label_name.title()} Head (ECE = {m['ece']:.4f}, Brier = {m['brier_score']:.4f})",
        )
        ax.set_title(f"Reliability Diagram — {label_name.title()} Probability Calibration", fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Empirical Fraction of Positives", fontsize=10)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        p = plots_dir / f"calibration_{label_name}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        saved_files.append(str(p))

    # 9: Label Quadrant Distribution across Splits
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    quad_labels = [
        "Q1: Factual & Safe\n(creative=0, safety=0)",
        "Q2: Creative & Safe\n(creative=1, safety=0)",
        "Q3: Non-Creative Harmful\n(creative=0, safety=1)",
        "Q4: Creative & Harmful\n(creative=1, safety=1)",
    ]
    splits = ["train", "validation", "test"]
    colors = ["#1E40AF", "#0EA5E9", "#F59E0B"]
    x = np.arange(len(quad_labels))
    width = 0.25

    for idx, (sp, col) in enumerate(zip(splits, colors)):
        sub = full_df[full_df["split"] == sp]
        counts = [
            int(((sub["creative_label"] == 0) & (sub["safety_label"] == 0)).sum()),
            int(((sub["creative_label"] == 1) & (sub["safety_label"] == 0)).sum()),
            int(((sub["creative_label"] == 0) & (sub["safety_label"] == 1)).sum()),
            int(((sub["creative_label"] == 1) & (sub["safety_label"] == 1)).sum()),
        ]
        bars = ax.bar(x + (idx - 1) * width, counts, width, label=f"{sp.title()} Split (n={len(sub)})", color=col)
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_title("ALIGN-PARETO M1 Dataset Distribution Across All 4 Quadrants & Splits", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(quad_labels, fontsize=9)
    ax.set_ylabel("Number of Prompts", fontsize=10)
    ax.legend(frameon=True, fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = plots_dir / "label_distribution.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    saved_files.append(str(p))

    return saved_files



def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> Tuple[float, float]:
    """Return (threshold, F1) maximizing F1 over `grid`; used on the validation split only."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = y_prob >= t
        tp = np.sum(pred & (y_true == 1))
        prec = tp / max(1, pred.sum())
        rec = tp / max(1, (y_true == 1).sum())
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return round(best_t, 4), round(best_f1, 4)


def tune_threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray, target_recall: float) -> Tuple[float, float, float]:
    """
    Return (threshold, recall, precision) for the highest threshold in `grid` whose recall is >= target_recall.
    Used for the safety head, where a missed harmful prompt costs more than a false alarm (spec §16).
    Falls back to the lowest grid value if no threshold reaches the target. Validation split only.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = max(1, int((y_true == 1).sum()))
    best = None
    for t in sorted(round(float(g), 4) for g in grid):  # np.arange gives e.g. 0.6000000000000001
        pred = y_prob >= t
        tp = int(np.sum(pred & (y_true == 1)))
        rec = tp / n_pos
        prec = tp / max(1, int(pred.sum()))
        if rec >= target_recall:
            best = (float(t), rec, prec)
    if best is None:
        t = float(min(grid))
        pred = y_prob >= t
        tp = int(np.sum(pred & (y_true == 1)))
        best = (t, tp / n_pos, tp / max(1, int(pred.sum())))
    return round(best[0], 4), round(best[1], 4), round(best[2], 4)


def subgroup_metrics(df: pd.DataFrame, label_col: str, prob_col: str, pred_col: str, group_col: str) -> Dict:
    """
    Per-group counts and error rates for one head, so a strong overall number cannot hide an easy subgroup
    (e.g. Dolly prompts are all safety=0). ROC-AUC is only reported when both classes occur in the group.
    """
    out = {}
    for g, sub in df.groupby(group_col):
        y = sub[label_col].to_numpy(dtype=int)
        yhat = sub[pred_col].to_numpy(dtype=int)
        tp = int(((y == 1) & (yhat == 1)).sum())
        fp = int(((y == 0) & (yhat == 1)).sum())
        fn = int(((y == 1) & (yhat == 0)).sum())
        tn = int(((y == 0) & (yhat == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        row = {
            "n": int(len(sub)),
            "n_positive": int(y.sum()),
            "accuracy": round((tp + tn) / max(1, len(sub)), 4),
            "precision": round(prec, 4) if (tp + fp) else None,
            "recall": round(rec, 4) if (tp + fn) else None,
            "f1": round(2 * prec * rec / max(1e-12, prec + rec), 4) if (tp + fn) and (tp + fp) else None,
            "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,
        }
        if 0 < y.sum() < len(y):
            row["roc_auc"] = round(compute_roc_curve_and_auc(y, sub[prob_col].to_numpy(dtype=float))[2], 4)
        out[str(g)] = row
    return out


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> Dict:
    """
    95% (by default) percentile bootstrap intervals over prompts for precision, recall, F1, and ROC-AUC
    at a fixed threshold. Resamples where a class is missing are skipped for ROC-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {"precision": [], "recall": [], "f1": [], "roc_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y, p = y_true[idx], y_prob[idx]
        pred = p >= threshold
        tp = int(np.sum(pred & (y == 1)))
        n_pred, n_pos = int(pred.sum()), int((y == 1).sum())
        prec = tp / n_pred if n_pred else np.nan
        rec = tp / n_pos if n_pos else np.nan
        stats["precision"].append(prec)
        stats["recall"].append(rec)
        stats["f1"].append(2 * prec * rec / (prec + rec) if n_pred and n_pos and (prec + rec) > 0 else np.nan)
        stats["roc_auc"].append(compute_roc_curve_and_auc(y, p)[2] if 0 < n_pos < n else np.nan)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for k, v in stats.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        out[k] = [round(float(np.percentile(v, lo)), 4), round(float(np.percentile(v, hi)), 4)] if len(v) else None
    out["n"] = n
    out["n_boot"] = n_boot
    return out


def bootstrap_ci_rate(flags: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap interval for a rate (mean of 0/1 flags), e.g. the false-alarm rate."""
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = flags[rng.integers(0, len(flags), (n_boot, len(flags)))].mean(axis=1)
    return [round(float(np.percentile(means, 100 * alpha / 2)), 4), round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4)]
