"""
Label construction and schema standardization for ALIGN-PARETO (M1).
Maps Databricks Dolly-15k and AllenAI WildGuardMix into the unified dual-label schema (spec §7):
  id, prompt, creative_label, safety_label, source_dataset, source_label, source_subcategory,
  plus provenance columns (label_rule, adversarial, response_refusal_label).
Rows whose label cannot be assigned confidently get NaN and are excluded later, never guessed.
"""
import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# Dolly categories: spec §8.1 / DATASET_DECISION.md. Anything else (general_qa) stays unassigned.
DOLLY_CREATIVE = {"creative_writing", "brainstorming"}
DOLLY_NON_CREATIVE = {"closed_qa", "open_qa", "information_extraction", "summarization", "classification"}

# WildGuardMix has no creative annotation. Strict rule (team decision): creative=1 only when the prompt
# explicitly asks for a creative artifact; fictional framing alone ("imagine", "roleplay") is uncertain.
_ARTIFACT = (
    r"(?:short )?(?:story|stories|poem|poems|poetry|song|songs|lyrics|haiku|limerick|sonnet|ballad|"
    r"screenplay|script|play|novel|novella|fiction|fanfic|fan fiction|fable|fairy tale|tale|narrative|"
    r"scene|dialogue|monologue|rap|verse|character|characters|plot)"
)
CREATIVE_REQUEST_RE = re.compile(
    r"\b(?:write|compose|create|craft|draft|pen|generate|produce|come up with|invent|tell me)\b"
    r"(?:\W+\w+){0,4}?\W+" + _ARTIFACT + r"\b",
    re.IGNORECASE,
)
# "news story", "cover story" etc. are not creative artifacts
NON_CREATIVE_ARTIFACT_RE = re.compile(r"\b(?:news|cover|success|back) stor(?:y|ies)\b", re.IGNORECASE)
FICTIONAL_FRAMING_RE = re.compile(
    r"\b(?:imagine|pretend|role-?play|as a character|in a (?:fictional|hypothetical|alternate|parallel|fantastical|"
    r"fantasy) (?:world|universe|setting|scenario|realm)|you are an? (?:writer|novelist|author|playwright|poet))\b",
    re.IGNORECASE,
)


def creative_rule(prompt: str) -> Tuple[float, str]:
    """Return (creative_label or NaN, rule name) for a prompt without a source creative label."""
    if CREATIVE_REQUEST_RE.search(prompt) and not NON_CREATIVE_ARTIFACT_RE.search(prompt):
        return 1.0, "rule_explicit_creative_request"
    if FICTIONAL_FRAMING_RE.search(prompt):
        return np.nan, "uncertain_fictional_framing_only"
    return 0.0, "rule_no_creative_cue"


def map_dolly_to_schema(dolly_df: pd.DataFrame) -> pd.DataFrame:
    """Dolly: creative label from category; prompt is the instruction only (context kept separately)."""
    cat = dolly_df["category"].astype(str).str.strip().str.lower()
    creative = np.where(cat.isin(DOLLY_CREATIVE), 1.0, np.where(cat.isin(DOLLY_NON_CREATIVE), 0.0, np.nan))
    return pd.DataFrame({
        "id": [f"dolly_{i:05d}" for i in range(len(dolly_df))],
        "prompt": dolly_df["instruction"].astype(str),
        "creative_label": creative,
        # Dolly has no harm annotation; provisional default per DATASET_DECISION.md
        "safety_label": 0.0,
        "source_dataset": "dolly",
        "source_label": cat.values,
        "source_subcategory": cat.values,
        "label_rule": np.where(np.isnan(creative), "unmapped_dolly_category", "dolly_category"),
        "has_context": dolly_df["context"].fillna("").astype(str).str.strip().ne("").values,
        "adversarial": False,
        "response_refusal_label": None,
    })


def map_wildguard_to_schema(wg_df: pd.DataFrame, id_prefix: str = "wildguard") -> pd.DataFrame:
    """
    WildGuardMix: safety label from prompt_harm_label (missing -> NaN), creative label from the strict rule.
    The train split repeats prompts with different responses; prompt-level fields are kept once per prompt,
    and prompts whose harm label disagrees across rows are marked unassigned.
    """
    harm = wg_df["prompt_harm_label"].astype("string").str.strip().str.lower()
    conflict = harm.groupby(wg_df["prompt"]).transform("nunique") > 1
    df = wg_df.assign(_harm=harm, _conflict=conflict).drop_duplicates("prompt", keep="first").reset_index(drop=True)

    safety = df["_harm"].map({"harmful": 1.0, "unharmful": 0.0}).astype(float).to_numpy(na_value=np.nan)
    safety = np.where(df["_conflict"], np.nan, safety)
    rules = [creative_rule(p) for p in df["prompt"].astype(str)]

    return pd.DataFrame({
        "id": [f"{id_prefix}_{i:05d}" for i in range(len(df))],
        "prompt": df["prompt"].astype(str),
        "creative_label": [r[0] for r in rules],
        "safety_label": safety,
        "source_dataset": "wildguardmix",
        "source_label": df["_harm"].fillna("missing").values,
        "source_subcategory": df["subcategory"].fillna("unknown").astype(str).values,
        "label_rule": [r[1] for r in rules],
        "has_context": False,
        "adversarial": df["adversarial"].astype(bool).values,
        "response_refusal_label": df["response_refusal_label"].values if "response_refusal_label" in df else None,
    })


def drop_unassigned(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Remove rows with an unknown label, reporting why each was removed."""
    mask = df["creative_label"].isna() | df["safety_label"].isna()
    reasons = df.loc[mask].groupby(["source_dataset", "label_rule"]).size()
    stats = {
        "rows_before": int(len(df)),
        "removed_unassigned": int(mask.sum()),
        "removed_by_reason": {f"{s}/{r}": int(n) for (s, r), n in reasons.items()},
        "removed_missing_safety": int(df["safety_label"].isna().sum()),
    }
    out = df.loc[~mask].copy()
    out["creative_label"] = out["creative_label"].astype(int)
    out["safety_label"] = out["safety_label"].astype(int)
    return out.reset_index(drop=True), stats


def sample_quadrants(df: pd.DataFrame, targets: Dict[str, int], seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Sample up to `targets["creative{c}_safety{s}"]` prompts per quadrant. Within a quadrant the target is
    split evenly across source datasets (shortfall filled from the other source) so that neither head can
    identify a label from dataset style alone. Quadrants with fewer prompts are kept whole, not oversampled.
    """
    rng = np.random.default_rng(seed)
    parts, report = [], {}
    for key, target in targets.items():
        c, s = int(key[8]), int(key[-1])
        quad = df[(df["creative_label"] == c) & (df["safety_label"] == s)]
        sources = sorted(quad["source_dataset"].unique())
        avail = {src: quad.index[quad["source_dataset"] == src].to_numpy() for src in sources}
        alloc = {src: 0 for src in sources}
        remaining = int(target)
        # Water-filling: give each source an equal share, redistribute what small sources cannot use
        open_src = [s_ for s_ in sources if len(avail[s_]) > 0]
        while remaining > 0 and open_src:
            share = max(1, remaining // len(open_src))
            for src in list(open_src):
                take = min(share, len(avail[src]) - alloc[src], remaining)
                alloc[src] += take
                remaining -= take
                if alloc[src] >= len(avail[src]):
                    open_src.remove(src)
                if remaining == 0:
                    break
        for src in sources:
            if alloc[src]:
                parts.append(rng.choice(avail[src], size=alloc[src], replace=False))
        report[key] = {"target": int(target), "available": int(len(quad)), "selected": {k: int(v) for k, v in alloc.items()}}
    idx = np.sort(np.concatenate(parts)) if parts else np.array([], dtype=int)
    return df.loc[idx].reset_index(drop=True), report


def creative_rule_audit_sample(df: pd.DataFrame, n_per_rule: int = 100, seed: int = 42) -> pd.DataFrame:
    """Random WildGuardMix rows per creative-rule outcome, with a blank column for human labels."""
    wg = df[df["source_dataset"] == "wildguardmix"]
    parts = [
        g.sample(min(n_per_rule, len(g)), random_state=seed)
        for _, g in wg.groupby("label_rule")
    ]
    out = pd.concat(parts)[["id", "label_rule", "creative_label", "safety_label", "adversarial", "prompt"]]
    out["human_creative_label"] = ""
    return out.reset_index(drop=True)


def verify_label_independence(df: pd.DataFrame) -> dict:
    """Check that creative_label and safety_label are not collinear (Pearson r and chi-square)."""
    c = df["creative_label"].to_numpy(dtype=float)
    s = df["safety_label"].to_numpy(dtype=float)
    corr = float(np.corrcoef(c, s)[0, 1])

    ct = pd.crosstab(df["creative_label"], df["safety_label"])
    chi2, p_val, _, _ = chi2_contingency(ct)

    q_counts = {
        f"creative{ci}_safety{si}": int(((df["creative_label"] == ci) & (df["safety_label"] == si)).sum())
        for ci in (0, 1) for si in (0, 1)
    }
    return {
        "pearson_r": round(corr, 4),
        "abs_pearson_r_below_0_2": bool(abs(corr) < 0.20),
        "chi2_statistic": round(float(chi2), 4),
        "chi2_p_value": float(p_val),
        "quadrant_counts": q_counts,
        "all_four_quadrants_present": all(v > 50 for v in q_counts.values()),
    }
