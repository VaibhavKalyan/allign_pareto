"""
M4 adaptive controller: choose lambda per prompt from the M1 classifier's prediction, calibrated on the validation
split of the M3 sweep (spec §20). No new generation: a policy's result is read from M3's judged replies at the
lambda it picks for each prompt (results/m3/per_prompt.csv).

Policy: one lambda per predicted quadrant (creative x harmful), chosen from the M3 grid to
    maximize   creativity = semantic diversity on creative, safe prompts (1_0)
    subject to safety     = 1 - harmful response rate on harmful prompts  >=  safety of fixed lambda = 1
i.e. as safe as the aligned model, but more creative. All metrics match M3's definitions (EVAL_PROTOCOL §5).
"""
import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

GROUPS = ["0_0", "0_1", "1_0", "1_1"]
CREATIVITY_QUADRANT = "1_0"


class SweepTable:
    """per_prompt.csv for one split as prompt x lambda arrays, so any policy is scored by fancy indexing."""

    def __init__(self, per_prompt: pd.DataFrame, lambdas: List[float]):
        self.lambdas = [float(x) for x in lambdas]
        pp = per_prompt[per_prompt["lambda"].isin(self.lambdas)]
        meta = pp.drop_duplicates("prompt_id").set_index("prompt_id")
        self.prompt_ids = meta.index.to_numpy()
        self.harmful_prompt = (meta["safety_label"] == 1).to_numpy()
        self.safe_prompt = ~self.harmful_prompt
        self.creative_safe = (meta["quadrant"] == CREATIVITY_QUADRANT).to_numpy()
        self.values = {}
        for col in ["harmful_share", "refusal_share", "stopped_share", "broken_share", "diversity", "cluster_count"]:
            wide = pp.pivot_table(index="prompt_id", columns="lambda", values=col, dropna=False)
            self.values[col] = wide.reindex(index=self.prompt_ids, columns=self.lambdas).to_numpy(float)

    def per_prompt(self, lam_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-prompt values under a policy (lam_idx[i] = grid index chosen for prompt i); nan where a metric does not apply."""
        rows = np.arange(len(self.prompt_ids))
        v = {k: arr[rows, lam_idx] for k, arr in self.values.items()}
        return {"safety": np.where(self.harmful_prompt, 1.0 - v["harmful_share"], np.nan),
                "creativity": np.where(self.creative_safe, v["diversity"], np.nan),
                "cluster_count": np.where(self.creative_safe, v["cluster_count"], np.nan),
                "over_refusal": np.where(self.safe_prompt, v["refusal_share"], np.nan),
                "refusal_harmful": np.where(self.harmful_prompt, v["refusal_share"], np.nan),
                "stopped": v["stopped_share"], "broken": v["broken_share"],
                "mean_lambda": np.asarray(self.lambdas)[lam_idx]}

    def score(self, lam_idx: np.ndarray) -> Dict[str, float]:
        return {k: float(np.nanmean(v)) for k, v in self.per_prompt(lam_idx).items()}

    def policy_index(self, groups: pd.Series, lam_by_group: Dict[str, float]) -> np.ndarray:
        """Grid index per prompt from its group (groups: prompt_id -> group)."""
        g = groups.reindex(self.prompt_ids).to_numpy()
        return np.array([self.lambdas.index(float(lam_by_group[x])) for x in g])

    def fixed_index(self, lam: float) -> np.ndarray:
        return np.full(len(self.prompt_ids), self.lambdas.index(float(lam)))


def calibrate(table: SweepTable, groups: pd.Series, safety_floor: float, tol: float = 1e-9) -> Dict:
    """
    Try every lambda per group (5^4 = 625 policies) and keep the most creative one whose safety >= safety_floor.
    Ties: higher safety, then lower over-refusal. Groups that hold no creative, safe prompt therefore get the
    safest lambda. Returns the policy and its validation scores.
    """
    best, best_key = None, None
    for combo in itertools.product(table.lambdas, repeat=len(GROUPS)):
        policy = dict(zip(GROUPS, combo))
        s = table.score(table.policy_index(groups, policy))
        if s["safety"] < safety_floor - tol:
            continue
        key = (round(s["creativity"], 9), round(s["safety"], 9), -round(s["over_refusal"], 9))
        if best_key is None or key > best_key:
            best, best_key = {"lambda_by_group": policy, "validation": s}, key
    if best is None:
        raise ValueError(f"No policy reaches safety {safety_floor:.3f}; the floor must be at most the safest fixed lambda.")
    return best


def bootstrap(table: SweepTable, policies: Dict[str, np.ndarray], reference: str, n_boot: int = 1000,
              seed: int = 0) -> pd.DataFrame:
    """
    95% intervals over prompts (resampled with replacement) for each policy's safety and creativity, and for the
    difference from `reference`, all on the same resamples.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(table.prompt_ids), size=(n_boot, len(table.prompt_ids)))
    per = {name: table.per_prompt(li) for name, li in policies.items()}
    rows = []
    for name, v in per.items():
        for metric in ["safety", "creativity", "over_refusal"]:
            draws = np.nanmean(v[metric][idx], axis=1)
            diff = draws - np.nanmean(per[reference][metric][idx], axis=1)
            rows.append({"policy": name, "metric": metric, "value": float(np.nanmean(v[metric])),
                         "ci_low": float(np.nanpercentile(draws, 2.5)), "ci_high": float(np.nanpercentile(draws, 97.5)),
                         f"diff_vs_{reference}": float(np.nanmean(v[metric]) - np.nanmean(per[reference][metric])),
                         "diff_ci_low": float(np.nanpercentile(diff, 2.5)), "diff_ci_high": float(np.nanpercentile(diff, 97.5))})
    return pd.DataFrame(rows)
