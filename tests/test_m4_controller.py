"""Unit tests for the M4 controller (no GPU, no downloads). Run: `pytest tests`."""
import numpy as np
import pandas as pd

from src.alignment.controller import SweepTable, bootstrap, calibrate


def _per_prompt():
    """Two lambdas. Creative-safe prompt c: diverse at 0, bland at 1. Harmful prompt h: unsafe at 0, safe at 1."""
    rows = []
    for lam in (0.0, 1.0):
        rows.append({"lambda": lam, "prompt_id": "c", "quadrant": "1_0", "safety_label": 0, "harmful_share": 0.0,
                     "refusal_share": 0.0, "stopped_share": 1.0, "broken_share": 0.0,
                     "diversity": 0.5 if lam == 0 else 0.2, "cluster_count": 3 if lam == 0 else 1})
        rows.append({"lambda": lam, "prompt_id": "h", "quadrant": "0_1", "safety_label": 1,
                     "harmful_share": 1.0 if lam == 0 else 0.2, "refusal_share": 0.0 if lam == 0 else 0.8,
                     "stopped_share": 1.0, "broken_share": 0.0, "diversity": np.nan, "cluster_count": np.nan})
    return pd.DataFrame(rows)


def test_fixed_and_policy_scores():
    t = SweepTable(_per_prompt(), [0.0, 1.0])
    assert t.score(t.fixed_index(0.0))["creativity"] == 0.5 and t.score(t.fixed_index(1.0))["safety"] == 0.8
    groups = pd.Series({"c": "1_0", "h": "0_1"})
    s = t.score(t.policy_index(groups, {"0_0": 1.0, "0_1": 1.0, "1_0": 0.0, "1_1": 1.0}))
    assert s["creativity"] == 0.5 and s["safety"] == 0.8


def test_calibrate_routes_creative_low_and_harmful_high():
    t = SweepTable(_per_prompt(), [0.0, 1.0])
    groups = pd.Series({"c": "1_0", "h": "0_1"})
    best = calibrate(t, groups, safety_floor=t.score(t.fixed_index(1.0))["safety"])
    assert best["lambda_by_group"]["1_0"] == 0.0 and best["lambda_by_group"]["0_1"] == 1.0


def test_misrouted_harmful_prompt_breaks_the_floor():
    t = SweepTable(_per_prompt(), [0.0, 1.0])
    wrong = pd.Series({"c": "1_0", "h": "1_0"})          # classifier calls the harmful prompt creative-safe
    best = calibrate(t, wrong, safety_floor=0.8)
    assert best["lambda_by_group"]["1_0"] == 1.0          # only way to stay safe is lambda = 1 for that group


def test_bootstrap_difference_is_zero_against_itself():
    t = SweepTable(_per_prompt(), [0.0, 1.0])
    b = bootstrap(t, {"fixed_1": t.fixed_index(1.0)}, reference="fixed_1", n_boot=50)
    assert (b["diff_vs_fixed_1"] == 0).all() and (b["diff_ci_low"] == 0).all()
