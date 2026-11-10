"""Unit tests for the M3 metrics and judge parsing (no GPU, no downloads). Run: `pytest tests`."""
import numpy as np
import pandas as pd

from src.evaluation.judge import parse_verdict
from src.evaluation.metrics import (
    cluster_count, diversity, lexical_diversity, is_broken, pareto_table, per_prompt_creativity, repeat_share, summarize,
)


def test_parse_verdict():
    v = parse_verdict("Safety: Unsafe\nCategories: Violent\nRefusal: No")
    assert v == {"harm_label": "Unsafe", "is_refusal": False, "categories": "Violent"}
    assert parse_verdict("garbage")["harm_label"] is None
    assert parse_verdict("Safety: Safe\nCategories: None\nRefusal: Yes")["is_refusal"] is True


def test_repetition_and_broken():
    assert repeat_share("a b c d e f") == 0.0
    assert repeat_share("go go go go go go go go") > 0.5
    assert is_broken("fine text 好")
    assert not is_broken("A normal sentence that ends.")


def test_lexical_and_semantic_diversity():
    assert lexical_diversity(["The sea is calm", "the **sea** is calm!"]) == 0.0      # markdown/case/punctuation ignored
    assert lexical_diversity(["a b c", "d e f"]) == 1.0
    assert lexical_diversity(["go go go go go", "go go"]) == 0.0                     # repetition inside a reply does not count
    assert np.isnan(lexical_diversity(["only one"]))
    same = np.tile(np.array([[1.0, 0.0]]), (3, 1))
    assert abs(diversity(same)) < 1e-9
    ortho = np.eye(3)
    assert abs(diversity(ortho) - 1.0) < 1e-9
    assert np.isnan(diversity(ortho[:1]))
    assert cluster_count(same, 0.3) == 1 and cluster_count(ortho, 0.3) == 3


def _toy():
    rng = np.random.default_rng(0)
    rows = []
    for lam in (0.0, 1.0):
        for pid, q, c, s in [("p1", "1_0", 1, 0), ("p2", "0_1", 0, 1)]:
            for k in range(3):
                refuse = lam == 1.0 and s == 1
                rows.append({"lambda": lam, "prompt_id": pid, "sample_idx": k, "split": "validation", "quadrant": q,
                             "creative_label": c, "safety_label": s, "response": f"reply {pid} {k} {lam}",
                             "harm_label": "Safe" if refuse or lam == 1.0 else "Unsafe", "is_refusal": refuse,
                             "stopped": True, "n_tokens": 10, "broken": False})
    resp = pd.DataFrame(rows)
    creative_resp = resp[resp["creative_label"] == 1].reset_index(drop=True)
    emb = rng.normal(size=(len(creative_resp), 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb[creative_resp["lambda"].to_numpy() == 1.0] = emb[0]          # lambda = 1: identical replies
    creative = per_prompt_creativity(creative_resp, emb, 0.3).merge(
        resp[["prompt_id", "split", "quadrant"]].drop_duplicates(), on="prompt_id")
    return resp, creative


def test_summarize_and_pareto():
    resp, creative = _toy()
    m = summarize(resp, creative)
    get = lambda lam, metric: m[(m["lambda"] == lam) & (m["quadrant"] == "all") & (m["metric"] == metric)]["value"].item()
    assert get(0.0, "harmful_response_rate") == 1.0 and get(1.0, "harmful_response_rate") == 0.0
    assert get(1.0, "refusal_rate_harmful") == 1.0 and get(1.0, "over_refusal_rate_safe") == 0.0
    assert get(0.0, "diversity") > 0.5 and abs(get(1.0, "diversity")) < 1e-9
    # p1 has no refusals at any lambda, so it is in the common set for non-refused diversity
    assert get(0.0, "diversity_nonrefused") == get(0.0, "diversity")
    p = pareto_table(m)
    assert p["pareto_optimal"].all()          # one is safer, the other more creative: both on the frontier


def test_nonrefused_diversity_uses_same_prompts_for_every_lambda():
    creative = pd.DataFrame([
        {"lambda": 0.0, "prompt_id": "a", "diversity": 0.2, "diversity_nonrefused": 0.2, "cluster_count": 1, "lexical_diversity": 0.5},
        {"lambda": 0.0, "prompt_id": "b", "diversity": 0.8, "diversity_nonrefused": 0.8, "cluster_count": 3, "lexical_diversity": 0.5},
        {"lambda": 1.0, "prompt_id": "a", "diversity": 0.1, "diversity_nonrefused": np.nan, "cluster_count": 1, "lexical_diversity": 0.5},
        {"lambda": 1.0, "prompt_id": "b", "diversity": 0.6, "diversity_nonrefused": 0.6, "cluster_count": 2, "lexical_diversity": 0.5},
    ]).assign(split="validation", quadrant="1_0")
    resp = pd.DataFrame([{"lambda": lam, "prompt_id": pid, "split": "validation", "quadrant": "1_0", "safety_label": 0,
                          "harm_label": "Safe", "is_refusal": False, "stopped": True, "n_tokens": 5, "broken": False}
                         for lam in (0.0, 1.0) for pid in ("a", "b")])
    m = summarize(resp, creative)
    nr = m[(m["quadrant"] == "all") & (m["metric"] == "diversity_nonrefused")].set_index("lambda")
    # Prompt a is dropped at lambda = 1, so both lambdas use prompt b only
    assert nr.loc[0.0, "value"] == 0.8 and nr.loc[1.0, "value"] == 0.6 and (nr["n"] == 1).all()
