"""Unit tests for the shared evaluation prompt set sampler. Run: `pytest tests`."""
import pandas as pd
import pytest

from src.evaluation.build_eval_set import eligible_prompts, sample_quadrant


def _rows(source, n, family_prefix="f", **extra):
    return pd.DataFrame({
        "prompt": [f"{source} prompt {i}" for i in range(n)],
        "source_dataset": source,
        "family_id": [f"{family_prefix}{source}{i}" for i in range(n)],
        "has_context": False,
        **extra,
    })


def test_eligible_prompts_drops_context_and_long_prompts():
    df = pd.DataFrame({"prompt": ["short", "x" * 50, "needs passage"], "has_context": [False, False, True]})
    assert eligible_prompts(df, max_chars=20, exclude_has_context=True)["prompt"].tolist() == ["short"]
    assert len(eligible_prompts(df, max_chars=20, exclude_has_context=False)) == 2


def test_sample_quadrant_balances_sources_and_tops_up():
    df = pd.concat([_rows("dolly", 10), _rows("wildguardmix", 10)], ignore_index=True)
    assert sample_quadrant(df, 8, seed=0)["source_dataset"].value_counts().to_dict() == {"dolly": 4, "wildguardmix": 4}
    df = pd.concat([_rows("dolly", 2), _rows("wildguardmix", 10)], ignore_index=True)
    assert sample_quadrant(df, 8, seed=0)["source_dataset"].value_counts().to_dict() == {"wildguardmix": 6, "dolly": 2}


def test_sample_quadrant_one_prompt_per_family():
    df = _rows("wildguardmix", 6)
    df["family_id"] = ["a", "a", "a", "b", "b", "c"]
    assert sample_quadrant(df, 3, seed=0)["family_id"].nunique() == 3
    with pytest.raises(ValueError):
        sample_quadrant(df, 4, seed=0)
