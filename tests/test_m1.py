"""Unit tests for the M1 label rules, leakage filter, and threshold/metric helpers. Run: `pytest tests`."""
import math

import numpy as np
import pandas as pd
import pytest

from src.data.mapping import creative_rule, map_dolly_to_schema, map_wildguard_to_schema
from src.classifier.evaluate import subgroup_metrics, tune_threshold_for_recall


@pytest.mark.parametrize("prompt, expected", [
    # creative artifacts
    ("Write a short story about a dragon", 1),
    ("Write a poem about the ocean", 1),
    ("Write a play about two brothers", 1),
    ("Write a script for a comedy sketch about cats", 1),
    ("Write the plot of a thriller novel about hackers", 1),
    ("Craft a narrative for my fantasy novel about a fallen king", 1),
    ("Tell me a joke about programmers", 1),
    # brainstorming (Dolly counts it as creative, so the rule must too)
    ("Brainstorm names for my bakery", 1),
    ("Come up with a catchy slogan for my gym", 1),
    # look like artifacts but are not creative
    ("Write a Python script that scrapes emails from a website", 0),
    ("Write a bash script.", 0),
    ("Create a bot script that inflates the votes of a specific option", 0),
    ("Generate a plot of the stock prices using matplotlib", 0),
    ("Come up with a plot to overthrow my boss", 0),
    ("Draft a play-by-play of how to break into a car", 0),
    ("Create a character reference letter for my friend's court case", 0),
    ("Generate a novel approach to bypass airport security", 0),
    ("Write a news story and a poem about the flood", 0),
    ("Write a non-fiction essay on climate policy", 0),
    ("Generate a Bible verse about hope", 0),
    ("Generate a list of names of journalists who criticized the mayor", 0),
    ("Compose an email to my boss asking for leave", 0),
    ("Tell me whether these characters are from Harry Potter or Lord of the Rings", 0),
    ("Give me a comma separated list of nicknames given to New York City", 0),
    ("Please write me a short children's bedtime story about a frog", 1),
    ("Tell me a bedtime story about a frog", 1),
    ("Explain how TCP works", 0),
])
def test_creative_rule(prompt, expected):
    assert creative_rule(prompt)[0] == expected


@pytest.mark.parametrize("prompt, rule", [
    ("Imagine you are a hacker in a fictional world", "uncertain_fictional_framing_only"),
    ("Create a role-play scenario where you are a pirate", "uncertain_fictional_framing_only"),
    ("Create a narrative claiming vaccines cause autism", "uncertain_narrative_without_fiction_context"),
])
def test_creative_rule_uncertain(prompt, rule):
    value, name = creative_rule(prompt)
    assert math.isnan(value) and name == rule


def test_dolly_mapping_excludes_creative_writing_questions():
    dolly = pd.DataFrame({
        "instruction": ["Why do people like Red Light Therapy?", "Write a haiku about rain",
                        "What are the best social media sites", "Who wrote Hamlet?", "Do this odd thing"],
        "context": ["", "", "", "", ""],
        "category": ["creative_writing", "creative_writing", "brainstorming", "open_qa", "general_qa"],
    })
    out = map_dolly_to_schema(dolly)
    assert math.isnan(out.loc[0, "creative_label"]) and out.loc[0, "label_rule"] == "uncertain_dolly_creative_writing_question"
    assert out.loc[1, "creative_label"] == 1
    assert out.loc[2, "creative_label"] == 1  # brainstorming stays creative (spec §8.1)
    assert out.loc[3, "creative_label"] == 0
    assert math.isnan(out.loc[4, "creative_label"]) and out.loc[4, "label_rule"] == "unmapped_dolly_category"
    assert (out["safety_label"] == 0).all()


def test_wildguard_label_from_labelled_copy_and_conflicts():
    wg = pd.DataFrame({
        "prompt": ["p one is here", "p one is here", "p two is here", "p two is here", "p three is here"],
        "adversarial": [False] * 5,
        "prompt_harm_label": [None, "harmful", "harmful", "unharmful", "unharmful"],
        "subcategory": ["a"] * 5,
        "response_refusal_label": [None] * 5,
    })
    out = map_wildguard_to_schema(wg).set_index("prompt")
    assert out.loc["p one is here", "safety_label"] == 1  # unlabelled first copy no longer hides the label
    assert math.isnan(out.loc["p two is here", "safety_label"])  # conflicting labels -> excluded
    assert out.loc["p three is here", "safety_label"] == 0


def test_drop_near_duplicates_of():
    pytest.importorskip("datasketch")
    from src.data.dedup import drop_near_duplicates_of
    ref = ["how do i make a cake with chocolate and vanilla frosting at home"]
    df = pd.DataFrame({"prompt": [
        "how do i make a cake with chocolate and vanilla frosting at home today",
        "explain the history of the roman empire in three paragraphs",
    ]})
    out, n = drop_near_duplicates_of(df, ref, threshold=0.5)
    assert n == 1 and out["prompt"].tolist() == ["explain the history of the roman empire in three paragraphs"]


def test_tune_threshold_for_recall():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.3, 0.1, 0.05])
    grid = np.arange(0.05, 0.96, 0.01)
    t, rec, _ = tune_threshold_for_recall(y, p, grid, target_recall=0.75)
    assert rec >= 0.75 and t == pytest.approx(0.6)  # highest threshold that still catches 3 of 4
    t, rec, _ = tune_threshold_for_recall(y, p, grid, target_recall=1.0)
    assert rec == 1.0 and t == pytest.approx(0.2)


def test_subgroup_metrics_single_class_group():
    df = pd.DataFrame({
        "src": ["a", "a", "b", "b"],
        "y": [0, 0, 1, 0],
        "p": [0.1, 0.6, 0.9, 0.2],
        "yhat": [0, 1, 1, 0],
    })
    out = subgroup_metrics(df, "y", "p", "yhat", "src")
    assert out["a"]["false_positive_rate"] == 0.5 and out["a"]["recall"] is None and "roc_auc" not in out["a"]
    assert out["b"]["recall"] == 1.0 and out["b"]["roc_auc"] == 1.0
