"""Unit tests for the M2 SFT data (no_robots, oasst2). Run: `pytest tests`."""
import numpy as np
import pandas as pd

from src.alignment.sft_data import filter_examples, is_refusal, no_robots_examples, oasst2_examples


def test_is_refusal():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("As an AI language model, I have no opinions.")
    assert not is_refusal("Here is a poem about rain.")
    assert not is_refusal("Sorry to hear that! Here are some tips.")


def test_no_robots_drops_system_prompts():
    raw = pd.DataFrame({"prompt_id": ["a", "b"], "messages": [
        [{"role": "user", "content": "Write a haiku"}, {"role": "assistant", "content": "Rain on tin roofs"}],
        [{"role": "system", "content": "You are a pirate"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Arr"}],
    ]})
    df, dropped = no_robots_examples(raw)
    assert df["example_id"].tolist() == ["no_robots_a"] and dropped["system_prompt"] == 1
    assert df.iloc[0]["prompt"] == [{"role": "user", "content": "Write a haiku"}] and df.iloc[0]["reply"] == "Rain on tin roofs"


def _msg(mid, parent, role, text, rank=np.nan, lang="en"):
    return {"message_id": mid, "parent_id": parent, "role": role, "text": text, "rank": rank, "lang": lang,
            "deleted": False, "review_result": True, "synthetic": False}


def test_oasst2_builds_thread_from_root():
    raw = pd.DataFrame([
        _msg("q1", np.nan, "prompter", "Tell me a joke"),                 # root: parent_id is NaN
        _msg("a1", "q1", "assistant", "Why did the chicken...", rank=0),
        _msg("a2", "q1", "assistant", "No.", rank=1),                      # not best-ranked
        _msg("q2", "a1", "prompter", "Another one"),
        _msg("a3", "q2", "assistant", "Knock knock", rank=0),
        _msg("q3", np.nan, "prompter", "Hola", lang="es"),
        _msg("a4", "q3", "assistant", "Hola!", rank=0, lang="es"),         # not English
    ])
    df, _ = oasst2_examples(raw)
    assert sorted(df["example_id"]) == ["oasst2_a1", "oasst2_a3"]
    thread = df.set_index("example_id").loc["oasst2_a3", "prompt"]
    assert [m["role"] for m in thread] == ["user", "assistant", "user"] and thread[0]["content"] == "Tell me a joke"


def test_filter_drops_refusals_and_eval_prompts():
    df = pd.DataFrame({"prompt": [[{"role": "user", "content": "Write a poem"}], [{"role": "user", "content": "Eval  Prompt"}],
                                  [{"role": "user", "content": "x"}]],
                       "reply": ["Roses...", "answer", "I'm sorry, I can't help with that."]})
    kept, dropped = filter_examples(df, frozenset({"eval prompt"}))
    assert kept["reply"].tolist() == ["Roses..."] and dropped == {"refusal_like": 1, "eval_prompt_overlap": 1}
