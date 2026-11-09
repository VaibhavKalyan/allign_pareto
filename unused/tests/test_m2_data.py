"""Unit tests for the M2 hh-rlhf pair parsing. Run: `pytest tests`."""
import pandas as pd

from src.alignment.dpo_data import parse_transcript, split_pair, to_pairs

PROMPT = "\n\nHuman: hi there\n\nAssistant: hello\n\nHuman: tell me a joke\n\nAssistant:"


def test_parse_transcript_roles_and_content():
    assert parse_transcript("\n\nHuman: a b\n\nAssistant: c") == [
        {"role": "user", "content": "a b"}, {"role": "assistant", "content": "c"},
    ]


def test_split_pair_shared_prompt_and_last_replies():
    prompt, chosen, rejected = split_pair(PROMPT + " a good joke", PROMPT + " a bad joke")
    assert [m["role"] for m in prompt] == ["user", "assistant", "user"]
    assert prompt[-1]["content"] == "tell me a joke"
    assert (chosen, rejected) == ("a good joke", "a bad joke")


def test_split_pair_rejects_unusable_pairs():
    assert split_pair(PROMPT + " same", PROMPT + " same") is None           # identical replies
    assert split_pair(PROMPT + " x", "\n\nHuman: other\n\nAssistant: y") is None  # different conversations
    assert split_pair(PROMPT + " ", PROMPT + " y") is None                  # empty reply
    assert split_pair("no turns at all", "no turns") is None


def test_split_pair_assistant_text_containing_marker():
    # only the last "Assistant:" marker separates the replies
    prompt, chosen, _ = split_pair(PROMPT + " ok", PROMPT + " no")
    assert len(prompt) == 3 and chosen == "ok"


def test_to_pairs_drops_eval_prompt_overlap():
    raw = pd.DataFrame({"chosen": [PROMPT + " good", "\n\nHuman: Tell  me a JOKE\n\nAssistant: x"],
                        "rejected": [PROMPT + " bad", "\n\nHuman: Tell  me a JOKE\n\nAssistant: y"]})
    pairs, dropped = to_pairs(raw, "harmless-base", frozenset({"tell me a joke"}))
    assert len(pairs) == 1 and dropped["eval_prompt_overlap"] == 1
    assert pairs.iloc[0]["pair_id"] == "harmless-base_000000"
