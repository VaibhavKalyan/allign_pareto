"""
SFT data for M2 stage 1 (spec §18): human-written helpful replies, used to turn Qwen3-0.6B-Base into a chat model
that answers and writes (lambda = 0) without any safety training.
  HuggingFaceH4/no_robots   9.5k replies written by paid annotators (stories, poems, brainstorming, QA, ...)
  OpenAssistant/oasst2      English conversations; only the reply ranked best by human reviewers is used
Replies that look like refusals are dropped, so lambda = 0 is not already partly aligned and M3 measures the full
effect of the DPO safety stage. No fallback dataset: if a source cannot be loaded the pipeline stops.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

NO_ROBOTS_REPO = "HuggingFaceH4/no_robots"
NO_ROBOTS_FILES = {"train": "data/train-00000-of-00001.parquet", "test": "data/test-00000-of-00001.parquet"}
OASST2_REPO = "OpenAssistant/oasst2"
OASST2_FILES = {"train": "data/train-00000-of-00001-88ba0162028a73fc.parquet",
                "test": "data/validation-00000-of-00001-1deeef95c3248fe0.parquet"}

REFUSAL = re.compile(
    r"\b(I(?:'m|’m| am) (?:sorry|unable|not able)|I can(?:not|'t|’t) (?:help|assist|provide|do that|comply)"
    r"|as an AI|I(?:'m|’m| am) just an AI|I cannot (?:help|assist|provide))",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_refusal(text: str) -> bool:
    return bool(REFUSAL.search(text))


def no_robots_examples(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """(prompt messages, reply) from no_robots; examples with a system prompt are dropped (eval uses none)."""
    rows, dropped = [], {"system_prompt": 0, "malformed": 0}
    for pid, messages in zip(raw["prompt_id"], raw["messages"]):
        messages = [{"role": m["role"], "content": m["content"].strip()} for m in messages]
        if messages[0]["role"] == "system":
            dropped["system_prompt"] += 1
            continue
        prompt, reply = messages[:-1], messages[-1]
        if not prompt or reply["role"] != "assistant" or prompt[-1]["role"] != "user" or not reply["content"]:
            dropped["malformed"] += 1
            continue
        rows.append({"example_id": f"no_robots_{pid}", "source": "no_robots", "prompt": prompt, "reply": reply["content"]})
    return pd.DataFrame(rows), dropped


def oasst2_examples(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    One example per best-ranked (rank 0) English assistant reply: the prompt is the conversation from the tree root
    down to that reply. Every message on the path must be English, reviewed, not deleted and human-written.
    """
    ok = (raw["lang"] == "en") & ~raw["deleted"] & raw["review_result"].fillna(False).astype(bool) & ~raw["synthetic"]
    by_id = raw[ok].set_index("message_id")[["parent_id", "text", "role"]].to_dict("index")
    rows, dropped = [], {"broken_thread": 0}
    targets = raw[ok & (raw["role"] == "assistant") & (raw["rank"] == 0)]
    for mid, text in zip(targets["message_id"], targets["text"]):
        chain, cur = [], by_id[mid]["parent_id"]
        while not pd.isna(cur):                     # a thread's root has parent_id NaN
            node = by_id.get(cur)
            if node is None:
                chain = None
                break
            chain.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"].strip()})
            cur = node["parent_id"]
        if not chain:
            dropped["broken_thread"] += 1
            continue
        prompt = chain[::-1]
        if prompt[0]["role"] != "user" or prompt[-1]["role"] != "user" or any(a["role"] == b["role"] for a, b in zip(prompt, prompt[1:])):
            dropped["broken_thread"] += 1
            continue
        rows.append({"example_id": f"oasst2_{mid}", "source": "oasst2", "prompt": prompt, "reply": text.strip()})
    return pd.DataFrame(rows), dropped


def filter_examples(df: pd.DataFrame, excluded_prompts: frozenset) -> Tuple[pd.DataFrame, Dict]:
    """Drop refusal-like replies, replies inside earlier assistant turns that refuse, and evaluation prompts."""
    refuses = df["reply"].map(is_refusal) | df["prompt"].map(lambda p: any(m["role"] == "assistant" and is_refusal(m["content"]) for m in p))
    in_eval = df["prompt"].map(lambda p: _normalize(p[0]["content"]) in excluded_prompts)
    dropped = {"refusal_like": int(refuses.sum()), "eval_prompt_overlap": int((in_eval & ~refuses).sum())}
    return df[~refuses & ~in_eval].reset_index(drop=True), dropped


LOADERS = {
    "no_robots": (NO_ROBOTS_REPO, NO_ROBOTS_FILES, no_robots_examples),
    "oasst2": (OASST2_REPO, OASST2_FILES, oasst2_examples),
}


def load_sft_examples(counts: Dict[str, int], split: str, raw_dir: Path, excluded: frozenset, seed: int, meta: Dict) -> pd.DataFrame:
    """Up to `counts[source]` examples per source from its `split` ("train" or "test"); provenance goes into `meta`."""
    from huggingface_hub import hf_hub_download
    from src.data.download import _file_version

    parts = []
    for source, n in counts.items():
        repo, files, build = LOADERS[source]
        try:
            path = Path(hf_hub_download(repo, files[split], repo_type="dataset", cache_dir=str(Path(raw_dir) / "hf_cache")))
        except Exception as e:
            raise RuntimeError(f"Could not download {repo}/{files[split]}: {type(e).__name__}: {e}. Is internet enabled?") from e
        examples, dropped = build(pd.read_parquet(path))
        examples, filtered = filter_examples(examples, excluded)
        take = min(int(n), len(examples))
        parts.append(examples.sample(n=take, random_state=seed))
        meta.setdefault("files", {})[f"{repo}/{files[split]}"] = {"usable": len(examples), "sampled": take, **_file_version(path)}
        meta.setdefault("dropped", {})[f"{repo}/{files[split]}"] = {**dropped, **filtered}
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
