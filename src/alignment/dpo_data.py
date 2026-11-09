"""
DPO preference pairs for ALIGN-PARETO (M2, spec §18).
Loads Anthropic/hh-rlhf transcripts and turns each pair into (prompt messages, chosen reply, rejected reply).
The prompt is the shared conversation up to the last assistant turn; the replies are that last turn.
No fallback dataset: if hh-rlhf cannot be loaded the pipeline stops (spec §29.12).
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

HH_REPO = "Anthropic/hh-rlhf"
HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"
_TURN = re.compile(r"\n\n(Human|Assistant):")


def parse_transcript(text: str) -> List[Dict[str, str]]:
    """'\\n\\nHuman: a\\n\\nAssistant: b' -> [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}]."""
    parts = _TURN.split(text)
    messages = []
    for speaker, content in zip(parts[1::2], parts[2::2]):
        messages.append({"role": "user" if speaker == "Human" else "assistant", "content": content.strip()})
    return messages


def split_pair(chosen: str, rejected: str) -> Optional[Tuple[List[Dict[str, str]], str, str]]:
    """Shared prompt messages + the two final assistant replies, or None if the pair is unusable."""
    cut = chosen.rfind(ASSISTANT)
    if cut < 0:
        return None
    prefix = chosen[: cut + len(ASSISTANT)]
    if not rejected.startswith(prefix):
        return None
    prompt = parse_transcript(chosen[:cut])
    chosen_reply, rejected_reply = chosen[len(prefix):].strip(), rejected[len(prefix):].strip()
    if not prompt or prompt[0]["role"] != "user" or prompt[-1]["role"] != "user":
        return None
    if any(m["role"] == prev["role"] for prev, m in zip(prompt, prompt[1:])):
        return None
    if not chosen_reply or not rejected_reply or chosen_reply == rejected_reply:
        return None
    return prompt, chosen_reply, rejected_reply


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def to_pairs(raw: pd.DataFrame, subset: str, excluded_prompts: frozenset = frozenset()) -> Tuple[pd.DataFrame, Dict]:
    """Parse raw hh-rlhf rows; drop unusable pairs and pairs whose first user turn is an evaluation prompt."""
    rows, dropped = [], {"unparseable": 0, "eval_prompt_overlap": 0}
    for i, (c, r) in enumerate(zip(raw["chosen"], raw["rejected"])):
        parsed = split_pair(c, r)
        if parsed is None:
            dropped["unparseable"] += 1
            continue
        prompt, chosen_reply, rejected_reply = parsed
        if _normalize(prompt[0]["content"]) in excluded_prompts:
            dropped["eval_prompt_overlap"] += 1
            continue
        rows.append({"pair_id": f"{subset}_{i:06d}", "subset": subset, "prompt": prompt,
                     "chosen": chosen_reply, "rejected": rejected_reply, "n_turns": len(prompt)})
    return pd.DataFrame(rows), dropped


def load_preference_pairs(cfg: Dict, raw_dir: Path, eval_prompts_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Return (train_pairs, eval_pairs, provenance) sampled per hh-rlhf subset as set in configs/alignment.yaml."""
    from huggingface_hub import hf_hub_download
    from src.data.download import _file_version

    dcfg = cfg["dpo"]
    seed = int(cfg.get("seed", 42))
    excluded = frozenset()
    if eval_prompts_path is not None and Path(eval_prompts_path).exists():
        excluded = frozenset(_normalize(p) for p in pd.read_csv(eval_prompts_path)["prompt"])

    out, meta = {"train": [], "test": []}, {"dataset": HH_REPO, "files": {}, "dropped": {}}
    for split, key in [("train", "train_pairs"), ("test", "eval_pairs")]:
        for subset, n in dcfg[key].items():
            filename = f"{subset}/{split}.jsonl.gz"
            try:
                path = Path(hf_hub_download(HH_REPO, filename, repo_type="dataset", cache_dir=str(Path(raw_dir) / "hf_cache")))
            except Exception as e:
                raise RuntimeError(f"Could not download {HH_REPO}/{filename}: {type(e).__name__}: {e}. Is internet enabled?") from e
            pairs, dropped = to_pairs(pd.read_json(path, lines=True), subset, excluded)
            if len(pairs) < n:
                raise ValueError(f"{filename}: only {len(pairs)} usable pairs, config asks for {n}")
            out[split].append(pairs.sample(n=int(n), random_state=seed))
            meta["files"][filename] = {"usable_pairs": len(pairs), "sampled": int(n), **_file_version(path)}
            meta["dropped"][filename] = dropped
    train = pd.concat(out["train"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    evals = pd.concat(out["test"]).reset_index(drop=True)
    return train, evals, meta
