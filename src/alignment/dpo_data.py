"""
hh-rlhf data for both M2 stages (spec §18): SFT examples and DPO preference pairs.
Loads Anthropic/hh-rlhf transcripts and turns each pair into (prompt messages, chosen reply, rejected reply).
The prompt is the shared conversation up to the last assistant turn; the replies are that last turn.
SFT uses (prompt, chosen reply) only.
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


def eval_prompt_set(eval_prompts_path: Optional[Path]) -> frozenset:
    """Normalized evaluation prompts; hh-rlhf conversations that start with one are never trained on."""
    if eval_prompts_path is None or not Path(eval_prompts_path).exists():
        return frozenset()
    return frozenset(_normalize(p) for p in pd.read_csv(eval_prompts_path)["prompt"])


def load_hh_pairs(counts: Dict[str, int], split: str, raw_dir: Path, excluded: frozenset, seed: int, meta: Dict) -> pd.DataFrame:
    """Sample `counts[subset]` usable pairs from each hh-rlhf `<subset>/<split>.jsonl.gz`; provenance goes into `meta`."""
    from huggingface_hub import hf_hub_download
    from src.data.download import _file_version

    parts = []
    for subset, n in counts.items():
        filename = f"{subset}/{split}.jsonl.gz"
        try:
            path = Path(hf_hub_download(HH_REPO, filename, repo_type="dataset", cache_dir=str(Path(raw_dir) / "hf_cache")))
        except Exception as e:
            raise RuntimeError(f"Could not download {HH_REPO}/{filename}: {type(e).__name__}: {e}. Is internet enabled?") from e
        pairs, dropped = to_pairs(pd.read_json(path, lines=True), subset, excluded)
        if len(pairs) < int(n):
            raise ValueError(f"{filename}: only {len(pairs)} usable pairs, config asks for {n}")
        parts.append(pairs.sample(n=int(n), random_state=seed))
        meta.setdefault("files", {})[filename] = {"usable_pairs": len(pairs), **_file_version(path)}
        meta.setdefault("dropped", {})[filename] = dropped
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_stage_data(cfg: Dict, raw_dir: Path, eval_prompts_path: Optional[Path] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    hh-rlhf data for both M2 stages, as set in configs/alignment.yaml:
      sft_train / sft_eval: chosen replies only (helpful-base: helpful, not safety-trained)
      dpo_train / dpo_eval: chosen vs rejected pairs (harmless-base: the safety alignment)
    Each SFT example and DPO pair is a different conversation (different subsets, or disjoint samples of one).
    """
    seed = int(cfg.get("seed", 42))
    excluded = eval_prompt_set(eval_prompts_path)
    meta = {"dataset": HH_REPO, "excluded_eval_prompts": len(excluded)}
    data = {}
    for stage, (train_key, eval_key) in {"sft": ("train_examples", "eval_examples"), "dpo": ("train_pairs", "eval_pairs")}.items():
        scfg = cfg[stage]
        data[f"{stage}_train"] = load_hh_pairs(scfg[train_key], "train", raw_dir, excluded, seed, meta)
        data[f"{stage}_eval"] = load_hh_pairs(scfg[eval_key], "test", raw_dir, excluded, seed + 1, meta)
    for split in ("train", "eval"):
        overlap = set(data[f"sft_{split}"]["pair_id"]) & set(data[f"dpo_{split}"]["pair_id"])
        if overlap:
            raise ValueError(f"{len(overlap)} hh-rlhf conversations are in both the SFT and DPO {split} data")
    return data, meta
