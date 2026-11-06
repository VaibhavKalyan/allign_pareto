"""
Text normalization, empty-prompt removal, and benchmark leakage filtering (spec §12).
"""
import re
from typing import Iterable, Optional, Tuple
import pandas as pd

from src.data.dedup import compute_sha256


HTML_TAG_RE = re.compile(r"<[^>]+>")
MD_HEADER_RE = re.compile(r"^#+\s*(Instruction|Prompt|User):?\s*", re.IGNORECASE | re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_prompt_text(text: str) -> str:
    """Clean HTML tags, markdown prompt headers, and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    cleaned = HTML_TAG_RE.sub(" ", text)
    cleaned = MD_HEADER_RE.sub("", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def clean_dataset(
    df: pd.DataFrame,
    min_char_len: int = 15,
    benchmark_prompts: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize text, remove empty/short prompts, and remove any prompt that also appears in a held-out
    benchmark (e.g. WildGuardTest), so classifier training never sees evaluation prompts (spec §13).
    """
    initial_count = len(df)
    out = df.copy()
    out["prompt"] = out["prompt"].apply(normalize_prompt_text)

    valid_len_mask = out["prompt"].str.len() >= min_char_len
    removed_empty = int((~valid_len_mask).sum())
    out = out[valid_len_mask].reset_index(drop=True)

    removed_leaks = 0
    if benchmark_prompts is not None:
        bench_hashes = {compute_sha256(normalize_prompt_text(p)) for p in benchmark_prompts}
        leak_mask = out["prompt"].apply(compute_sha256).isin(bench_hashes)
        removed_leaks = int(leak_mask.sum())
        out = out[~leak_mask].reset_index(drop=True)

    stats = {
        "initial_rows": initial_count,
        "removed_empty_or_short": removed_empty,
        "removed_benchmark_overlap": removed_leaks,
        "remaining_after_cleaning": len(out),
    }
    return out, stats
