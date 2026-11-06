"""
Exact SHA-256 deduplication, MinHash LSH near-duplicate removal, prompt-family grouping, and
leakage-safe family splits (spec §12-13).
"""
import hashlib
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash of normalized lowercase prompt."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def exact_deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove exact duplicate prompts via SHA-256 hash."""
    out = df.copy()
    out["prompt_sha256"] = out["prompt"].apply(compute_sha256)
    before = len(out)
    out = out.drop_duplicates(subset=["prompt_sha256"], keep="first").reset_index(drop=True)
    removed = before - len(out)
    return out, removed


def _word_shingles(text: str, k: int = 3) -> Set[str]:
    tokens = text.lower().split()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _minhashes(prompts: List[str], k: int, num_perm: int, seed: int):
    from datasketch import MinHash
    out = []
    for p in prompts:
        m = MinHash(num_perm=num_perm, seed=seed)
        m.update_batch([s.encode("utf-8") for s in _word_shingles(p, k=k)])
        out.append(m)
    return out


def _similar_pairs(hashes, threshold: float, num_perm: int) -> List[Tuple[int, int]]:
    """Index pairs (i < j) whose estimated Jaccard similarity is at least `threshold`."""
    from datasketch import MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    pairs = []
    for i, m in enumerate(hashes):
        for j in lsh.query(m):
            if hashes[j].jaccard(m) >= threshold:
                pairs.append((j, i))
        lsh.insert(i, m)
    return pairs


def near_deduplicate_minhash(
    df: pd.DataFrame, threshold: float = 0.9, k: int = 3, num_perm: int = 128, seed: int = 42
) -> Tuple[pd.DataFrame, int]:
    """Remove prompts whose k-word-shingle Jaccard similarity to an earlier prompt is >= threshold."""
    hashes = _minhashes(df["prompt"].tolist(), k, num_perm, seed)
    drop = {i for _, i in _similar_pairs(hashes, threshold, num_perm)}
    keep = [i not in drop for i in range(len(df))]
    return df[keep].reset_index(drop=True), len(drop)


def assign_prompt_families(
    df: pd.DataFrame, threshold: float = 0.5, k: int = 3, num_perm: int = 128, seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Group related prompt variants (Jaccard >= threshold, transitively) into one `family_id`, so that
    templated or paraphrased variants never end up on both sides of a split.
    """
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    hashes = _minhashes(df["prompt"].tolist(), k, num_perm, seed)
    for a, b in _similar_pairs(hashes, threshold, num_perm):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = [find(i) for i in range(n)]
    out = df.copy()
    out["family_id"] = [f"fam_{r:06d}" for r in roots]
    sizes = out["family_id"].value_counts()
    return out, {
        "family_threshold": threshold,
        "n_families": int(len(sizes)),
        "n_multi_prompt_families": int((sizes > 1).sum()),
        "largest_family_size": int(sizes.max()) if n else 0,
    }


def leakage_safe_family_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Split dataset into train/validation/test by prompt family_id stratified across
    the 4 label quadrants so related prompt families never cross splits.
    Verifies zero SHA-256 hash overlap across splits.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["quadrant"] = out["creative_label"].astype(str) + "_" + out["safety_label"].astype(str)
    out["split"] = "train"

    # Group families by their primary quadrant so each family_id is assigned to exactly one split
    fam_quad = out.groupby("family_id")["quadrant"].agg(lambda s: s.iloc[0]).reset_index()
    fam_to_split = {}
    for quad, fgroup in fam_quad.groupby("quadrant"):
        families = fgroup["family_id"].tolist()
        rng.shuffle(families)
        n_fam = len(families)
        n_train = int(round(n_fam * train_ratio))
        n_val = max(1, int(round(n_fam * val_ratio)))
        for f in families[:n_train]:
            fam_to_split[f] = "train"
        for f in families[n_train : n_train + n_val]:
            fam_to_split[f] = "validation"
        for f in families[n_train + n_val :]:
            fam_to_split[f] = "test"
    out["split"] = out["family_id"].map(fam_to_split)

    # Verify zero exact hash leakage and zero family leakage across splits
    train_hashes = set(out[out["split"] == "train"]["prompt_sha256"])
    val_hashes = set(out[out["split"] == "validation"]["prompt_sha256"])
    test_hashes = set(out[out["split"] == "test"]["prompt_sha256"])

    train_fams = set(out[out["split"] == "train"]["family_id"])
    val_fams = set(out[out["split"] == "validation"]["family_id"])
    test_fams = set(out[out["split"] == "test"]["family_id"])

    hash_leak_count = len(train_hashes & val_hashes) + len(train_hashes & test_hashes) + len(val_hashes & test_hashes)
    fam_leak_count = len(train_fams & val_fams) + len(train_fams & test_fams) + len(val_fams & test_fams)

    assert hash_leak_count == 0, f"CRITICAL: Found {hash_leak_count} SHA-256 hash leaks across splits!"
    assert fam_leak_count == 0, f"CRITICAL: Found {fam_leak_count} family_id leaks across splits!"

    split_summary = {}
    for sp in ["train", "validation", "test"]:
        sub = out[out["split"] == sp]
        split_summary[sp] = {
            "total": int(len(sub)),
            "creative0_safety0": int(((sub["creative_label"] == 0) & (sub["safety_label"] == 0)).sum()),
            "creative1_safety0": int(((sub["creative_label"] == 1) & (sub["safety_label"] == 0)).sum()),
            "creative0_safety1": int(((sub["creative_label"] == 0) & (sub["safety_label"] == 1)).sum()),
            "creative1_safety1": int(((sub["creative_label"] == 1) & (sub["safety_label"] == 1)).sum()),
        }

    return out, {
        "splits": split_summary,
        "exact_hash_leakage_count": hash_leak_count,
        "family_cluster_leakage_count": fam_leak_count,
    }
