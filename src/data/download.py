"""
Raw dataset download for ALIGN-PARETO (M1).
Fetches the published files of Databricks Dolly-15k and AllenAI WildGuardMix (train + test)
from the Hugging Face Hub, and the XSTest prompts (external evaluation only) from GitHub.
There is no synthetic fallback: if a dataset cannot be loaded the pipeline stops with an
explanation (spec §8, §29.12).
"""
import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd


DOLLY_REPO = "databricks/databricks-dolly-15k"
DOLLY_FILE = "databricks-dolly-15k.jsonl"
WILDGUARD_REPO = "allenai/wildguardmix"
WILDGUARD_TRAIN_FILE = "train/wildguard_train.parquet"
WILDGUARD_TEST_FILE = "test/wildguard_test.parquet"
# XSTest (CC BY 4.0): 250 safe prompts that sound unsafe + 200 unsafe contrasts. Evaluation only.
XSTEST_FILE = "xstest_prompts.csv"
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"


def resolve_hf_token() -> Optional[str]:
    """HF token from the environment, a Kaggle secret, or the local `hf auth login` cache (None)."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def _local_copy(filename: str) -> Optional[Path]:
    """
    A pre-downloaded copy of a published file, e.g. the private Kaggle dataset `align-pareto-raw`
    (Kaggle API-pushed notebooks cannot read secrets, so the gated WildGuardMix files are attached there).
    """
    roots = [os.environ.get("ALIGN_PARETO_RAW_DIR"), "/kaggle/input"]
    name = Path(filename).name
    for root in filter(None, roots):
        hits = sorted(Path(root).rglob(name)) if Path(root).exists() else []
        if hits:
            return hits[0]
    return None


def _fetch(repo_id: str, filename: str, cache_dir: Path, token: Optional[str]) -> Path:
    local = _local_copy(filename)
    if local is not None:
        return local
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(repo_id, filename, repo_type="dataset", cache_dir=str(cache_dir), token=token))
    except Exception as e:
        raise RuntimeError(
            f"Could not download {repo_id}/{filename}: {type(e).__name__}: {e}\n"
            "Check that: (1) the Kaggle dataset 'align-pareto-raw' is attached, or internet is enabled and (2) an HF token is available (HF_TOKEN env var, "
            "Kaggle secret 'HF_TOKEN', or `hf auth login`), and (3) the account has accepted the "
            "WildGuardMix terms at https://huggingface.co/datasets/allenai/wildguardmix"
        ) from e


def _file_version(path: Path) -> Dict[str, Optional[str]]:
    """Dataset version info (spec §31): file SHA-256, plus the Hub commit when the file came from the HF cache."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    parts = path.parts
    revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    return {"sha256": h.hexdigest(), "hf_revision": revision}


def load_raw_datasets(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Return (dolly, wildguard_train, wildguard_test) exactly as published, plus provenance metadata."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = raw_dir / "hf_cache"
    token = resolve_hf_token()

    dolly_path = _fetch(DOLLY_REPO, DOLLY_FILE, cache_dir, token)
    wg_train_path = _fetch(WILDGUARD_REPO, WILDGUARD_TRAIN_FILE, cache_dir, token)
    wg_test_path = _fetch(WILDGUARD_REPO, WILDGUARD_TEST_FILE, cache_dir, token)
    dolly = pd.read_json(dolly_path, lines=True)
    wg_train = pd.read_parquet(wg_train_path)
    wg_test = pd.read_parquet(wg_test_path)

    for name, df, cols in [
        ("Dolly-15k", dolly, {"instruction", "context", "category"}),
        ("WildGuardMix train", wg_train, {"prompt", "adversarial", "prompt_harm_label", "subcategory"}),
        ("WildGuardMix test", wg_test, {"prompt", "prompt_harm_label"}),
    ]:
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing expected columns {sorted(missing)}; the published schema changed.")

    meta = {
        "source_mode": "local_copy" if _local_copy(DOLLY_FILE) else "huggingface_hub",
        "dolly": {"repo": DOLLY_REPO, "file": DOLLY_FILE, "rows": len(dolly), **_file_version(dolly_path)},
        "wildguard_train": {"repo": WILDGUARD_REPO, "file": WILDGUARD_TRAIN_FILE, "rows": len(wg_train), **_file_version(wg_train_path)},
        "wildguard_test": {"repo": WILDGUARD_REPO, "file": WILDGUARD_TEST_FILE, "rows": len(wg_test), **_file_version(wg_test_path)},
    }
    return dolly, wg_train, wg_test, meta


def load_xstest(raw_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """XSTest prompts with `safety_label` (unsafe -> 1, safe -> 0). Local copy first, else GitHub."""
    path = _local_copy(XSTEST_FILE)
    source = "local_copy"
    if path is None:
        import urllib.request
        path = Path(raw_dir) / XSTEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(XSTEST_URL, path)
        except Exception as e:
            raise RuntimeError(
                f"Could not download XSTest from {XSTEST_URL}: {type(e).__name__}: {e}\n"
                "Attach a dataset containing xstest_prompts.csv or enable internet."
            ) from e
        source = XSTEST_URL
    df = pd.read_csv(path)
    missing = {"prompt", "type", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"XSTest is missing expected columns {sorted(missing)}; the published schema changed.")
    labels = df["label"].astype(str).str.strip().str.lower()
    if not labels.isin(["safe", "unsafe"]).all():
        raise ValueError(f"Unexpected XSTest labels: {sorted(labels.unique())}")
    df = df.assign(safety_label=(labels == "unsafe").astype(int))
    meta = {"source": source, "file": XSTEST_FILE, "rows": len(df), **_file_version(path)}
    return df, meta


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    dolly, wg_train, wg_test, meta = load_raw_datasets(root / "data/raw")
    print(meta)
