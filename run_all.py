"""
Master Single-Command Orchestrator for ALIGN-PARETO (M1 -> M2 -> M3 -> M4).
Usage:
  python3 run_all.py               # Run full end-to-end pipeline (M1 to M4)
  python3 run_all.py --verify-only # Check that all required input files exist before running
"""
import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Grows milestone by milestone: add each M2-M4 file here once it is written.
REQUIRED_PRE_RUN_FILES = [
    "requirements.txt",
    "configs/classifier.yaml",
    "src/__init__.py",
    "src/utils/seed.py",
    "src/utils/logging.py",
    "src/utils/io.py",
    "src/data/download.py",
    "src/data/mapping.py",
    "src/data/cleaning.py",
    "src/data/dedup.py",
    "src/classifier/train.py",
    "src/classifier/predict.py",
    "src/classifier/evaluate.py",
    "src/run_m1.py",
    "configs/alignment.yaml",
    "data/eval/eval_prompts.csv",
    "src/alignment/sft_data.py",
    "src/alignment/models.py",
    "src/alignment/train_sft.py",
    "src/run_m2.py",
    "configs/evaluation.yaml",
    "src/alignment/interpolate.py",
    "src/evaluation/generate.py",
    "src/evaluation/judge.py",
    "src/evaluation/metrics.py",
    "src/run_m3.py",
]


def verify_pre_run_files(root: Path = ROOT_DIR) -> bool:
    missing = [rel for rel in REQUIRED_PRE_RUN_FILES if not (root / rel).exists()]
    if missing:
        print(f"[ERROR] Missing {len(missing)} required pre-run files:")
        for m in missing:
            print(f"  - {m}")
        return False
    print(f"[OK] All {len(REQUIRED_PRE_RUN_FILES)} mandatory pre-run source & config files verified in {root}")
    return True


def main():
    parser = argparse.ArgumentParser(description="ALIGN-PARETO End-to-End Orchestrator")
    parser.add_argument("--verify-only", action="store_true", help="Only verify required files without running")
    args = parser.parse_args()

    if not verify_pre_run_files(ROOT_DIR):
        sys.exit(1)

    if args.verify_only:
        return

    t0 = time.time()
    from src.run_m1 import run_m1

    print("\n=== Phase 1: Running Milestone 1 (Dataset & Dual-Head Classifier) ===")
    m1_res = run_m1(ROOT_DIR)
    print(f"M1 Macro-F1: {m1_res['aggregate']['macro_f1']} | Exact-Match: {m1_res['aggregate']['exact_match_ratio']}")
    elapsed = time.time() - t0
    print(f"\n[SUCCESS] Pipeline completed in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
