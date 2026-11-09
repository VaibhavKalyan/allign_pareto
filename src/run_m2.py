"""
Milestone 2 (M2): build the lambda-controllable aligned model from the unaligned Qwen3-0.6B-Base.
  Stage 1 (SFT): helpful-only chat model, human-written replies (no_robots, oasst2) -> models/m2/sft_adapter (lambda = 0)
  Stage 2 (DPO): safety alignment, hh-rlhf preference pairs                         -> models/m2/dpo_adapter (lambda = 1)
Run from project root: `python3 src/run_m2.py` (`--stages sft` trains and checks lambda = 0 only; `--quick` for a
tiny smoke test). Checks: losses fall, adapters reload to the trained model, lambda = 0 is exactly the SFT model,
and replies of the raw base model, lambda = 0 and lambda = 1 on evaluation prompts are saved for reading.
"""
import json
import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.seed import environment_info, set_global_seed
from src.utils.logging import StructuredLogger
from src.utils.io import load_yaml, save_json, save_parquet
from src.alignment.adapters import set_peft_lora_lambda
from src.alignment.dpo_data import load_stage_data
from src.alignment.models import (
    generate, has_cuda, last_checkpoint, last_token_logits, load_aligned_model, load_base_model, load_sft_model,
    load_tokenizer, render_prompt, save_adapter,
)
from src.alignment.train_sft import build_sft_dataset, make_sft_trainer
from src.alignment.train_dpo import build_dpo_dataset, make_dpo_trainer, pairs_to_parquet_frame

PROBE_PROMPT = "How do I pick a lock?"
STAGES = ("sft", "dpo")


def plot_training(log: pd.DataFrame, path: Path, title: str, second: tuple) -> None:
    """Loss (train + eval) and one more logged metric, e.g. DPO reward accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cols, ylabel in [(axes[0], ("loss", "eval_loss"), "loss"), (axes[1], second[:2], second[2])]:
        for col, label in zip(cols, ("train", "eval")):
            if col in log:
                d = log.dropna(subset=[col])
                ax.plot(d["step"], d[col], marker="o", ms=3, label=label)
        ax.set(xlabel="step", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{title} loss")
    axes[1].set_title(f"{title} {second[2]}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _first_last(log: pd.DataFrame, col: str, k: int = 3) -> dict:
    """Mean of the first and last k logged values (single log points are noisy)."""
    vals = log[col].dropna() if col in log else pd.Series(dtype=float)
    if vals.empty:
        return {"first": None, "last": None}
    return {"first": round(float(vals.head(k).mean()), 4), "last": round(float(vals.tail(k).mean()), 4)}


def _rounded(d: dict) -> dict:
    return {k: round(float(v), 4) for k, v in d.items() if isinstance(v, (int, float))}


def _max_diff(a, b) -> float:
    return round(float((a - b).abs().max()), 6)


def run_stage(name, make_trainer, model, tokenizer, train_ds, eval_ds, cfg, models_dir, results_dir, adapter_dir, max_steps, logger):
    """Train one stage, save its adapter, log and final eval. Returns (trainer, log, final_eval)."""
    t0 = time.time()
    trainer = make_trainer(model, tokenizer, train_ds, eval_ds, cfg, models_dir / f"{name}_checkpoints", max_steps=max_steps)
    trainer.train(resume_from_checkpoint=last_checkpoint(models_dir / f"{name}_checkpoints"))
    final_eval = trainer.evaluate()
    save_adapter(trainer, tokenizer, adapter_dir)
    log = pd.DataFrame(trainer.state.log_history)
    log.to_csv(results_dir / f"{name}_train_log.csv", index=False)
    save_json({"minutes": round((time.time() - t0) / 60, 1), **final_eval}, results_dir / f"{name}_final_eval.json")
    logger.info(f"{name.upper()} finished", minutes=round((time.time() - t0) / 60, 1), eval_loss=final_eval.get("eval_loss"))
    return trainer, log, final_eval


def load_stage_results(name, results_dir):
    log = pd.read_csv(results_dir / f"{name}_train_log.csv")
    final_eval = json.loads((results_dir / f"{name}_final_eval.json").read_text())
    return log, final_eval


def apply_quick_test(cfg: dict) -> int:
    """Shrink both stages to a few examples and steps; returns max_steps."""
    q = cfg["quick_test"]
    for stage_cfg, train_key, eval_key in [(cfg["sft"], "train_examples", "eval_examples"), (cfg["dpo"], "train_pairs", "eval_pairs")]:
        stage_cfg[train_key] = {k: min(int(v), int(q["train_examples"])) for k, v in stage_cfg[train_key].items()}
        stage_cfg[eval_key] = {k: min(int(v), int(q["eval_examples"])) for k, v in stage_cfg[eval_key].items()}
        stage_cfg.update(max_length=int(q["max_length"]), eval_steps=1, logging_steps=1)
    cfg["dpo"]["max_prompt_length"] = int(q["max_prompt_length"])
    return int(q["max_steps"])


def run_m2(root_dir: Path = Path("."), quick: bool = False, retrain: bool = False, stages=STAGES) -> dict:
    stages = tuple(s for s in STAGES if s in stages)
    cfg = load_yaml(root_dir / "configs/alignment.yaml")
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    mcfg, scfg, dcfg, smoke_cfg = cfg["model"], cfg["sft"], cfg["dpo"], cfg["smoke_test"]
    max_steps = apply_quick_test(cfg) if quick else None
    enable_thinking = bool(mcfg.get("enable_thinking", False))

    tag = "m2_quick" if quick else "m2"
    results_dir = root_dir / "results" / tag
    models_dir = root_dir / "models" / tag
    processed_dir = root_dir / "data/processed" / tag
    sft_dir, dpo_dir = models_dir / "sft_adapter", models_dir / "dpo_adapter"
    logger = StructuredLogger("m2_pipeline", results_dir / "execution.jsonl")
    logger.info("Starting M2 pipeline", seed=seed, quick=quick, stages=list(stages), cuda=has_cuda(), base_model=mcfg["base_model_name"])
    if not has_cuda() and not quick:
        raise RuntimeError("M2 needs a CUDA GPU (Kaggle: Settings -> Accelerator -> GPU T4). Use --quick for a CPU smoke test.")
    if "dpo" in stages and "sft" not in stages and not (sft_dir / "adapter_config.json").exists():
        raise RuntimeError(f"DPO starts from the SFT model, but {sft_dir} has no adapter: run the sft stage first.")

    # 1. Training data (evaluation prompts excluded so M3 never scores a trained-on prompt)
    eval_prompts_path = root_dir / "data/eval/eval_prompts.csv"
    data, data_meta = load_stage_data(cfg, root_dir / "data/raw", eval_prompts_path, stages)
    for name, df in data.items():
        save_parquet(pairs_to_parquet_frame(df), processed_dir / f"{name}.parquet")
    tokenizer = load_tokenizer(mcfg)
    datasets, used, dropped = {}, {}, {}
    for name, df in data.items():
        if name.startswith("sft"):
            datasets[name], dropped[name] = build_sft_dataset(df, tokenizer, int(scfg["max_length"]), enable_thinking)
        else:
            datasets[name], dropped[name] = build_dpo_dataset(df, tokenizer, int(dcfg["max_prompt_length"]), int(dcfg["max_length"]), enable_thinking)
        used[name] = len(datasets[name])
    data_meta.update(used=used, dropped_over_length=dropped)
    logger.info("Training data ready", used=used, dropped_over_length=dropped)

    metrics = {"quick_test": quick, "base_model": mcfg["base_model_name"], "stages": list(stages), "trained_this_run": {}}
    reload_check = {}
    probe = render_prompt([{"role": "user", "content": PROBE_PROMPT}], tokenizer, enable_thinking)

    # 2. Stage 1: SFT (skipped when an adapter is already saved)
    sft_trained = False
    if "sft" in stages:
        sft_trained = retrain or not (sft_dir / "adapter_config.json").exists()
        if sft_trained:
            trainer, sft_log, sft_eval_res = run_stage("sft", make_sft_trainer, load_base_model(mcfg), tokenizer, datasets["sft_train"],
                                                       datasets["sft_eval"], cfg, models_dir, results_dir, sft_dir, max_steps, logger)
            trainer.model.eval()
            logits_trained = last_token_logits(trainer.model, tokenizer, probe)
            del trainer
            reload_check["max_abs_logit_diff_sft_trained_vs_reloaded"] = _max_diff(logits_trained, last_token_logits(load_sft_model(mcfg, sft_dir), tokenizer, probe))
        else:
            sft_log, sft_eval_res = load_stage_results("sft", results_dir)
            logger.info("SFT adapter already saved; skipping SFT", adapter_dir=str(sft_dir))
        plot_training(sft_log, results_dir / "plots/sft_training.png", "SFT", ("mean_token_accuracy", "eval_mean_token_accuracy", "token accuracy"))
        metrics["trained_this_run"]["sft"] = sft_trained
        metrics["sft"] = {"train_steps": int(sft_log["step"].max()), "train_loss": _first_last(sft_log, "loss"),
                          "train_token_accuracy": _first_last(sft_log, "mean_token_accuracy"), "final_eval": _rounded(sft_eval_res)}

    # 3. Stage 2: DPO on the merged SFT model (reference policy = SFT model)
    if "dpo" in stages:
        dpo_trained = retrain or sft_trained or not (dpo_dir / "adapter_config.json").exists()
        if dpo_trained:
            trainer, dpo_log, dpo_eval_res = run_stage("dpo", make_dpo_trainer, load_sft_model(mcfg, sft_dir), tokenizer, datasets["dpo_train"],
                                                       datasets["dpo_eval"], cfg, models_dir, results_dir, dpo_dir, max_steps, logger)
            trainer.model.eval()
            logits_trained = last_token_logits(trainer.model, tokenizer, probe)
            del trainer
        else:
            dpo_log, dpo_eval_res = load_stage_results("dpo", results_dir)
            logger.info("DPO adapter already saved; skipping DPO", adapter_dir=str(dpo_dir))
        plot_training(dpo_log, results_dir / "plots/dpo_training.png", "DPO", ("rewards/accuracies", "eval_rewards/accuracies", "reward accuracy"))
        metrics["trained_this_run"]["dpo"] = dpo_trained
        metrics["dpo"] = {"train_steps": int(dpo_log["step"].max()), "train_loss": _first_last(dpo_log, "loss"),
                          "train_reward_accuracy": _first_last(dpo_log, "rewards/accuracies"),
                          "train_reward_margin": _first_last(dpo_log, "rewards/margins"), "final_eval": _rounded(dpo_eval_res)}

        # Reload check: the saved adapters reproduce the trained model, and lambda = 0 is exactly the SFT model
        aligned = load_aligned_model(mcfg, sft_dir, dpo_dir)
        set_peft_lora_lambda(aligned, 1.0)
        logits_l1 = last_token_logits(aligned, tokenizer, probe)
        if dpo_trained:
            reload_check["max_abs_logit_diff_dpo_trained_vs_reloaded"] = _max_diff(logits_trained, logits_l1)
        with aligned.disable_adapter():
            logits_sft = last_token_logits(aligned, tokenizer, probe)
        set_peft_lora_lambda(aligned, 0.0)
        reload_check["max_abs_logit_diff_lambda0_vs_sft"] = _max_diff(last_token_logits(aligned, tokenizer, probe), logits_sft)
        reload_check["max_abs_logit_diff_lambda1_vs_sft"] = _max_diff(logits_l1, logits_sft)
        if reload_check["max_abs_logit_diff_lambda0_vs_sft"] > 1e-4:
            raise AssertionError(f"lambda=0 does not reproduce the SFT model: {reload_check}")
    logger.info("Adapter reload check", **reload_check)
    # Small fp16-autocast differences only: the trained copy may still run in autocast, the reloaded one does not
    if any(v > 0.1 for k, v in reload_check.items() if "trained_vs_reloaded" in k):
        raise AssertionError(f"Reloaded adapters do not match the trained model: {reload_check}")
    metrics["reload_check"] = reload_check

    # 4. Smoke test: sampled replies (M3's settings, fixed seed) of the raw base model, lambda = 0 and lambda = 1
    ev = pd.read_csv(eval_prompts_path)
    ev = ev[ev["split"] == "validation"].groupby("quadrant", group_keys=False).head(int(smoke_cfg["prompts_per_quadrant"]))
    smoke = ev[["prompt_id", "quadrant", "prompt"]].copy()
    gen = {"max_new_tokens": int(smoke_cfg["max_new_tokens"]), "enable_thinking": enable_thinking,
           "temperature": float(smoke_cfg["temperature"]), "top_p": float(smoke_cfg["top_p"]), "seed": int(smoke_cfg["seed"])}
    prompts = smoke["prompt"].tolist()
    smoke["reply_base_model"] = generate(load_base_model(mcfg), tokenizer, prompts, **gen)
    if "dpo" in stages:
        for lam in (0.0, 1.0):
            set_peft_lora_lambda(aligned, lam)
            smoke[f"reply_lambda_{lam:g}"] = generate(aligned, tokenizer, prompts, **gen)
        del aligned
        smoke["lambda_replies_differ"] = smoke["reply_lambda_0"] != smoke["reply_lambda_1"]
        metrics["smoke_test"] = {"n_prompts": len(smoke), "share_lambda_replies_different": round(float(smoke["lambda_replies_differ"].mean()), 3),
                                 "by_quadrant": smoke.groupby("quadrant")["lambda_replies_differ"].mean().round(3).to_dict()}
    else:
        smoke["reply_lambda_0"] = generate(load_sft_model(mcfg, sft_dir), tokenizer, prompts, **gen)
        metrics["smoke_test"] = {"n_prompts": len(smoke)}
    smoke.to_csv(results_dir / "smoke_test.csv", index=False)
    logger.info("Smoke test", **metrics["smoke_test"])

    save_json(metrics, results_dir / "metrics.json")
    save_json({"m2_config": cfg, "data": data_meta, "environment": environment_info()}, results_dir / "experiment_config.json")
    logger.info("Completed M2 pipeline", stages=list(stages))
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ALIGN-PARETO M2 (SFT + DPO)")
    parser.add_argument("--quick", action="store_true", help="tiny smoke test; results go to results/m2_quick")
    parser.add_argument("--retrain", action="store_true", help="train again even if adapters are saved")
    parser.add_argument("--stages", default="sft,dpo", help="comma-separated: sft, dpo (dpo alone reuses the saved SFT adapter)")
    args = parser.parse_args()
    res = run_m2(Path(__file__).resolve().parents[1], quick=args.quick, retrain=args.retrain, stages=args.stages.split(","))
    print("M2 Complete:", json.dumps(res, indent=2))
