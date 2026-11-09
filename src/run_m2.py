"""
Milestone 2 (M2): DPO alignment of Qwen3-0.6B with a LoRA adapter on hh-rlhf preference pairs.
Run from project root: `python3 src/run_m2.py` (add `--quick` for a tiny smoke test).
Checks (spec §36 M2 / implementation plan): loss decreases, the adapter reloads to the same outputs, and
lambda=0 vs lambda=1 give different replies on the evaluation prompts.
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
from src.alignment.dpo_data import load_preference_pairs
from src.alignment.train_dpo import (
    build_dpo_dataset, generate, has_cuda, is_quantized, last_checkpoint, last_token_logits, load_adapter_model,
    load_base_model, load_tokenizer, make_trainer, pairs_to_parquet_frame, render_prompt, save_adapter, train_log_frame,
)


def plot_training(log: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col, label in [("loss", "train"), ("eval_loss", "eval")]:
        if col in log:
            d = log.dropna(subset=[col])
            axes[0].plot(d["step"], d[col], marker="o", ms=3, label=label)
    axes[0].set(title="DPO loss", xlabel="step", ylabel="loss")
    for col, label in [("rewards/accuracies", "train"), ("eval_rewards/accuracies", "eval")]:
        if col in log:
            d = log.dropna(subset=[col])
            axes[1].plot(d["step"], d[col], marker="o", ms=3, label=label)
    axes[1].set(title="Reward accuracy (chosen > rejected)", xlabel="step", ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
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


def run_m2(root_dir: Path = Path("."), quick: bool = False, retrain: bool = False) -> dict:
    cfg = load_yaml(root_dir / "configs/alignment.yaml")
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    mcfg, dcfg = cfg["model"], cfg["dpo"]
    max_steps = None
    if quick:
        q = cfg["quick_test"]
        dcfg["train_pairs"] = {k: min(int(v), int(q["train_pairs"])) for k, v in dcfg["train_pairs"].items()}
        dcfg["eval_pairs"] = {k: min(int(v), int(q["eval_pairs"])) for k, v in dcfg["eval_pairs"].items()}
        dcfg.update(max_length=int(q["max_length"]), max_prompt_length=int(q["max_prompt_length"]), eval_steps=1, logging_steps=1)
        max_steps = int(q["max_steps"])
    enable_thinking = bool(mcfg.get("enable_thinking", False))

    tag = "m2_quick" if quick else "m2"
    results_dir = root_dir / "results" / tag
    models_dir = root_dir / "models" / tag
    processed_dir = root_dir / "data/processed" / tag
    adapter_dir = models_dir / "dpo_adapter"
    logger = StructuredLogger("m2_pipeline", results_dir / "execution.jsonl")
    logger.info("Starting M2 pipeline", seed=seed, quick=quick, cuda=has_cuda())
    if not has_cuda() and not quick:
        raise RuntimeError("M2 needs a CUDA GPU (Kaggle: Settings -> Accelerator -> GPU T4). Use --quick for a CPU smoke test.")

    # 1. Preference pairs (evaluation prompts are excluded so M3 never scores a prompt the adapter trained on)
    eval_prompts_path = root_dir / "data/eval/eval_prompts.csv"
    train_pairs, eval_pairs, data_meta = load_preference_pairs(cfg, root_dir / "data/raw", eval_prompts_path)
    save_parquet(pairs_to_parquet_frame(train_pairs), processed_dir / "train_pairs.parquet")
    save_parquet(pairs_to_parquet_frame(eval_pairs), processed_dir / "eval_pairs.parquet")

    tokenizer = load_tokenizer(mcfg["base_model_name"])
    train_ds, train_dropped = build_dpo_dataset(train_pairs, tokenizer, int(dcfg["max_prompt_length"]), int(dcfg["max_length"]), enable_thinking)
    eval_ds, eval_dropped = build_dpo_dataset(eval_pairs, tokenizer, int(dcfg["max_prompt_length"]), int(dcfg["max_length"]), enable_thinking)
    data_meta.update(train_pairs_used=len(train_ds), eval_pairs_used=len(eval_ds),
                     dropped_over_length={"train": train_dropped, "eval": eval_dropped})
    logger.info("Preference pairs ready", **{k: data_meta[k] for k in ["train_pairs_used", "eval_pairs_used", "dropped_over_length"]})

    # 2. DPO training (skipped when an adapter is already saved; resumes from the last checkpoint if interrupted)
    t0 = time.time()
    trained_now = retrain or not (adapter_dir / "adapter_config.json").exists()
    trained_model = None
    if trained_now:
        model = load_base_model(mcfg)
        trainer = make_trainer(model, tokenizer, train_ds, eval_ds, cfg, models_dir / "checkpoints", max_steps=max_steps)
        trainer.train(resume_from_checkpoint=last_checkpoint(models_dir / "checkpoints"))
        final_eval = trainer.evaluate()
        save_adapter(trainer, tokenizer, adapter_dir)
        log = train_log_frame(trainer)
        log.to_csv(results_dir / "train_log.csv", index=False)
        save_json(final_eval, results_dir / "final_eval.json")
        trained_model = trainer.model
        trained_model.eval()
        logger.info("DPO training finished", minutes=round((time.time() - t0) / 60, 1), eval_loss=final_eval.get("eval_loss"))
    else:
        log = pd.read_csv(results_dir / "train_log.csv")
        final_eval = json.loads((results_dir / "final_eval.json").read_text())
        logger.info("Adapter already saved; skipping training", adapter_dir=str(adapter_dir))
    plot_training(log, results_dir / "plots/dpo_training.png")

    # 3. Adapter reload check: the saved adapter must reproduce the trained model, and lambda=0 must be the base model
    reloaded = load_adapter_model(mcfg, adapter_dir)
    probe = render_prompt([{"role": "user", "content": "How do I pick a lock?"}], tokenizer, enable_thinking)
    reload_check = {}
    set_peft_lora_lambda(reloaded, 1.0)
    logits_reloaded = last_token_logits(reloaded, tokenizer, probe)
    if trained_model is not None:
        logits_trained = last_token_logits(trained_model, tokenizer, probe)
        reload_check["max_abs_logit_diff_trained_vs_reloaded"] = round(float((logits_trained - logits_reloaded).abs().max()), 6)
        del trained_model, trainer, model
    with reloaded.disable_adapter():
        logits_base = last_token_logits(reloaded, tokenizer, probe)
    set_peft_lora_lambda(reloaded, 0.0)
    logits_l0 = last_token_logits(reloaded, tokenizer, probe)
    reload_check["max_abs_logit_diff_lambda0_vs_base"] = round(float((logits_l0 - logits_base).abs().max()), 6)
    reload_check["max_abs_logit_diff_lambda1_vs_base"] = round(float((logits_reloaded - logits_base).abs().max()), 6)
    logger.info("Adapter reload check", **reload_check)
    if reload_check["max_abs_logit_diff_lambda0_vs_base"] > 1e-3:
        raise AssertionError(f"lambda=0 does not reproduce the base model: {reload_check}")
    # Not exact: during training the norms run in float32 (prepare_model_for_kbit_training), after reload in float16
    if reload_check.get("max_abs_logit_diff_trained_vs_reloaded", 0.0) > 0.25:
        raise AssertionError(f"Reloaded adapter does not match the trained model: {reload_check}")

    # 4. Smoke test: greedy replies at lambda=0 (base) and lambda=1 (aligned) on validation evaluation prompts
    scfg = cfg["smoke_test"]
    ev = pd.read_csv(eval_prompts_path)
    ev = ev[ev["split"] == "validation"].groupby("quadrant", group_keys=False).head(int(scfg["prompts_per_quadrant"]))
    smoke = ev[["prompt_id", "quadrant", "prompt"]].copy()
    for lam in (0.0, 1.0):
        set_peft_lora_lambda(reloaded, lam)
        smoke[f"reply_lambda_{lam:g}"] = generate(reloaded, tokenizer, smoke["prompt"].tolist(), int(scfg["max_new_tokens"]), enable_thinking)
    smoke["differs"] = smoke["reply_lambda_0"] != smoke["reply_lambda_1"]
    smoke.to_csv(results_dir / "smoke_test.csv", index=False)
    logger.info("Smoke test", n=len(smoke), share_different=round(float(smoke["differs"].mean()), 3))

    metrics = {
        "quick_test": quick,
        "trained_this_run": trained_now,
        "train_minutes": round((time.time() - t0) / 60, 1) if trained_now else None,
        "train_steps": int(log["step"].max()) if "step" in log else None,
        "train_loss": _first_last(log, "loss"),
        "train_reward_accuracy": _first_last(log, "rewards/accuracies"),
        "train_reward_margin": _first_last(log, "rewards/margins"),
        "final_eval": {k: round(float(v), 4) for k, v in final_eval.items() if isinstance(v, (int, float))},
        "reload_check": reload_check,
        "smoke_test": {"n_prompts": len(smoke), "share_replies_different": round(float(smoke["differs"].mean()), 3),
                       "by_quadrant": smoke.groupby("quadrant")["differs"].mean().round(3).to_dict()},
        "quantized_4bit": is_quantized(reloaded),
    }
    save_json(metrics, results_dir / "metrics.json")
    save_json({"m2_config": cfg, "data": data_meta, "environment": environment_info()}, results_dir / "experiment_config.json")
    logger.info("Completed M2 pipeline", train_loss=metrics["train_loss"], share_different=metrics["smoke_test"]["share_replies_different"])
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ALIGN-PARETO M2")
    parser.add_argument("--quick", action="store_true", help="tiny smoke test; results go to results/m2_quick")
    parser.add_argument("--retrain", action="store_true", help="train again even if an adapter is already saved")
    args = parser.parse_args()
    res = run_m2(Path(__file__).resolve().parents[1], quick=args.quick, retrain=args.retrain)
    print("M2 Complete:", json.dumps(res, indent=2))
