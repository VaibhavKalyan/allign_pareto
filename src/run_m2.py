"""
Milestone 2 (M2): supervised fine-tuning (SFT) of the unaligned Qwen3-0.6B-Base into a helpful-only chat model.
  Human-written replies (no_robots, oasst2) -> models/m2/sft_adapter
Run from project root: `python3 src/run_m2.py` (`--quick` for a tiny smoke test).
Checks: loss falls, the saved adapter reloads to the trained model, and replies of the raw base model and the SFT model
on evaluation prompts are saved, with the share of replies that stop on their own and the share with stray characters.
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
from src.alignment.models import (
    generate, has_cuda, last_checkpoint, last_token_logits, load_base_model, load_sft_model, load_tokenizer,
    render_prompt, save_adapter,
)
from src.alignment.sft_data import load_sft_data
from src.alignment.train_sft import build_sft_dataset, make_sft_trainer

PROBE_PROMPT = "How do I pick a lock?"
# Characters from scripts the training data does not use; they appeared where the v3 SFT model failed to stop
STRAY_CHARS = r"[฀-๿一-鿿Ѐ-ӿ�]"


def plot_training(log: pd.DataFrame, path: Path) -> None:
    """Loss and token accuracy, train and eval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cols, ylabel in [(axes[0], ("loss", "eval_loss"), "loss"),
                             (axes[1], ("mean_token_accuracy", "eval_mean_token_accuracy"), "token accuracy")]:
        for col, label in zip(cols, ("train", "eval")):
            if col in log:
                d = log.dropna(subset=[col])
                ax.plot(d["step"], d[col], marker="o", ms=3, label=label)
        ax.set(xlabel="step", ylabel=ylabel, title=f"SFT {ylabel}")
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


def _rounded(d: dict) -> dict:
    return {k: round(float(v), 4) for k, v in d.items() if isinstance(v, (int, float))}


def apply_quick_test(cfg: dict) -> int:
    """Shrink SFT to a few examples and steps; returns max_steps."""
    q, scfg = cfg["quick_test"], cfg["sft"]
    scfg["train_examples"] = {k: min(int(v), int(q["train_examples"])) for k, v in scfg["train_examples"].items()}
    scfg["eval_examples"] = {k: min(int(v), int(q["eval_examples"])) for k, v in scfg["eval_examples"].items()}
    scfg.update(max_length=int(q["max_length"]), eval_steps=1, logging_steps=1)
    return int(q["max_steps"])


def reply_quality(replies: pd.Series, tokenizer, max_new_tokens: int) -> dict:
    """Share of replies that ended before the token limit, and share containing stray characters."""
    n_tokens = replies.astype(str).map(lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"]))
    return {"share_stopped": round(float((n_tokens < max_new_tokens - 5).mean()), 3),
            "share_stray_chars": round(float(replies.astype(str).str.contains(STRAY_CHARS).mean()), 3),
            "mean_reply_tokens": round(float(n_tokens.mean()), 1)}


def run_m2(root_dir: Path = Path("."), quick: bool = False, retrain: bool = False) -> dict:
    cfg = load_yaml(root_dir / "configs/alignment.yaml")
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    mcfg, scfg, smoke_cfg = cfg["model"], cfg["sft"], cfg["smoke_test"]
    max_steps = apply_quick_test(cfg) if quick else None
    enable_thinking = bool(mcfg.get("enable_thinking", False))

    tag = "m2_quick" if quick else "m2"
    results_dir = root_dir / "results" / tag
    models_dir = root_dir / "models" / tag
    processed_dir = root_dir / "data/processed" / tag
    sft_dir = models_dir / "sft_adapter"
    logger = StructuredLogger("m2_pipeline", results_dir / "execution.jsonl")
    logger.info("Starting M2 pipeline (SFT only)", seed=seed, quick=quick, cuda=has_cuda(), base_model=mcfg["base_model_name"])
    if not has_cuda() and not quick:
        raise RuntimeError("M2 needs a CUDA GPU (Kaggle: Settings -> Accelerator -> GPU T4). Use --quick for a CPU smoke test.")

    # 1. Training data (evaluation prompts excluded so M3 never scores a trained-on prompt)
    eval_prompts_path = root_dir / "data/eval/eval_prompts.csv"
    data, data_meta = load_sft_data(cfg, root_dir / "data/raw", eval_prompts_path)
    tokenizer = load_tokenizer(mcfg)
    datasets, used, dropped = {}, {}, {}
    for name, df in data.items():
        save_parquet(df.assign(prompt=df["prompt"].map(json.dumps)), processed_dir / f"{name}.parquet")
        datasets[name], dropped[name] = build_sft_dataset(df, tokenizer, int(scfg["max_length"]), enable_thinking)
        used[name] = len(datasets[name])
    data_meta.update(used=used, dropped_over_length=dropped, end_of_reply_token=tokenizer.eos_token)
    logger.info("Training data ready", used=used, dropped_over_length=dropped, end_of_reply_token=tokenizer.eos_token)

    metrics = {"quick_test": quick, "base_model": mcfg["base_model_name"], "stages": ["sft"],
               "end_of_reply_token": tokenizer.eos_token}
    probe = render_prompt([{"role": "user", "content": PROBE_PROMPT}], tokenizer, enable_thinking)

    # 2. SFT (skipped when an adapter is already saved, unless retrain)
    sft_trained = retrain or not (sft_dir / "adapter_config.json").exists()
    if sft_trained:
        t0 = time.time()
        ckpt_dir = models_dir / "sft_checkpoints"
        trainer = make_sft_trainer(load_base_model(mcfg), tokenizer, datasets["sft_train"], datasets["sft_eval"], cfg, ckpt_dir, max_steps=max_steps)
        trainer.train(resume_from_checkpoint=last_checkpoint(ckpt_dir))
        final_eval = trainer.evaluate()
        save_adapter(trainer, tokenizer, sft_dir)
        log = pd.DataFrame(trainer.state.log_history)
        log.to_csv(results_dir / "sft_train_log.csv", index=False)
        minutes = round((time.time() - t0) / 60, 1)
        save_json({"minutes": minutes, **final_eval}, results_dir / "sft_final_eval.json")
        logger.info("SFT finished", minutes=minutes, eval_loss=final_eval.get("eval_loss"))
        trainer.model.eval()
        logits_trained = last_token_logits(trainer.model, tokenizer, probe)
        del trainer
        diff = round(float((logits_trained - last_token_logits(load_sft_model(mcfg, sft_dir), tokenizer, probe)).abs().max()), 6)
        metrics["reload_check"] = {"max_abs_logit_diff_sft_trained_vs_reloaded": diff}
        logger.info("Adapter reload check", **metrics["reload_check"])
        # Small fp16-autocast differences only: the trained copy may still run in autocast, the reloaded one does not
        if diff > 0.1:
            raise AssertionError(f"Reloaded adapter does not match the trained model: {metrics['reload_check']}")
    else:
        log = pd.read_csv(results_dir / "sft_train_log.csv")
        final_eval = json.loads((results_dir / "sft_final_eval.json").read_text())
        logger.info("SFT adapter already saved; skipping training", adapter_dir=str(sft_dir))
    plot_training(log, results_dir / "plots/sft_training.png")
    metrics["sft"] = {"trained_this_run": sft_trained, "train_steps": int(log["step"].max()),
                      "train_loss": _first_last(log, "loss"), "train_token_accuracy": _first_last(log, "mean_token_accuracy"),
                      "final_eval": _rounded(final_eval)}

    # 3. Smoke test: sampled replies (M3's settings, fixed seed) of the raw base model and the SFT model
    ev = pd.read_csv(eval_prompts_path)
    ev = ev[ev["split"] == "validation"].groupby("quadrant", group_keys=False).head(int(smoke_cfg["prompts_per_quadrant"]))
    smoke = ev[["prompt_id", "quadrant", "prompt"]].copy()
    gen = {"max_new_tokens": int(smoke_cfg["max_new_tokens"]), "enable_thinking": enable_thinking,
           "temperature": float(smoke_cfg["temperature"]), "top_p": float(smoke_cfg["top_p"]), "seed": int(smoke_cfg["seed"])}
    prompts = smoke["prompt"].tolist()
    smoke["reply_base_model"] = generate(load_base_model(mcfg), tokenizer, prompts, **gen)
    smoke["reply_sft"] = generate(load_sft_model(mcfg, sft_dir), tokenizer, prompts, **gen)
    smoke.to_csv(results_dir / "smoke_test.csv", index=False)
    # Reply quality: a reply that runs to the token limit never stopped (v3's failure); stray characters mark a broken end
    metrics["smoke_test"] = {"n_prompts": len(smoke), "max_new_tokens": gen["max_new_tokens"],
                             "base_model": reply_quality(smoke["reply_base_model"], tokenizer, gen["max_new_tokens"]),
                             "sft": reply_quality(smoke["reply_sft"], tokenizer, gen["max_new_tokens"])}
    logger.info("Smoke test", **metrics["smoke_test"])

    save_json(metrics, results_dir / "metrics.json")
    save_json({"m2_config": cfg, "data": data_meta, "environment": environment_info()},
              results_dir / "experiment_config.json")
    logger.info("Completed M2 pipeline (SFT only)")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ALIGN-PARETO M2 (SFT on Qwen3-0.6B-Base)")
    parser.add_argument("--quick", action="store_true", help="tiny smoke test; results go to results/m2_quick")
    parser.add_argument("--retrain", action="store_true", help="train again even if an adapter is saved")
    args = parser.parse_args()
    res = run_m2(Path(__file__).resolve().parents[1], quick=args.quick, retrain=args.retrain)
    print("M2 Complete:", json.dumps(res, indent=2))
