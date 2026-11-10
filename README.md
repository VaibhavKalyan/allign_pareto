# ALIGN-PARETO

Prompt-adaptive alignment for studying and reducing the alignment tax in LLMs.
Master spec: [`docs/ALIGN_PARETO_ANTIGRAVITY_INSTRUCTIONS_v2.md`](docs/ALIGN_PARETO_ANTIGRAVITY_INSTRUCTIONS_v2.md) (section numbers below refer to it).

**Status: M1 re-run on Kaggle with the revised code (2026-11-09); `results/m1/` is current. Creative F1 0.75 and safety F1 0.74 are still below the 0.80 target (safety recall 0.96). M2 done (v4, 2026-11-10): SFT on no_robots + oasst2, no DPO. M3 done (2026-11-10): λ blends our SFT model and Qwen's `Qwen/Qwen3-0.6B`; safety rises and creativity falls steadily with λ, every λ is Pareto-optimal (`results/m3/pareto.csv`, `results/m3/plots/`). M4 not started.** Every number reported here must come from a real run. Never type results in by hand (spec §37.11).

## Folder layout

```
configs/      classifier.yaml, alignment.yaml, evaluation.yaml   (all hyperparameters)
docs/         spec, implementation plan, dataset decision, annotation guide
referrence/   the three reference papers
src/          library code; notebooks only call into it
notebooks/    one Kaggle notebook per milestone
data/         raw/ processed/ metadata/   (filled by M1; not committed)
models/       classifier + DPO adapter    (filled by M1/M2; not committed)
results/      CSV/JSON metrics + plots    (filled by every milestone)
reports/      final write-up
_archive/     earlier simulated code, kept for reference only; never import from it
unused/       code no longer used by the pipeline (DPO stage, λ adapter scaling); see unused/README.md
```

## Roadmap: which file each step goes in

Work strictly in order. Do not start a milestone until the previous one has run on Kaggle (§36).

### M1: Creative + safety classifier (§5–17, §36)
| Step | File | State |
|---|---|---|
| Load real Dolly-15k + WildGuardMix (fail loudly; no synthetic fallback) | `src/data/download.py` | done |
| Map to `creative_label` / `safety_label` (strict creative rule for WildGuardMix, incl. brainstorming; Dolly `creative_writing` rows that are plain questions excluded); unknown labels excluded | `src/data/mapping.py` | done |
| Clean text, drop WildGuardTest overlap (exact here; near-duplicates in `dedup.py`) | `src/data/cleaning.py` | done |
| SHA-256 + MinHash near-dup removal, prompt families, family-safe 80/10/10 split | `src/data/dedup.py` | done |
| MiniLM embeddings + two LR heads (no silent TF-IDF fallback) | `src/classifier/train.py` | done |
| Inference API | `src/classifier/predict.py` | done |
| Metrics, confusion matrices, calibration, val-tuned thresholds (safety: recall-first), per-source breakdowns | `src/classifier/evaluate.py` | done |
| Orchestrator (+ WildGuardTest external eval, fictional-framing eval slice, error analysis, audit files) | `src/run_m1.py` | done |
| Unit tests for label rules, leakage filter, thresholds (`pytest tests`) | `tests/test_m1.py` | done |
| Kaggle notebook (Kaggle code: dataset `align-pareto-src`, kernel `align-pareto-m1-classifier`) | `notebooks/01_classifier_kaggle.ipynb` | done (run with `kaggle/run_on_kaggle.sh m1`) |
| Hand-check creative labels (fill `human_creative_label`) | `results/m1/creative_rule_audit.csv`, `results/m1/fictional_framing_predictions.csv` | to do (team) |
| Decide whether Dolly `brainstorming` (mostly lists/recommendations) and `creative_writing` letters/emails count as creative | `docs/ANNOTATION_GUIDE.md` | to do (team) |

### M2: SFT from the unaligned base model (§18)
`Qwen3-0.6B-Base` (pretrained only) → **SFT** (LoRA, float32 weights) on human-written replies (no_robots + oasst2,
refusal-like replies removed): helpful-only chat model. DPO is no longer run; its code is in `unused/`.
Replies end with `<|endoftext|>`: v3 (`results/m2_v3_no_stop/`) ended them with `<|im_end|>`, whose base-model
weights are untrained, and never stopped.
Earlier runs, kept for the report: v1 (`results/m2_v1_instruct_start/`, DPO on the already-aligned Qwen3-0.6B: small
effect) and v2 (`results/m2_v2_hh_sft/`, SFT on hh-rlhf helpful replies: λ = 0 deflected and looped, λ = 1 refused
almost everything).
| Step | File | State |
|---|---|---|
| SFT data: no_robots + oasst2, refusal-like replies and eval prompts removed | `src/alignment/sft_data.py` | done |
| SFT training, loss on the reply only | `src/alignment/train_sft.py` | done |
| Model loading (base, base + merged SFT), chat template, generation | `src/alignment/models.py` | done |
| Orchestrator (SFT only; reload check; smoke test with share of replies stopped / with stray characters) | `src/run_m2.py` | done |
| Kaggle notebook (kernel `align-pareto-m2-dpo`, GPU T4) | `notebooks/02_alignment_training_kaggle.ipynb` | done: v4 in `results/m2/` (10/16 replies stop, rest hit the 200-token limit; no stray characters) |

### M3: λ sweep + Pareto frontier (§22–24, §35)
Two axes only (team decision): **safety** and **creativity**. Capability (§22) and uncertainty (§25) are
left out; they are not in the §35 "must work" list. Diversity is kept as the way creativity is measured (§23–24).

Models (`configs/evaluation.yaml` → `models`, details in [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)):
λ blends the weights of `sft-helpful-only` (our M2 SFT model, λ = 0) and `qwen3-instruct` (Qwen's aligned
`Qwen/Qwen3-0.6B`, λ = 1): W(λ) = W_sft + λ · (W_qwen3 − W_sft). Both start from Qwen3-0.6B-Base. The λ direction
carries all of Qwen's post-training, not only safety. Group B compares the two endpoints.
| Step | File | State |
|---|---|---|
| Batched seeded sampling (N = 5 per prompt) | `src/evaluation/generate.py` | done |
| Safety judge: Qwen3Guard-Gen-4B (harm + refusal in one call) | `src/evaluation/judge.py` | done |
| Metrics: harmful response / refusal / over-refusal rates; diversity, clusters; lexical diversity (style side metric); broken replies; Pareto table | `src/evaluation/metrics.py`, `tests/test_m3_metrics.py` | done |
| Orchestrator: generate → judge → score, resumable, `--quick` | `src/run_m3.py` | done |
| λ dial: weight interpolation SFT → Qwen3-0.6B (endpoints checked exact at the start of every run) | `src/alignment/interpolate.py` | done |
| Shared eval prompt set (400 prompts from M1 val/test) + protocol for Group B | `src/evaluation/build_eval_set.py`, `docs/EVAL_PROTOCOL.md` | done |
| Publish the M2 SFT adapter as Kaggle dataset `align-pareto-m2-adapters` (M3 + Group B input) | `kaggle/publish_adapters.sh` | done |
| Kaggle notebook (kernel `align-pareto-m3-sweep`, GPU T4) | `notebooks/03_lambda_sweep_kaggle.ipynb` | done (`kaggle/run_on_kaggle.sh m3`): `results/m3/` |
| Hand-check ~50 judge labels against `raw_responses.jsonl` (local only, not in git) | `docs/EVAL_PROTOCOL.md` §7 | to do |

Known limitation of the M3 run: blended models stop less often than the endpoints, worst at λ = 0.5 (`stopped_rate`
in `results/m3/metrics.csv`; λ = 0 ends with `<|endoftext|>`, λ = 1 with `<|im_end|>`). Their replies run longer,
which can affect harm and diversity scores at middle λ.

### M4: Adaptive controller (§20)
| Step | File | State |
|---|---|---|
| λ*(x) from classifier probs, calibrated on the **val** sweep: maximise creativity subject to safety ≥ floor | `src/alignment/lambda_controller.py` | to write |
| Final comparison on **test**: base, λ=1, λ=0.5, adaptive | `notebooks/04_adaptive_controller_kaggle.ipynb` | to write |

### Optional (only after M4 works, §35)
Null-space projection `src/alignment/nullspace.py` (§27) · value-shift analysis (§26) · Streamlit dashboard.

## Estimated Kaggle GPU time (T4/P100)

| Milestone | Estimate |
|---|---|
| M1 | 15–30 min (CPU is enough) |
| M2 | 1.5–2 h (SFT 8k examples + DPO 5k pairs × 2 epochs) |
| M3 | ≈ 2.5–4 h (400 prompts × 5 samples × 5 λ = 10,000 replies, then judging) |
| M4 | 0.5–1 h |
| **Total** | **≈ 4–8 GPU hours** (weekly free quota is ~30 h, which leaves room for re-runs) |

## Run

```bash
python3 run_all.py --verify-only   # check files
python3 run_all.py                 # run implemented milestones
pytest tests                       # unit tests (no data download needed)
kaggle/run_on_kaggle.sh m1         # upload code, run the M1 notebook on Kaggle, copy results back
```
