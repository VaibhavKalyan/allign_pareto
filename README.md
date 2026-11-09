# ALIGN-PARETO

Prompt-adaptive alignment for studying and reducing the alignment tax in LLMs.
Master spec: [`docs/ALIGN_PARETO_ANTIGRAVITY_INSTRUCTIONS_v2.md`](docs/ALIGN_PARETO_ANTIGRAVITY_INSTRUCTIONS_v2.md) (section numbers below refer to it).

**Status: M1 re-run on Kaggle with the revised code (2026-11-09); `results/m1/` is current. Creative F1 0.75 and safety F1 0.74 are still below the 0.80 target (safety recall 0.96). M2 (SFT + DPO from Qwen3-0.6B-Base) written and tested locally; Kaggle run needed. M3–M4 not started.** Every number reported here must come from a real run. Never type results in by hand (spec §37.11).

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

### M2: SFT + DPO alignment from the unaligned base model (§18–19)
`Qwen3-0.6B-Base` (pretrained only) → **SFT** on hh-rlhf helpful-base replies (helpful-only chat model, **λ = 0**)
→ **DPO** on hh-rlhf harmless-base pairs (safety alignment, **λ = 1**). Both stages are LoRA adapters on float32
weights; the SFT adapter is merged, the DPO adapter is scaled by λ. Starting from the already-aligned `Qwen3-0.6B`
(v1, `results/m2_v1_instruct_start/`) gave only a small effect (eval reward accuracy 0.58), because that model is
already safety-trained.
| Step | File | State |
|---|---|---|
| Load hh-rlhf: SFT examples (helpful-base) and DPO pairs (harmless-base); eval prompts excluded | `src/alignment/dpo_data.py` | done |
| Model loading (base, base + merged SFT, + DPO adapter), chat template, generation | `src/alignment/models.py` | done |
| Stage 1: SFT, loss on the reply only | `src/alignment/train_sft.py` | done |
| Stage 2: DPO, reference policy = SFT model | `src/alignment/train_dpo.py` | done |
| Orchestrator (+ reload check, base vs λ=0 vs λ=1 smoke test) | `src/run_m2.py` | done |
| Set adapter strength λ | `src/alignment/adapters.py` | done |
| Kaggle notebook (kernel `align-pareto-m2-dpo`, GPU T4, ~1.5–2 h) | `notebooks/02_alignment_training_kaggle.ipynb` | run needed |

### M3: λ sweep + Pareto frontier (§22–24, §35)
Two axes only (team decision): **safety** and **creativity**. Capability (§22) and uncertainty (§25) are
left out; they are not in the §35 "must work" list. Diversity is kept as the way creativity is measured (§23–24).
| Step | File | State |
|---|---|---|
| Safety: harmful refusal rate, benign false-refusal rate | `src/evaluation/safety.py` | to write |
| Creativity on creative prompts: diversity (1 − mean pairwise cosine), semantic cluster count, novelty | `src/evaluation/creativity.py` | to write |
| Shared eval prompt set (400 prompts from M1 val/test) + protocol for Group B | `src/evaluation/build_eval_set.py`, `docs/EVAL_PROTOCOL.md` | done |
| N=5 generations per λ, raw responses, Pareto frontier + plots | `notebooks/03_lambda_sweep_kaggle.ipynb` | to write |

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
| M3 | 2–4 h (≈400 prompts × 5 samples × 5 λ, on val + test) |
| M4 | 0.5–1 h |
| **Total** | **≈ 4–8 GPU hours** (weekly free quota is ~30 h, which leaves room for re-runs) |

## Run

```bash
python3 run_all.py --verify-only   # check files
python3 run_all.py                 # run implemented milestones
pytest tests                       # unit tests (no data download needed)
kaggle/run_on_kaggle.sh m1         # upload code, run the M1 notebook on Kaggle, copy results back
```
