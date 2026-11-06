# ALIGN-PARETO — Implementation Plan

## Project Summary

**ALIGN-PARETO** investigates the *alignment tax*: the capability and creativity losses that emerge when LLMs are aligned for safety. The system uses two lightweight binary classifiers (creative, safety-sensitive) to route prompts to an adaptive alignment controller that selects the optimal alignment strength λ — maintaining a safety floor while reducing unnecessary creative/diversity loss.

> **Core question:** Can prompt-adaptive alignment navigate the safety–creativity/diversity Pareto frontier?

---

## Milestone Roadmap

| Milestone | What it builds | Notebook |
|---|---|---|
| **M1** | Dataset pipeline + dual-head classifier | `01_classifier_kaggle.ipynb` |
| **M2** | DPO alignment training on Qwen3-0.6B | `02_alignment_training_kaggle.ipynb` |
| **M3** | λ sweep + Pareto frontier analysis | `03_lambda_sweep_kaggle.ipynb` |
| **M4** | Prompt-adaptive controller + final eval | `04_adaptive_controller_kaggle.ipynb` |

> [!IMPORTANT]
> **Start with M1 only.** Do not implement DPO, lambda sweep, or the adaptive controller until M1 works end-to-end on Kaggle.

---

## Milestone 1 — Classifier Dataset + Dual-Head Classifier

### Overview

Build a clean, leakage-safe dataset with two independent binary labels (`creative_label`, `safety_label`) and train a two-head classifier using `all-MiniLM-L6-v2` embeddings + Logistic Regression.

---

### 1.1 Repository Structure to Create Now

```text
align-pareto/
├── README.md
├── requirements.txt
├── configs/
│   ├── classifier.yaml
│   ├── alignment.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── notebooks/
│   └── 01_classifier_kaggle.ipynb      ← MAIN DELIVERABLE
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── mapping.py
│   │   ├── cleaning.py
│   │   └── dedup.py
│   ├── classifier/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── io.py
├── results/
└── reports/
```

---

### 1.2 Data Sources & Label Mapping

#### Primary Source 1: Databricks Dolly-15k
**HuggingFace:** `databricks/databricks-dolly-15k`

| Source Category | `creative_label` | `safety_label` | Reasoning |
|---|---|---|---|
| `creative_writing` | 1 | 0 (default) | Primary positive class |
| `brainstorming` | 1 | 0 (default) | Creative by nature |
| `closed_qa` | 0 | 0 | Factual, no creative intent |
| `open_qa` | 0 | 0 | Factual Q&A |
| `information_extraction` | 0 | 0 | Non-creative |
| `summarization` | 0 | 0 | Non-creative |
| `classification` | 0 | 0 | Non-creative |

> [!NOTE]
> Dolly data provides creative=1 and creative=0 examples but almost all safety=0. We rely on WildGuardMix for safety=1 examples.

#### Primary Source 2: WildGuardMix
**HuggingFace:** `allenai/wildguardmix`

| Condition | `creative_label` | `safety_label` |
|---|---|---|
| Prompt flagged as harmful/safety-sensitive | 0 (default) or 1 if creative | 1 |
| Prompt flagged as benign | 0 | 0 |

- Preserve: `harm_label`, `refusal_label`, `subcategory` (fine-grained metadata)
- The creative label for WildGuardMix entries needs inspection — some safety-sensitive prompts are creative in nature (e.g., "write a story involving a weapon"). Those get `creative=1, safety=1`.

#### Ensuring All 4 Quadrants

| Quadrant | Source strategy | Target |
|---|---|---|
| creative=1, safety=0 | Dolly creative_writing + brainstorming | ~3k |
| creative=0, safety=0 | Dolly closed_qa + open_qa etc. | ~7k |
| creative=0, safety=1 | WildGuardMix harmful, non-creative | ~3k |
| creative=1, safety=1 | WildGuardMix harmful+creative context | ~1k |

> [!WARNING]
> **Critical:** Do NOT let the classifier accidentally learn `creative → safe` and `non-creative → unsafe`. The two labels MUST be statistically separable.  
> After label construction, verify with a χ² or correlation test that the labels are not collinear.

---

### 1.3 Data Schema

Every processed row must have:

```python
{
  "id": str,              # unique row id
  "prompt": str,          # cleaned user prompt/instruction
  "creative_label": int,  # 0 or 1
  "safety_label": int,    # 0 or 1
  "source_dataset": str,  # "dolly" | "wildguardmix" | "tulu"
  "source_label": str,    # original category from source dataset
  "source_subcategory": str,  # fine-grained metadata if available
  "split": str            # "train" | "validation" | "test"
}
```

---

### 1.4 Data Quality Pipeline

Implemented in `src/data/cleaning.py` and `src/data/dedup.py`:

1. **Load** raw datasets from HuggingFace
2. **Extract** the actual user instruction (not assistant response)
3. **Remove** empty/null prompts
4. **Normalize** whitespace and obvious formatting noise (strip markdown artifacts, HTML tags)
5. **Exact deduplication** via SHA-256 hash of normalized prompt
6. **Near-duplicate detection** using MinHash LSH or cosine similarity on embeddings (threshold ~0.90)
7. **Benchmark leakage check** — flag prompts that appear in standard benchmarks (TruthfulQA, MT-Bench, etc.)
8. **Leakage-safe splits** — split by source family/cluster, NOT purely random row-level split
9. **Save statistics** — distribution of all 4 label quadrants per split

#### Split Strategy
```
Train: 80%  (~12k rows)
Val:   10%  (~1.5k rows)
Test:  10%  (~1.5k rows)
```
Split is performed per source-dataset to prevent data leakage from related prompt families.

---

### 1.5 Classifier Architecture

```text
Prompt text
     ↓
all-MiniLM-L6-v2 (384-dim sentence embedding, frozen)
     ↓
Two INDEPENDENT Logistic Regression heads:
     ├── creative_head  → sigmoid → creative_probability ∈ [0, 1]
     └── safety_head    → sigmoid → safety_probability ∈ [0, 1]
```

**Why this architecture?**
- Fast to train on Kaggle CPU (< 5 min)
- Deterministic and reproducible
- Provides calibrated probabilities (needed for the adaptive controller)
- Easy to swap out the embedding backbone later (MiniLM → DistilBERT → DeBERTa-v3-small)

**Hyperparameters (configurable in `configs/classifier.yaml`):**
```yaml
model:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  classifier: LogisticRegression
  C: 1.0
  max_iter: 1000
  solver: lbfgs
  class_weight: balanced   # important for imbalanced safety label

training:
  seed: 42
  batch_size: 256          # for embedding batches
  
inference:
  threshold_creative: 0.5
  threshold_safety: 0.5
```

---

### 1.6 Evaluation Protocol

Per-label metrics (reported separately for `creative` and `safety`):

| Metric | Reported |
|---|---|
| Precision | ✅ |
| Recall | ✅ |
| F1 | ✅ |
| ROC-AUC | ✅ |
| PR-AUC | ✅ |
| Confusion Matrix | ✅ |
| Calibration (reliability diagram) | ✅ |

Aggregate metrics:
- Macro-F1 across both labels
- Multilabel exact-match accuracy

**Special attention:** For the `safety` head, print and analyse all **false negatives** (predicted safe, actually unsafe). These are the most critical failure mode.

**Error analysis cells:** Show 10 examples each of:
- Creative FP (predicted creative, not creative)
- Creative FN (missed creative)
- Safety FP (over-flagged)
- Safety FN (missed safety) ← prioritize review
- Dual label examples (creative=1, safety=1)

---

### 1.7 Saved Artifacts (to `/kaggle/working/`)

```text
/kaggle/working/
├── data/
│   ├── classifier_dataset.parquet
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   └── dataset_statistics.json
├── models/
│   ├── creative_classifier.pkl
│   ├── safety_classifier.pkl
│   └── label_maps.json
├── results/
│   ├── predictions.csv
│   ├── metrics.json
│   ├── experiment_config.json
│   └── plots/
│       ├── confusion_matrix_creative.png
│       ├── confusion_matrix_safety.png
│       ├── roc_creative.png
│       ├── roc_safety.png
│       ├── pr_curve_creative.png
│       ├── pr_curve_safety.png
│       ├── calibration_creative.png
│       ├── calibration_safety.png
│       └── label_distribution.png
```

---

## Milestone 2 — DPO Alignment Training

### Overview

Fine-tune `Qwen/Qwen3-0.6B` with LoRA + DPO using TRL. Save the aligned adapter. This produces the "aligned model" that will differ from the base model in the lambda sweep.

### 2.1 Model & Training Strategy

```python
MODEL_NAME = "Qwen/Qwen3-0.6B"   # configurable
```

**Alignment method:** DPO (Direct Preference Optimization)  
- No reward model needed
- Works with preference pairs (chosen, rejected)
- Supported natively by HuggingFace TRL

**PEFT:** LoRA via `peft` library
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

**Quantization:** 4-bit QLoRA via `bitsandbytes` for Kaggle GPU VRAM constraints.

### 2.2 Preference Dataset

Use a public, permissively licensed preference dataset for DPO training:
- **Primary:** `Anthropic/hh-rlhf` (helpful/harmless pairs)
- **Fallback:** `Intel/orca_dpo_pairs`

Map to `(prompt, chosen, rejected)` format required by TRL DPOTrainer.

### 2.3 Lambda Parameterization

λ controls the interpolation between base and aligned adapter weights:

```text
W_λ = W_base + λ × ΔW_alignment
```

At training time, the adapter is saved at λ=1.0. At inference time, adapter weights are scaled:

```python
# Scale LoRA adapter weights by lambda
for name, param in model.named_parameters():
    if "lora" in name:
        param.data = lambda_val * original_lora_weight
```

This allows continuous λ interpolation from 0.0 (base) to 1.0 (fully aligned) without retraining.

---

## Milestone 3 — Lambda Sweep + Pareto Frontier

### Overview

Systematically evaluate the model at λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} on safety, capability, and creativity/diversity metrics.

### 3.1 Evaluation Metrics

#### Safety
- **LlamaGuard / rule-based safety classifier** on generated responses
- Refusal rate on known harmful prompts

#### Capability
- **ROUGE / BERTScore** on factual Q&A prompts against reference answers
- Optionally: subset of MMLU-style questions

#### Creativity / Diversity
```python
# For N responses per prompt:
diversity = 1 - mean(pairwise_cosine_similarity(sentence_embeddings))

# Cluster-based metric (per paper inspiration):
n_unique_clusters = len(KMeans(n_clusters=k).fit(embeddings).labels_)
```

Generate **N=5** responses per prompt with temperature=0.8.

#### Uncertainty
```python
# Token-level entropy from logits:
H_t = -sum(p * log(p) for p in softmax(logits))
sequence_entropy = mean(H_t over tokens)
```

### 3.2 Pareto Analysis

For each λ value, we have a tuple `(safety, capability, diversity)`. Plot:
- **Pareto frontier** of safety vs diversity
- **Pareto frontier** of safety vs capability
- Full 3-metric radar chart per λ

This directly answers **Q4**: *Can we observe a safety–creativity/diversity trade-off as alignment strength changes?*

---

## Milestone 4 — Adaptive Controller

### Overview

Wire the M1 classifier to drive the M3 lambda selection. For a given prompt, the controller selects λ that maximizes:

```
maximize: capability + α × creativity_diversity
subject to: safety_score ≥ safety_floor
```

### 4.1 Controller Logic

```python
def select_lambda(creative_prob: float, safety_prob: float) -> float:
    # High safety sensitivity → enforce higher lambda (more alignment)
    # High creative probability → prefer lower lambda (more freedom)
    # Calibrated on validation set
    
    safety_floor = 0.85  # configurable
    
    for lambda_val in sorted(candidate_lambdas, reverse=True):
        if expected_safety[lambda_val] >= safety_floor:
            if creative_prob > creative_threshold:
                # Take lowest safe lambda to preserve creativity
                return lambda_val
    
    return 1.0  # fallback: full alignment
```

Calibrate `safety_floor`, `creative_threshold`, and `α` on the validation set using grid search.

### 4.2 Final Comparison

Compare 4 conditions:
1. Base model (λ=0.0)
2. Fully aligned model (λ=1.0)
3. Uniform λ=0.5
4. **Adaptive controller** (λ chosen per prompt)

Using the same prompts and metrics for all 4 conditions.

---

## Advanced / Optional Components

### Null-Space Alignment (Section 27)

> [!NOTE]
> Only implement after all 4 main milestones work end-to-end.

Project safety gradient into the null-space of capability-preserving directions:
```python
g_projected = g_s - ((g_s @ g_c) / (||g_c||² + ε)) * g_c
```

Compare ordinary alignment vs projected alignment on safety, capability, and creativity.

### Value-Shift Analysis (Section 26)

Represent responses as value vectors and compute `ΔV = V_aligned - V_base`.

### Streamlit Dashboard

Interactive dashboard showing classifier output + adaptive controller decisions in real time.

---

## Open Questions

> [!IMPORTANT]
> **Q1 — WildGuardMix creative+safety examples:** How do we reliably identify prompts in WildGuardMix that are *both* safety-sensitive *and* creative? Do we manually label a sample, use keyword heuristics, or use a secondary classifier?

> [!IMPORTANT]
> **Q2 — DPO preference dataset:** Is there a preference on which dataset to use for DPO training? `hh-rlhf` is the most standard but `orca_dpo_pairs` may align better with instruction-following quality. Should we prioritize safety-alignment or general helpfulness pairs?

> [!IMPORTANT]
> **Q3 — Group B coordination:** What format will Group B (Mukund + Simar) share their benchmark results in? We need a compatible evaluation protocol from the start so we can slot their base-vs-aligned results into our Pareto analysis.

> [!IMPORTANT]
> **Q4 — Kaggle dataset upload:** Will you upload Dolly-15k and WildGuardMix as Kaggle datasets (for offline mode), or should the notebook pull from HuggingFace directly using `huggingface_hub`? Offline Kaggle datasets are faster and more reliable.

> [!NOTE]
> **Q5 — Tulu-3 subset:** Should we include a small subset of Tulu-3 SFT mixture for additional creative/safety examples, or keep M1 strictly to Dolly + WildGuardMix?

---

## Kaggle-Specific Engineering Requirements

Every notebook must satisfy:

| Requirement | Implementation |
|---|---|
| Detect GPU | `torch.cuda.get_device_name(0)`, print VRAM |
| Package versions | `pkg_resources` version dump at top |
| Configurable seed | `SEED = 42` at top, seed all: `random`, `numpy`, `torch` |
| Output path | All outputs to `/kaggle/working/` |
| Checkpointing | Save model after each epoch or at training end |
| Mixed precision | `torch.autocast` + `bitsandbytes` 4-bit where applicable |
| Restartable | Load from checkpoint if exists, skip if already done |
| Clear failures | `assert` + descriptive errors for missing datasets/models |
| Config | All hyperparams in `configs/*.yaml` loaded at top |

---

## Verification Plan

### M1 (Classifier)
- [ ] All 4 label quadrants present in dataset
- [ ] creative_label and safety_label correlation < 0.2
- [ ] Exact-match accuracy on test set > chance baseline
- [ ] F1 (creative) > 0.80
- [ ] F1 (safety) > 0.80
- [ ] ROC-AUC (both) > 0.85
- [ ] Zero data leakage between train/test (verified by exact hash check)
- [ ] Notebook runs from clean Kaggle session in < 30 min

### M2 (DPO Training)
- [ ] Loss decreases over training steps
- [ ] Smoke test: base model vs aligned model show different outputs on safety-sensitive prompt
- [ ] Adapter saves and reloads correctly

### M3 (Lambda Sweep)
- [ ] Monotonic safety increase with λ (expected trend)
- [ ] Diversity decrease with λ (alignment tax demonstrated)
- [ ] Pareto plots generated and saved

### M4 (Adaptive Controller)
- [ ] Adaptive controller outperforms fixed λ=0.5 on diversity for creative prompts
- [ ] Adaptive controller meets safety_floor on safety-sensitive prompts
- [ ] Final comparison table generated

---

## Files to Create in M1

### [NEW] `notebooks/01_classifier_kaggle.ipynb`
Main Kaggle notebook — full pipeline from data loading to classifier evaluation.

### [NEW] `src/data/download.py`
HuggingFace dataset downloader with caching and version logging.

### [NEW] `src/data/mapping.py`
Label construction logic — maps source dataset categories to `creative_label` and `safety_label`.

### [NEW] `src/data/cleaning.py`
Text normalization, empty removal, formatting cleanup.

### [NEW] `src/data/dedup.py`
Exact (SHA-256) and near-duplicate (MinHash LSH) deduplication.

### [NEW] `src/classifier/train.py`
Embedding + dual Logistic Regression training pipeline.

### [NEW] `src/classifier/predict.py`
Inference API returning `{creative_probability, safety_probability}`.

### [NEW] `src/classifier/evaluate.py`
Full evaluation suite: per-label metrics, ROC, PR curves, calibration, error analysis.

### [NEW] `src/utils/seed.py`
Set random seeds for `random`, `numpy`, `torch`.

### [NEW] `src/utils/logging.py`
Structured JSON logging with timestamps.

### [NEW] `src/utils/io.py`
Save/load helpers for Parquet, JSON, PKL, plots.

### [NEW] `configs/classifier.yaml`
All hyperparameters for M1.

### [NEW] `requirements.txt`
Pinned versions for all dependencies.

### [NEW] `README.md`
Project overview and setup instructions.
