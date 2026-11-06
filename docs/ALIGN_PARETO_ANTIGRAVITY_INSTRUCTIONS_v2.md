# ALIGN-PARETO — Antigravity Project Instructions (v2)

## 0. Purpose

This file is the working specification for the GenAI course project.

Treat it as the source of truth for repository structure, data preparation, classifier training, experiments, and Kaggle notebooks.

The project is **ongoing**. Never claim an improvement before it is demonstrated by experiments.

---

# 1. Project title

## ALIGN-PARETO
### Prompt-Adaptive Alignment for Studying and Reducing the Alignment Tax in LLMs

### Core research question

> **What changes when an LLM is aligned, and can alignment be adapted so that we maintain the required safety level while reducing unnecessary losses in creativity and response diversity?**

We are not trying to prove that alignment is bad.

We are not trying to prove that less alignment is always better.

We want to measure the trade-off and investigate whether a prompt-aware alignment strategy can preserve useful behavior while maintaining safety.

---

# 2. Team split

## Group A — Adaptive Alignment / System
### Vaibhav + Sumath + Usnish

Main responsibilities:

- Public dataset collection and preparation
- **Creative + Safety-Sensitivity prompt classifier**
- Prompt routing
- Alignment-strength controller
- `lambda` selection
- Adaptive alignment logic
- Lightweight mitigation experiments
- Integration of the ALIGN-PARETO pipeline

## Group B — Evaluation / Benchmarking
### Mukund + Simar

Main responsibilities:

- Benchmark already existing LLMs
- Compare **base** and **instruction-tuned/aligned** models
- Measure:
  - Safety
  - Capability
  - Creativity / response diversity
  - Uncertainty where applicable
- Establish baseline measurements that Group A will use for comparison

### Separation of work

Group B primarily asks:

> **What changes between existing base and aligned/instruction-tuned LLMs?**

Group A primarily asks:

> **Can we use prompt characteristics to choose alignment behavior that reduces unnecessary loss in creativity/diversity while maintaining safety?**

The evaluation protocol should be shared so that the final systems can be compared fairly.

---

# 3. Simplified project architecture

The classifier is intentionally **small and focused**.

We only want two independent labels:

1. `creative`
2. `safety_sensitive`

This is a **multi-label problem**, not a six-class mutually exclusive classification problem.

A prompt can have both labels.

Example:

> "Write a fictional story about a dangerous chemical experiment."

Output:

```text
creative = 1
safety_sensitive = 1
```

Another prompt:

> "Explain how TCP works."

Output:

```text
creative = 0
safety_sensitive = 0
```

Another:

> "Give a safe, non-actionable explanation of a high-risk topic."

Output may be:

```text
creative = 0
safety_sensitive = 1
```

Therefore use two independent binary heads.

---

# 4. Final system flow

```text
                         ALIGN-PARETO

                            PROMPT
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Creative Classifier │
                    │ + Safety Classifier │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
            Creative?                 Safety-sensitive?
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    Alignment Controller
                               │
                               ▼
                         Choose λ
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
               Base model            Aligned model
                   │                       │
                   └───────────┬───────────┘
                               ▼
                           RESPONSE
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                  Safety   Capability  Diversity
                               │
                               ▼
                         Uncertainty
                               │
                               ▼
                        Final Analysis
```

---

# 5. What the classifier does

The classifier is only responsible for answering two questions:

### Question A
> Is this prompt creative in nature?

Output:

```text
creative_probability ∈ [0, 1]
```

### Question B
> Does this prompt require a heightened safety sensitivity?

Output:

```text
safety_probability ∈ [0, 1]
```

Example API:

```python
{
    "creative_probability": 0.91,
    "safety_probability": 0.08
}
```

For an ambiguous prompt:

```python
{
    "creative_probability": 0.57,
    "safety_probability": 0.82
}
```

The controller should use probabilities/confidence rather than only hard labels whenever practical.

---

# 6. Why only two labels?

We intentionally removed:

- reasoning
- coding
- factual
- general

from the classifier.

Those categories are not needed for the first version of our controller.

The important distinction for this project is:

```text
Creative?       → potentially preserve more generation freedom
Safety-sensitive? → enforce a higher safety floor
```

Everything else can be treated as non-creative for the first prototype.

This makes the classifier:

- easier to train
- easier to evaluate
- easier to explain
- less prone to class-boundary confusion
- more directly connected to the alignment-control hypothesis

---

# 7. Data schema

The processed classifier dataset should contain:

```text
id
prompt
creative_label
safety_label
source_dataset
source_label
source_subcategory
split
```

Labels:

```text
creative_label ∈ {0, 1}
safety_label ∈ {0, 1}
```

Do not discard original source labels.

Do not discard source provenance.

---

# 8. Public datasets

Use publicly released datasets as the main academic data source.

Do not use the synthetic starter dataset as the main final dataset.

## 8.1 Databricks Dolly-15k

https://huggingface.co/datasets/databricks/databricks-dolly-15k

Use for creative positives and non-creative negatives.

Creative-positive source categories:

```text
creative_writing
brainstorming
```

These become:

```text
creative_label = 1
```

Potential non-creative categories:

```text
closed_qa
open_qa
information_extraction
summarization
classification
```

These become candidate:

```text
creative_label = 0
```

Inspect before final inclusion.

---

## 8.2 WildGuardMix

https://huggingface.co/datasets/allenai/wildguardmix

Use primarily for the safety label.

Map human/curated prompt safety information to:

```text
safety_label = 1
```

for prompts identified as safety-sensitive/harmful.

Use clearly benign prompts for:

```text
safety_label = 0
```

Preserve original:

- harm label
- refusal label when present
- fine-grained subcategory when present

Do not throw away fine-grained safety metadata.

---

## 8.3 Optional secondary sources

Tulu 3 SFT mixture:

https://huggingface.co/datasets/allenai/tulu-3-sft-mixture

This can be used later for:

- more creative positives/negatives
- safety examples
- robustness
- cross-dataset testing

Do not start by loading the entire mixture.

Track the original component/source and license.

---

# 9. Label construction

The project now has **two independent labels**.

## Creative label

### Positive examples

Prompts whose primary intent is creative generation, such as:

- story writing
- fictional scenes
- poetry
- character creation
- imaginative world-building
- creative dialogue

Set:

```text
creative_label = 1
```

### Negative examples

Prompts that ask for:

- factual explanation
- Q&A
- summarization
- information extraction
- classification
- non-creative instruction following

Set:

```text
creative_label = 0
```

Do not use keyword matching alone.

Review ambiguous examples.

---

# 10. Safety label

### Positive examples

Use public safety datasets with established harmful/sensitive annotations.

Set:

```text
safety_label = 1
```

### Negative examples

Use clearly benign prompts.

Set:

```text
safety_label = 0
```

Important:

The safety classifier is a **routing signal**, not the actual safety mechanism.

A separate model/evaluation layer must still verify safety.

---

# 11. Multi-label examples

Include examples covering all four combinations:

| Creative | Safety-sensitive | Example type |
|---:|---:|---|
| 0 | 0 | ordinary factual request |
| 1 | 0 | harmless story-writing request |
| 0 | 1 | safety-sensitive informational request |
| 1 | 1 | creative prompt involving a sensitive topic |

This is critical.

Do not accidentally create only:

```text
creative → safe
non-creative → unsafe
```

because then the classifier will learn the wrong correlation.

The two labels must be statistically separable.

---

# 12. Data quality rules

The dataset pipeline is a core contribution.

Required steps:

1. Load the public datasets.
2. Extract the actual user prompt/instruction.
3. Construct `creative_label`.
4. Construct `safety_label`.
5. Preserve original dataset labels.
6. Remove missing/empty prompts.
7. Normalize obvious formatting noise.
8. Exact-deduplicate.
9. Detect near duplicates where practical.
10. Remove benchmark-answer leakage from classifier text.
11. Create leakage-safe train/validation/test splits.
12. Save all mapping logic.
13. Save dataset statistics.

---

# 13. Leakage prevention

This is especially important because we will later use separate benchmark datasets.

Do not put the same benchmark prompt into both:

```text
classifier training
```

and

```text
final LLM evaluation
```

where that could contaminate the experiment.

Also avoid near-identical prompt variants across train/test.

When practical, split by source/example family rather than pure random row splitting.

---

# 14. Dataset target

Start with a clean pilot dataset.

Target roughly:

```text
10k–20k unique prompts
```

with adequate representation of the four label combinations.

A possible first target:

```text
creative=1, safety=0     ~3k
creative=0, safety=0     ~7k
creative=1, safety=1     ~1k
creative=0, safety=1     ~3k
```

These numbers are starting targets, not mandatory exact counts.

Do not distort the dataset unnaturally just to reach equal class sizes.

Report the real distribution.

---

# 15. Recommended classifier

Start with a simple, strong baseline.

```text
Prompt
   ↓
all-MiniLM-L6-v2 embedding
   ↓
two independent Logistic Regression heads
   ├── creative probability
   └── safety probability
```

Do not fine-tune a large causal LLM for the first classifier experiment.

After the baseline is stable, optionally compare against:

- MiniLM
- DistilBERT
- DeBERTa-v3-small

---

# 16. Classifier evaluation

Because this is multi-label classification, report per-label and aggregate metrics.

For `creative`:

- Precision
- Recall
- F1
- ROC-AUC if probabilities are available
- PR-AUC if useful

For `safety`:

- Precision
- Recall
- F1
- ROC-AUC if probabilities are available
- PR-AUC if useful

Also report:

- macro F1 across the two labels
- multilabel exact-match accuracy if useful
- confusion matrices for each binary head
- calibration where practical

For safety routing, prioritize examining **false negatives** carefully.

---

# 17. Error analysis

Show real examples of:

```text
creative predicted incorrectly
safety predicted incorrectly
creative + safety combinations
```

Pay particular attention to:

```text
creative vs non-creative ambiguity
safe creative prompts
sensitive creative prompts
safe informational discussion of sensitive topics
```

This is likely to be more useful than chasing tiny improvements in aggregate accuracy.

---

# 18. Model/alignment strategy

Do not attempt full RLHF/PPO initially.

Use:

```text
small open causal LM
        +
LoRA / QLoRA
        +
DPO or another lightweight preference-alignment method
```

Initial model:

```python
MODEL_NAME = "Qwen/Qwen3-0.6B"
```

Keep model name configurable.

Use parameter-efficient training.

References:

- https://huggingface.co/docs/trl/dpo_trainer
- https://huggingface.co/docs/trl/peft_integration
- https://huggingface.co/docs/transformers/peft

---

# 19. Alignment strength λ

Conceptually:

```text
W_lambda = W_base + lambda * DeltaW_alignment
```

with:

```text
lambda = 0.00
lambda = 0.25
lambda = 0.50
lambda = 0.75
lambda = 1.00
```

Interpretation:

```text
0.00 → base behavior
1.00 → full alignment-adapter contribution
```

Do not assume in advance that a particular lambda is best.

Measure it.

Use supported PEFT adapter workflows rather than manually modifying model tensors at first.

---

# 20. Adaptive controller logic

The first version should use the classifier probabilities.

Inputs:

```text
creative_probability
safety_probability
```

Output:

```text
selected_lambda
```

Example conceptual behavior:

```text
high safety probability
      ↓
higher safety floor

high creative probability
      ↓
preserve more generation freedom where safely possible
```

Do not hard-code final lambda values as the scientific result.

Instead, calibrate candidate lambda values on a validation set.

A useful initial optimization objective is:

```text
maximize:
    capability + alpha * creativity/diversity

subject to:
    safety >= safety_floor
```

The exact formulation can be refined after baseline experiments.

---

# 21. First experiment

Before adaptive control, compare:

```text
Base model
vs
Instruction/aligned model
```

using the benchmarks produced by Group B.

Measure:

- safety
- capability
- diversity/creativity
- uncertainty where applicable

This establishes the motivation for the adaptive controller.

---

# 22. Lambda sweep

After alignment training, run:

```text
lambda = [0.0, 0.25, 0.5, 0.75, 1.0]
```

Keep the following fixed across lambda values:

- evaluation prompts
- model version
- generation parameters
- number of samples
- evaluation metrics

Save:

```text
lambda
safety
capability
diversity
uncertainty
raw_responses
```

---

# 23. Creativity measurement

Because the project now focuses on **creativity rather than general task categories**, make creativity an explicit evaluation dimension.

For creative prompts:

1. Generate multiple responses.
2. Measure semantic diversity.
3. Measure pairwise similarity.
4. Optionally score novelty using embedding distance.
5. Compare base vs aligned vs adaptive settings.

Do not equate:

```text
more diverse = always better
```

Instead, investigate whether alignment causes measurable homogenization and whether adaptive control can reduce unnecessary homogenization while satisfying the safety constraint.

---

# 24. Response diversity

For selected prompts:

```text
generate N = 5–10 responses
```

Use sentence embeddings to calculate an initial diversity statistic.

Possible simple metric:

```text
diversity = 1 - mean(pairwise_cosine_similarity)
```

Later implement a clustering-based metric inspired by the response-homogenization paper.

Keep both metrics separately named.

---

# 25. Uncertainty

Use generation scores/logits where supported.

Compute token-level entropy:

```text
H_t = -Σ p(token) log p(token)
```

Compare:

```text
sampling diversity
vs
token-level entropy
```

Investigate whether alignment changes the relationship between them.

---

# 26. Value-shift analysis

This is optional for the first MVP.

If included, measure whether alignment changes behavior across multiple value dimensions.

Represent a response as:

```text
V = [v1, v2, ..., vk]
```

and compare:

```text
Delta_V = V_aligned - V_base
```

Label this as an adapted analysis unless the exact published protocol is reproduced.

---

# 27. Null-space experiment

This is an advanced stage only.

Do not implement it before the main pipeline works.

For a safety gradient `g_s` and a capability-preservation direction `g_c`:

```python
g_projected =
    g_s - ((g_s @ g_c) / (||g_c||^2 + eps)) * g_c
```

For multiple capability directions, project the safety update into the complement/null space of their span.

Compare:

```text
ordinary alignment
vs
projected alignment
```

on:

- safety
- capability
- creativity/diversity

Clearly label this as a lightweight adaptation inspired by null-space constrained safety alignment unless the full published method is reproduced.

---

# 28. Kaggle workflow

All first-stage training and experiments must be runnable in Kaggle.

## Notebook 01

```text
01_classifier_kaggle.ipynb
```

Pipeline:

```text
Environment
↓
Install dependencies
↓
Load public datasets
↓
Create creative + safety labels
↓
Clean / deduplicate
↓
Leakage-safe split
↓
Dataset audit
↓
Train two binary classifier heads
↓
Evaluate
↓
Error analysis
↓
Save artifacts
```

## Notebook 02

```text
02_alignment_training_kaggle.ipynb
```

Pipeline:

```text
Load small open model
↓
Load preference dataset
↓
LoRA/QLoRA + DPO
↓
Save aligned adapter
↓
Smoke-test generation
```

## Notebook 03

```text
03_lambda_sweep_kaggle.ipynb
```

Pipeline:

```text
Load base + aligned adapter
↓
Evaluate lambda values
↓
Generate responses
↓
Safety
Capability
Diversity
Uncertainty
↓
Save results
↓
Plot Pareto frontier
```

## Notebook 04

```text
04_adaptive_controller_kaggle.ipynb
```

Pipeline:

```text
Prompt
↓
Creative probability
Safety probability
↓
Alignment controller
↓
Select lambda
↓
Generate response
↓
Evaluate
↓
Compare with baselines
```

---

# 29. Kaggle coding requirements

Every notebook must:

1. Detect whether GPU is available.
2. Print GPU name and VRAM.
3. Print package versions.
4. Use a configurable random seed.
5. Avoid assuming a specific GPU model.
6. Use `/kaggle/working/` for generated artifacts.
7. Save checkpoints where applicable.
8. Save CSV/JSON results.
9. Use mixed precision/quantization where supported.
10. Avoid unnecessary downloads.
11. Be restartable from a fresh session.
12. Fail clearly when a dependency or dataset cannot be loaded.
13. Keep model/dataset names configurable.

Kaggle notebook documentation:

https://www.kaggle.com/docs/notebooks

---

# 30. Repository structure

```text
align-pareto/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── classifier.yaml
│   ├── alignment.yaml
│   └── evaluation.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── notebooks/
│   ├── 01_classifier_kaggle.ipynb
│   ├── 02_alignment_training_kaggle.ipynb
│   ├── 03_lambda_sweep_kaggle.ipynb
│   └── 04_adaptive_controller_kaggle.ipynb
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── mapping.py
│   │   ├── cleaning.py
│   │   └── dedup.py
│   │
│   ├── classifier/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── alignment/
│   │   ├── train_dpo.py
│   │   ├── adapters.py
│   │   ├── lambda_controller.py
│   │   └── nullspace.py
│   │
│   ├── evaluation/
│   │   ├── safety.py
│   │   ├── capability.py
│   │   ├── creativity.py
│   │   ├── diversity.py
│   │   └── uncertainty.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── io.py
│
├── results/
└── reports/
```

---

# 31. Reproducibility

Every major experiment must save:

```text
seed
model_name
dataset source
dataset version when available
dataset size
train/validation/test sizes
hyperparameters
lambda
temperature
top_p
max_new_tokens
package versions
GPU information
experiment configuration
```

Save:

```text
experiment_config.json
metrics.json
predictions.csv
```

Also save raw generated responses for the key experiments.

---

# 32. Benchmark separation

Group B's existing-model benchmark is the main baseline.

When integrating results:

```text
Existing base model
Existing instruction/aligned model
        ↓
Group B benchmark
        ↓
baseline measurements
        ↓
Group A adaptive system
        ↓
same/compatible evaluation
        ↓
final comparison
```

Do not modify Group B's results to fit our hypothesis.

Use the same prompts/settings where possible.

---

# 33. Scientific questions

The final project should answer these questions:

### Q1
What measurable differences exist between existing base and instruction/aligned models?

### Q2
Does alignment create measurable changes in creative-response diversity?

### Q3
Does alignment affect uncertainty signals?

### Q4
Can we observe a safety–creativity/diversity trade-off as alignment strength changes?

### Q5
Can the `creative + safety-sensitive` classifier provide useful information for choosing alignment strength?

### Q6
Can a prompt-adaptive controller maintain a required safety floor while reducing unnecessary loss in creativity/diversity?

### Q7
Does the optional null-space approach reduce capability/creativity loss relative to ordinary alignment?

Mixed or negative results are valid if the evaluation is rigorous.

---

# 34. What is our actual novelty?

The project contribution should be stated carefully.

### Literature provides:

- alignment-tax measurement
- response-homogenization analysis
- value trade-off analysis
- safety-alignment mitigation ideas

### Our project investigates:

> **A lightweight prompt-adaptive alignment controller driven by two prompt attributes—creative intent and safety sensitivity—and evaluates whether this can navigate the observed safety–creativity/diversity trade-off.**

Do not claim novelty beyond what our experiments support.

---

# 35. Priority order

## MUST WORK

```text
Public datasets
↓
Creative + Safety labels
↓
Two-head classifier
↓
Classifier evaluation
```

## MUST WORK

```text
Base vs aligned model
↓
Benchmark
```

## MUST WORK

```text
lambda sweep
↓
Safety / creativity / diversity
↓
Pareto analysis
```

## MAIN NOVELTY

```text
Classifier
↓
Adaptive lambda controller
↓
Safety-constrained alignment
```

## ADVANCED

```text
Null-space alignment
```

## OPTIONAL

```text
Value-shift analysis
Streamlit dashboard
```

Do not delay the main project because an optional component is unfinished.

---

# 36. Exact first coding task for Antigravity

## Milestone 1 — Build the classifier dataset and classifier

Create:

```text
notebooks/01_classifier_kaggle.ipynb
```

and:

```text
src/data/download.py
src/data/mapping.py
src/data/cleaning.py
src/data/dedup.py
src/classifier/train.py
src/classifier/predict.py
src/classifier/evaluate.py
```

### The notebook must:

1. Detect Kaggle hardware.
2. Verify/install dependencies.
3. Load Dolly-15k and WildGuardMix first.
4. Optionally load a small, clearly identified Tulu subset.
5. Build `creative_label`.
6. Build `safety_label`.
7. Preserve source labels and source metadata.
8. Ensure all four creative/safety combinations occur.
9. Clean and deduplicate.
10. Create leakage-safe train/validation/test splits.
11. Train an `all-MiniLM-L6-v2` embedding model pipeline with two independent Logistic Regression heads.
12. Report per-label:
    - precision
    - recall
    - F1
    - ROC-AUC if available
    - PR-AUC if useful
13. Report macro-F1 across labels.
14. Produce separate confusion matrices for creative and safety.
15. Print representative false positives and false negatives.
16. Evaluate calibration where practical.
17. Save:
    - processed dataset
    - classifier
    - label maps
    - predictions
    - metrics
    - plots
    - dataset statistics
    - experiment config
18. Ensure the notebook can run from a fresh Kaggle session.

### Do not do yet

Do not implement:

- DPO
- alignment training
- lambda sweep
- adaptive controller
- null-space alignment
- Streamlit

The first milestone is only the **Creative + Safety classifier and its data pipeline**.

---

# 37. Antigravity behavior requirements

When implementing this project:

1. Inspect the repository first.
2. Do not overwrite working code without checking it.
3. Give a concise plan before large changes.
4. Prefer small modular changes.
5. Run smoke tests after major changes.
6. Never silently change the two-label taxonomy.
7. Never silently change evaluation metrics.
8. Keep dataset provenance.
9. Record assumptions.
10. Verify API compatibility with installed package versions.
11. Do not fabricate data, results, or citations.
12. Do not claim exact reproduction of a paper unless its protocol is actually reproduced.
13. Prefer reproducibility over cleverness.
14. Optimize for Kaggle runtime, GPU memory, and disk limits.
15. Keep raw results and configurations.

---

# 38. Current immediate action

### START ONLY HERE

Build the public-dataset pipeline and the two independent classifiers:

```text
                   PROMPT
                      │
             ┌────────┴────────┐
             ▼                 ▼
        CREATIVE HEAD     SAFETY HEAD
             │                 │
             ▼                 ▼
      probability         probability
```

Do not build the full project in the first step.

Once this works end-to-end, proceed to base-vs-aligned benchmarking and then lambda experiments.
