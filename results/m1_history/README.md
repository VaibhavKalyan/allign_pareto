# Earlier M1 runs

`results/m1/` always holds the latest run. Each finished Kaggle run is kept here so numbers can be compared.

| Folder | Kaggle kernel version | Date | What changed |
|---|---|---|---|
| `kaggle_v1/` | `comanderq/align-pareto-m1-classifier` v1 (CPU) | 2026-11-09 | Revised label rules, held-out near-dup filter, recall-first safety threshold, per-source metrics, fictional-framing slice. Encoder: all-MiniLM-L6-v2, C = 2. |
| `kaggle_v2/` | v2 (T4 GPU) | 2026-11-09 | Encoder and per-head C chosen on validation (selected e5-base-v2, C = 8 for both heads), XSTest evaluation, bootstrap 95% CIs, 4 extra seeds. Its split differs from v1's, so shared eval prompts may be in its training split too (not checked: its training split was not kept). |
| `kaggle_v4/` | v4 (T4 GPU; v3 was cancelled) | 2026-11-09 | Safety head also trains on role-play prompts (safety labels only); C grid up to 64 (flat from C = 32); false-alarm breakdown. **126 of the 400 shared eval prompts (`data/eval/eval_prompts.csv`) were in this run's training split**, so do not use it for anything that touches the eval set. |

A teammate's re-run of the v1 code (`vaibhavkalyan/align-pareto-m1-classifier`, merged in PR #2) reproduced
`kaggle_v1/` exactly (same metrics to 4 decimals), so it is not stored separately.

The current `results/m1/` is Kaggle v5: v4's setup plus the shared eval set held out of the classifier data.
