# Earlier M1 runs

`results/m1/` always holds the latest run. Each finished Kaggle run is kept here so numbers can be compared.

| Folder | Kaggle kernel version | Date | What changed |
|---|---|---|---|
| `kaggle_v1/` | `comanderq/align-pareto-m1-classifier` v1 (CPU) | 2026-11-09 | Revised label rules, held-out near-dup filter, recall-first safety threshold, per-source metrics, fictional-framing slice. Encoder: all-MiniLM-L6-v2, C = 2. |
| `kaggle_v2/` | v2 (T4 GPU) | 2026-11-09 | Encoder and per-head C chosen on validation (selected e5-base-v2, C = 8 for both heads), XSTest evaluation, bootstrap 95% CIs, 4 extra seeds. |

The current `results/m1/` is Kaggle v4 (v3 was cancelled before finishing): the safety head also trains on the
role-play prompts (safety labels only), C grid widened to 64 (validation ROC-AUC is flat from C = 32 on), and a
false-alarm breakdown on harmless test prompts is reported.
