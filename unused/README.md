# Unused code

Moved here when M2 became SFT only (2026-11-10). Nothing in `src/`, `tests/` or the notebooks imports it.
Kept so the DPO work (M2 v1, v2) can be traced or brought back. The imports inside still use the old
`src.alignment.*` paths, so move a file back before running it.

| File | What it was |
|---|---|
| `src/alignment/dpo_data.py` | hh-rlhf preference pairs for DPO (and the old data entry point; its SFT part now lives in `src/alignment/sft_data.py`) |
| `src/alignment/train_dpo.py` | DPO trainer (TRL `DPOTrainer`, reference policy = SFT model) |
| `src/alignment/adapters.py` | `set_peft_lora_lambda`: scales a LoRA adapter by λ |
| `src/alignment/models_dpo.py` | `load_aligned_model`: SFT model + DPO adapter |
| `tests/test_m2_data.py` | Unit tests for the hh-rlhf pair parsing |
| `configs/alignment_dpo.yaml` | The `dpo:` and `lambda_interpolation:` blocks from `configs/alignment.yaml` |
| `models.json` | Early list of Tulu 3 / Zephyr model families; never used by the code |
