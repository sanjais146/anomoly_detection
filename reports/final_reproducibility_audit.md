# Final Reproducibility Audit

## 1. Directory Structure
All dataset files exist correctly under `data/raw/` and `data/processed/`. The chronological splits strictly follow the IEEE-CIS train/validation/test design.

## 2. Methodology Integrity
- **Causal Boundaries:** All final models strictly enforce `t_history < t_target`.
- **Label Masking:** E6 features enforce `t_history <= t_target - 7 days`.
- **Threshold Selection:** Extracted on Val, mapped blindly to Test.
- **Preprocessing:** No global scaling or leakage. Missing values handled independently.

## 3. Discarded/Invalid Pipelines
- `causal_history_card_v1.parquet`: Invalidated during E4 audit due to 0-day validation leakage.
- `E2-B Checkpoints`: Maintained for methodological historical record only. Should not be deployed.

## 4. Final Saved State
The finalized E6 delayed-induction CatBoost model architecture is documented within `experiments/final_e6_test.py`.
