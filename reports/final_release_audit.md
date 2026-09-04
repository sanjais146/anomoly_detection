# Final Release Audit & Reproducibility Instructions

## Environment Requirements
- **Python Version:** 3.10+
- **GPU Requirements:** NVIDIA GPU (e.g., GTX 1650 4GB or better)
- **Package Dependencies:** `numpy`, `pandas`, `catboost`, `scikit-learn`, `numba`, `matplotlib`, `seaborn`, `pytest`.

## Dataset Requirements
- `train_transaction.csv`
- `train_identity.csv`
- Place under `data/raw/`.

## Execution Commands

### 1. Integrity Tests
Run the delayed-label causal boundary verification:
```bash
python release/final/tests/test_e6_delayed_induction.py
```
*Expected Output:* `All E6 delayed-label induction tests passed.`

### 2. Feature Generation & Training
The E6 pipeline dynamically processes raw data, applies the 7-day chargeback delay via Numba sliding windows, and trains CatBoost:
```bash
python release/final/experiments/e6_delayed_induction.py
```

### 3. Evaluation
To run the final model on the sequestered Test Set:
```bash
python release/final/experiments/final_e6_test.py
```

## Expected Final Metrics
- **Test F1:** 56.48%
- **Test Precision:** 67.02%
- **Test Recall:** 48.81%
- **Test AUROC:** 0.9236
- **Test AUPRC:** 0.5847

## Known Limitations
The preprocessing pipeline strictly drops string categorical variables with massive cardinalities without applying target encoding, as target encoding risks future leakage. Expanding the feature pipeline with strict inductive embeddings could yield further improvements.
