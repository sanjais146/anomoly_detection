# Project Reproduction Guide

## 1. Required Python Version
Python 3.10+ is required.

## 2. Required Packages
```text
numpy
pandas
catboost
scikit-learn
numba
matplotlib
seaborn
pytest
```

## 3. Dataset Preparation
Download the IEEE-CIS Fraud Detection dataset from Kaggle.
Place `train_transaction.csv` and `train_identity.csv` in `data/raw/`.

## 4. Integrity Tests
Run the boundary verification tests to prove causality:
```bash
python release/final/tests/test_e6_delayed_induction.py
```

## 5. Feature Generation, Training, Validation
Execute the $\Delta$-induction runner. This will load the raw data, generate 431 base features and 30 delayed-causal features, and train the CatBoost model, extracting the threshold from the validation set:
```bash
python release/final/experiments/e6_delayed_induction.py
```

## 6. Final Evaluation
Evaluate exactly once on the Test Set:
```bash
python release/final/experiments/final_e6_test.py
```
*Note: Expected Test F1 is 56.48%.*
