# Final E6 Test Integrity Audit

## 1. Zero Future Leakage
The feature generation function `generate_e6_features` explicitly processes transactions in chronological order. A strict condition `t_hist < t_target` prevents any feature from including the target transaction itself or any subsequent transactions.

## 2. Zero Same-Timestamp Leakage
Because the chronological threshold is strictly less than (`<`), no batch of transactions processed at the exact same timestamp can mutually influence one another.

## 3. Strict Delayed Label Bounds (7-Day $\Delta$)
Test labels are NEVER used for hyperparameter tuning, threshold selection, or architectural decisions. Furthermore, within the feature engineering phase, a label from $t_{hist}$ is only mathematically visible to a transaction at $t_{target}$ if:
$$t_{hist} \leq t_{target} - 7 \text{ days}$$
This was independently verified using the synthetic transaction `pytest` suite prior to running the final test, guaranteeing that the model realistically simulates a 7-day chargeback delay.

## 4. One-Shot Evaluation
The test set (`y[te]`) was invoked exactly ONCE in the script `experiments/final_e6_test.py` purely to record the confusion matrix and compute the F1 score. No subsequent training loops, parameter sweeps, or threshold adjustments were executed. The classification threshold was frozen entirely on the Validation set.

## 5. Architectural Record
- **Model:** CatBoostClassifier (`iterations=600`, `learning_rate=0.1`, `depth=6`, `early_stopping=50`).
- **Features:** 431 Base IEEE-CIS Features + `card1` 7d-causal + `card1_addr1` 7d-causal + `card1_DeviceInfo` 7d-causal.
- **Train/Val/Test Split:** Standard chronological 80/20 train/val/test boundary (Days 0-120 Train, 121-150 Val, 151-182 Test).
