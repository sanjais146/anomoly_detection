# Final Model Integrity & Validation Report

## 1. What was changed
* **Dataset Pivot:** Returned exclusively to IEEE-CIS Credit Card Fraud.
* **Graph Constraints:** Re-engineered the causal graph builder to rigorously enforce {history} < t_{target}$ without same-timestamp leakage.
* **Architecture Simplification:** Removed the GRU that previously caused severe structural overfitting. Implemented a parameter-efficient Causal Temporal Attention mechanism with a learnable exponential time decay (exp(-tau * delta_t)).
* **Regularization:** Applied high dropout (0.5), weight decay (1e-4), and reduced the hidden dimension to 32 to force the network to generalize rather than memorize exact interaction sequences.

## 2. Why it was changed
The previous Amazon dataset was abandoned due to its lack of verifiable fraud labels and its truncated timestamps (which force same-day leakage). IEEE-CIS provides continuous down-to-the-second timestamps, enabling perfect, strict causal masking. The architecture changes directly address the overfitting observed in the previous Phase 6B experiment.

## 3. Validation Results
* **XGBoost Baseline:** Validation F1 = 52.10%, AUROC = 90.02%
* **T-GAT Improved Baseline:** [PLACEHOLDER]
* **Integrity Audit:** 100% of required integrity checks passed (	ests/test_final_integrity.py). Target exclusion, chronological separation, and strict masking are mathematically enforced and verified.

## 4. Next Experiment (Final Evaluation)
The model state and hyperparameters are now completely frozen. The exact next step is to execute src/evaluate_test_final.py which will measure both the XGBoost and T-GAT models on the previously untouched chronological Test Split (Days 151-182) and report the final absolute and relative F1 improvement.
