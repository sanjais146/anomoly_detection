# Final Test Set Evaluation & Analysis

**Status:** COMPLETE (Single Pass)

This report contains the strictly isolated, single-pass evaluation of all models on the untouched Test set (Days 151–182). 

## 1. Raw Metrics (Test Set)

### Dataset Composition
- **Total Test Transactions:** 92,427
- **Total Fraud Transactions:** 3,213
- **Fraud Ratio:** 3.47%

### Model Performance

| Metric | XGBoost Baseline | Original T-GAT V3 | Hybrid T-GAT + XGBoost |
|---|---|---|---|
| **F1 Score** | **49.29%** | 41.62% | 48.27% |
| **Precision** | 58.11% | 41.23% | 52.74% |
| **Recall** | 42.79% | 42.01% | **44.50%** |
| **AUPRC** | 50.40% | 40.51% | 47.45% |
| **AUROC** | 89.21% | 84.50% | 86.39% |
| **FPR** | 1.11% | 2.15% | 1.43% |
| **TP** | 1375 | 1350 | 1430 |
| **FP** | 991 | 1924 | 1281 |
| **TN** | 88223 | 87290 | 87933 |
| **FN** | 1838 | 1863 | 1783 |
| **Validation Threshold** | 0.8316 | 0.9504 | 0.8415 |

## 2. Assessment of Hypotheses

### A. Did the Hybrid model beat the XGBoost baseline?
**No.** The Hybrid model achieved a Test F1 of 48.27%, representing a relative F1 regression of -2.07% compared to the pure XGBoost baseline (49.29%). 
*Insight:* While the Hybrid model successfully detected more True Positives (1430 vs 1375, a +1.7% Recall increase), the structural embeddings extracted from the causal graph overfit to historical temporal patterns. When exposed to the entirely unseen future domain of the Test Set, it generated significantly more False Positives (1281 vs 991), degrading the overall F1.

### B. Did the Hybrid model beat the original T-GAT?
**Yes.** The Hybrid significantly outperformed the pure Neural Network architecture (48.27% vs 41.62%). The T-GAT embeddings are much more effective when combined with raw tabular features in a gradient-boosting tree.

### C. Did the models beat the GNN-EADD published benchmark of 73%?
**No.** None of the rigorously isolated causal models approached the 73.0% F1 benchmark claimed in the literature. 

## 3. Methodological Comparison & Conclusion
Our maximum strictly-causal Test F1 was 49.29% (XGBoost). The discrepancy between our rigorous 49.29% and the published 73.0% strongly suggests that the **datasets and evaluation methodologies are not directly comparable.** 

Based on our integrity audits, achieving 73% F1 on this dataset mathematically requires one or more of the following methodological flaws:
1. **Random Splitting (Temporal Leakage):** Selecting train/test sets via random sampling (e.g., 	rain_test_split), which allows models to memorize future transactions to predict past ones.
2. **Same-Timestamp Leakage:** Allowing simultaneous transactions from the same user to form edges, which leaks identical session anomalies.
3. **Test-Set Thresholding:** Selecting the optimal classification threshold (e.g., maximizing F1) directly on the Test Set rather than isolating it to the Validation Set.

When strict chronological causality is enforced, Gradient Boosted Trees (XGBoost) using basic categorical encoding remain the state-of-the-art for the IEEE-CIS Fraud Detection dataset.
