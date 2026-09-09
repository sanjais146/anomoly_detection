# Authoritative Results

**Project:** E-Commerce Anomaly Detection using TGAT + E10

This document serves as the single authoritative source for the project's validated model performance.

## 1. Primary Track: Amazon E-Commerce Anomaly Detection
**Model:** Temporal Graph Attention Network (TGAT) - `HybridAmazonModel`
**Dataset:** Amazon Electronics Reviews
**Task:** Unsupervised Structural Link Reconstruction (Anomaly Scoring)
**Evaluation Protocol:** Causal Inductive (Future edges strictly masked: $t_{hist} < t_{target}$)

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Test F1** | 77.01% | F1 score on the hold-out chronological test set. |
| **Test Precision** | 71.83% | Precision of flagged anomalies. |
| **Test Recall** | 83.00% | Recall of synthetic test anomalies. |
| **Test AUROC** | 0.7997 | Area under the ROC curve. |
| **Test AUPRC** | 0.7821 | Area under the Precision-Recall curve. |
| **Learned $\tau$ (User)**| 0.431 | Extracted temporal decay factor (Half-life ≈ 1.6 days). |
| **Learned $\tau$ (Prod)**| 0.431 | Extracted temporal decay factor (Half-life ≈ 1.6 days). |
| **Training Threshold** | 0.5200 | Optimal threshold derived from full-graph training phase. |
| **Demo Threshold** | 0.6668 | Statically calibrated threshold for the 100-record cold-start visualization. |

*Note: The 73% GNN-EADD reference from external literature utilizes a different evaluation protocol (transductive propagation with external spam labels) and is not directly comparable to our strictly causal inductive pipeline.*

---

## 2. Baseline Track: IEEE-CIS Transaction Fraud Detection
**Model:** E10 Static Causal CatBoost Ensemble
**Dataset:** IEEE-CIS Transactions
**Task:** Supervised Fraud Classification (`isFraud`)
**Evaluation Protocol:** Strict 7-day chronological causal boundary.

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Test F1** | 62.25% | The best validated E10 Test F1 obtained under our defined evaluation protocol. |
| **Baseline F1**| 49.69% | Performance of the baseline XGBoost approach. |
| **Improvement**| +12.56 pp | Percentage point improvement over the baseline. |

*Note: The 62.25% is the best validated E10 Test F1 obtained under our defined evaluation protocol. TGAT Test performance for the supervised IEEE-CIS task has not been established. The E10 result serves purely as a historical supervised benchmark for the project's evolution.*
