# Final Frozen Test Evaluation (E2-B)

## 1. Evaluation Protocol
The exact E2-B CatBoost model (Full Features + Causal Card History) was frozen and evaluated strictly once on the chronological test set (days 150-182). 

**Integrity Assertions Verified:**
- Strict historical timeline (`t_history < t_target`).
- Zero same-timestamp connections.
- Zero feature preprocessing leakage (scalers fitted only on train).
- Classification threshold chosen **exclusively** on the Validation Set (`0.8226`).

## 2. Test Set Metrics
- **Test F1 Score**: 69.22%
- **Test AUROC**: 0.9582
- **Test AUPRC**: 0.7299
- **Test Precision**: 74.03%
- **Test Recall**: 65.00%
- **Test False Positive Rate (FPR)**: 0.82%

**Confusion Matrix (Test Set: 89,326 samples):**
- True Negatives (TN): 85,502
- False Positives (FP): 710
- False Negatives (FN): 1,090
- True Positives (TP): 2,024

## 3. Comparison to Benchmarks

| Model | Validation F1 | Untouched Test F1 | Status vs Benchmark |
| :--- | :--- | :--- | :--- |
| **GNN-EADD Base Paper** | -- | **73.00%** | Benchmark |
| **Original XGBoost Baseline** | 52.10% | 49.69% | Did not beat |
| **Original Hybrid T-GAT** | 50.50% | 49.99% | Did not beat |
| **Final Causal CatBoost (E2-B)** | 71.21% | **69.22%** | **Did not beat** |

## 4. Conclusion
While the E2-B causal tabular model achieved a massive ~20 absolute percentage point improvement over our initial 49% XGBoost baseline, **it did not surpass the 73.00% F1 benchmark reported in the base paper.** The final untouched Test F1 is 69.22%.
