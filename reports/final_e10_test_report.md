# E10 Final Test Report

## Absolute Test Performance
- **Test F1:** 0.6226
- **Precision:** 0.7225
- **Recall:** 0.5469
- **AUROC:** 0.9378
- **AUPRC:** 0.6455
- **FPR:** 0.0076
- **Confusion Matrix:** TP=1703, TN=85558, FP=654, FN=1411
- **Frozen Threshold:** 0.404

## Comparisons
- **vs E6 (56.48%):** 5.78 percentage points
- **vs Clean XGBoost (49.69%):** 12.57 percentage points
- **vs Published GNN-EADD (73.00%):** -10.74 percentage points

## Authoritative Scientific Result
The mathematically sound, causality-preserving Test evaluation completed. 
The new authoritative Test F1 is **0.6226** (E10 Final Ensemble).
The invalidated 69.22% leakage result is rejected from deployment consideration. The 73.00% external benchmark was not matched under strict realistic conditions.

This represents the ceiling of the dataset under absolute chronological deployment honesty.
