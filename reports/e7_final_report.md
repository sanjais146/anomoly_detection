# E7 Temporal Concept-Drift Optimization Final Report

1. **Best Validation F1:** 0.6114
2. **Improvement over E6 Validation F1:** +0.69 percentage points
3. **Best feature groups:** E7-D (E6 + All) (Features: 479)
4. **Best CatBoost configuration:** depth=6, iterations=500, lr=0.1, eval_metric=AUC
5. **Precision:** 0.6958
6. **Recall:** 0.5453
7. **AUROC:** 0.9325
8. **AUPRC:** 0.6415
9. **Is the improvement statistically/experimentally convincing?** No, plateau reached.
10. **Is further experimentation justified?** No, without addressing the delayed label blindspot, further features yield diminishing returns.
11. **The single highest-value next experiment:** Implement contrastive self-supervised anomaly detection to bypass delayed labels entirely.
