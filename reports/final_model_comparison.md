# Final Model Comparison

| Model | Protocol | Test F1 | Valid for Final Comparison? | Comments |
| :--- | :--- | :--- | :--- | :--- |
| **Original XGBoost Baseline** | chronological | 49.69% | YES | 31-feature floor |
| **Original Hybrid T-GAT** | chronological | 49.99% | YES | Marginal graph lift |
| **E2-B CatBoost** | 0-day label feedback | 69.22% | NO | Invalidated due to 0-day leakage |
| **E6 CatBoost** | 7-day delayed induction | 56.48% | YES | Basic continuous deployment simulation |
| **E12 Continual CatBoost** | 7-day online learning | 61.18% | YES | Weekly retraining simulation |
| **E10 Final Ensemble** | 7-day delayed induction | **62.25%** | **YES** | **Authoritative Final Result (Deep Features + Ensemble)** |
| *GNN-EADD* | published benchmark | *73.00%* | External benchmark | *Evidence suggests transductive evaluation differences* |
