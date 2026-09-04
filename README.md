# IEEE-CIS E-Commerce Fraud Detection

## 1. Project Overview
This project constructs a deployment-realistic fraud detection system on the IEEE-CIS e-commerce dataset. We demonstrate that information availability (specifically the chargeback delay) fundamentally bounds model performance, and we provide a rigorously evaluated causal ensemble architecture.

## 2. Problem Statement
Many benchmark publications in fraud detection inadvertently allow future information or same-day label feedback to leak into the training process. This creates artificially high metrics that fail in production. This project addresses the challenge of building a system constrained by a strict 7-day label availability window.

## 3. Why Fraud Detection?
E-commerce fraud causes billions in losses annually. Accurate detection protects merchants and consumers, but machine learning models must adapt to rapid adversarial concept drift without violating causal timelines.

## 4. Dataset
**IEEE-CIS Fraud Detection** (Kaggle). A heavily anonymized dataset of financial transactions including card details, device information, and V-features.

## 5. Proposed Approach
We use an **Inductive Sequential Splitting** methodology. The dataset is ordered strictly by time. We enforce a `7-day chargeback delay` constraint, meaning a target label is only available for training 7 days after the transaction occurs.

## 6. Causal Information Availability
For any prediction at time `t`, the model only uses:
- Instantaneous transaction features at time `t`.
- Historical aggregations strictly bounded by `t_history < t`.
- Training labels where `t_label <= t - 7 days`.

## 7. Model Architecture
The final model is an **E10 Static Causal CatBoost Ensemble**. It averages predictions from:
- A depth=6 Base CatBoost model
- A depth=8 Deep CatBoost model
- A Weighted CatBoost model

## 8. Feature Engineering
Using Numba-accelerated aggregations, we generate delayed historical features (e.g., `c1_count_7d_delayed`) that mathematically exclude information inside the 7-day chargeback window, preserving deployment realism.

## 9. E10 Ensemble
The E10 static ensemble was frozen after extensive validation and evaluated exactly once on the chronological Test set to prevent test-set optimization.

## 10. Results
| Model | Test F1 |
| --- | --- |
| Original XGBoost | 49.69% |
| Original Hybrid T-GAT | 49.99% |
| E6 Delayed CatBoost | 56.48% |
| E10 Static CatBoost Ensemble | 62.25% |

## 11. Comparison with Baseline
The E10 Ensemble achieved a **+12.56 percentage point** improvement over the clean XGBoost baseline under identical causal constraints.

## 12. Why the 73% Benchmark Was Not Used as a Target
The external 73% benchmark (GNN-EADD) was investigated and found to be evaluated on a completely different Amazon product/seller dataset using a transductive node classification task. It is not apples-to-apples comparable to an inductive financial transaction stream. 

## 13. T-GAT Findings
Early experiments (E3/E8) demonstrated that applying temporal graph attention networks directly as dense embeddings into a tabular boosting model degraded performance, primarily because the tabular features already saturated the available causal signal.

## 14. Leakage Investigation
Our E13 ablations proved that artificially collapsing the 7-day delay to a 0-day delay instantly spiked Validation F1 to ~75%. This demonstrates that information availability has a substantial effect on F1. However, the external benchmark cannot be attributed to leakage without reproducing and auditing its original implementation.

## 15. Live Demo
A FastAPI + HTML/JS web application is included to demonstrate the inference pipeline. It simulates historical behavioral context based on manual input.

## 16. Google Colab Setup
1. Open `colab/run_demo.ipynb` in Google Colab.
2. Add your ngrok token to Colab Secrets as `NGROK_AUTHTOKEN`.
3. Run all cells.

## 17. ngrok Setup
The Colab notebook automatically bridges the local FastAPI server to a public URL using the `pyngrok` wrapper in a background thread.

## 18. API Documentation
- `GET /` : Serves the HTML dashboard.
- `GET /health` : Returns system health.
- `GET /model-info` : Returns frozen metrics.
- `POST /predict` : Accepts a JSON payload of transaction fields and returns fraud probability and risk signals.

## 19. Repository Structure
- `app/`: FastAPI backend and CatBoost predictor.
- `frontend/`: HTML/CSS/JS dashboard.
- `models/`: Frozen E10 CatBoost `.cbm` artifacts.
- `colab/`: Google Colab runner.
- `tests/`: Application and integrity tests.
- `release/historical/`: Archived experiments and reports.

## 20. Reproducibility
The objective was not to maximize leaderboard performance at the expense of information leakage. The objective was to construct and evaluate a deployment-realistic fraud detection system under strict chronological information constraints.

## 21. Limitations
- IEEE-CIS is a historical benchmark dataset.
- Live predictions are demonstration predictions, not real financial decisions.
- The 7-day delay is a simulation of chargeback availability.
- Real production deployment would require live feature stores and streaming infrastructure.
- Concept drift remains an important challenge.
- Model performance depends on feature availability.
- We cannot claim the external benchmark is definitively "leaked" without its implementation.

## 22. Future Work
While adaptive continual learning (E14) showed minor improvements via Recency Weighting, further research should investigate streaming graph neural networks capable of purely inductive, causal edge updates.
