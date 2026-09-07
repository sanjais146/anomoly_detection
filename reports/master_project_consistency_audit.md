# Master Project Consistency Audit

**Date:** 2026-09-07
**Project:** E-Commerce Anomaly Detection using Temporal Graph Attention Networks

This audit verifies that the project is internally consistent, scientifically defensible, and adheres strictly to the constraints of the final directive.

## A. What is the current canonical dataset?
**Amazon Electronics.** This dataset represents real e-commerce user-product review interactions, providing a genuine topological structure and temporal timestamps required to train and evaluate a Temporal Graph Attention Network (TGAT). 

## B. What is the intended final project domain?
**E-Commerce Anomaly Detection.** The project operates specifically within the e-commerce domain, analyzing temporal and topological patterns of interactions (reviews) to detect structural anomalies (e.g., unexpected link formation).

## C. What exactly is the anomaly being detected?
A **Topological and Temporal Anomaly.** Specifically, it is defined as a user-product interaction (link) that has a very low predicted probability of occurring given the historical temporal neighborhood of both the user and the product. The model evaluates whether the causal past predicts the current interaction. If it does not, the interaction is anomalous.

## D. Which model is the final anomaly model?
**HybridAmazonModel (TGAT).** Located at `models/amazon_tgat.pt`. This is a PyTorch-based temporal graph attention network consisting of a `CausalAmazonEncoder` and a dot-product interaction scorer.

## E. Is TGAT currently actually executed?
**YES.** The FastAPI backend (`app/amazon_predictor.py` and `app/analytics_anomalies.py`) loads the `amazon_tgat.pt` checkpoint. When the `/predict/amazon` or `/analytics/amazon/anomalies` endpoints are called, the code actively passes tensors through the TGAT PyTorch model and computes the anomaly probability.

## F. Which model produced the 62.25% Test F1?
**CatBoost Ensemble (E10).** This is a static, supervised machine learning model.

## G. Which dataset produced that score (62.25%)?
**IEEE-CIS Fraud Detection.** This was a supervised classification task where ground-truth fraud labels (isFraud) existed.

## H. Which dataset is currently shown by the dashboard?
**Amazon Electronics (100-record sample).** The dashboard uses `data/amazon/raw/sample_reviews.json`, which contains real Amazon Electronics interactions.

## I. Are Amazon Electronics and IEEE-CIS being accidentally mixed?
**NO.** The frontend and backend explicitly separate the systems. The final dashboard operates entirely on the Amazon TGAT pipeline. The 62.25% E10 CatBoost result is kept strictly isolated as a historical supervised fraud classification benchmark on the "Research Metrics" and "Overview" pages.

## J. Which artifacts are historical only?
- The IEEE-CIS dataset processing scripts.
- The E10, E9, etc. CatBoost checkpoints.
- The `isFraud` classification metrics.
These serve as evidence of the research progression (from static supervised fraud classification to unsupervised temporal graph anomaly detection).

## K. Which artifacts belong in the final deployment?
- `models/amazon_tgat.pt`
- `app/main.py`, `app/amazon_predictor.py`, `app/analytics_anomalies.py`, `app/analytics.py`
- `data/amazon/raw/sample_reviews.json`
- `colab/run_demo.ipynb`
- `frontend/index.html`, `frontend/style.css`, `frontend/script.js`
- `src/models/amazon_contrastive_tgat.py`
