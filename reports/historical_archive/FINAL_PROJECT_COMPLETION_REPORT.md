# FINAL PROJECT COMPLETION REPORT

## PROJECT:
Amazon E-Commerce Anomaly Detection using TGAT

## DATASET:
Amazon Electronics Reviews (1.77 GB).
Extracted: Users, Products, Reviews (Timestamps, Ratings).

## GRAPH:
- **Nodes**: Users (eviewerID), Products (sin).
- **Edges**: Chronological Reviews (with unixReviewTime).
- **Constraint**: Bipartite graph with no external heuristic data required.

## MODEL:
- **Architecture**: HybridAmazonModel with CausalAmazonEncoder.
- **Temporal Handling**: Strict causal 	_hist < t_target masking with learned exponential decay τ.

## ANOMALY METHOD:
- **Type**: Self-Supervised Link Reconstruction Error.
- **Logic**: Genuine interactions act as positives; synthetically sampled counterfactual interactions act as anomalies. The model predicts the probability of the link based on temporal context similarity.

## RESULT:
- **Amazon TGAT F1**: 77.01% (Independently trained and verified, no leakage detected).
- **AUROC**: 0.7997.

## IEEE-CIS BASELINE:
- **E10 Test F1**: 62.25%.
- **Status**: Preserved as a standalone parallel baseline for supervised transaction fraud.

## PAPER COMPARABILITY:
The GNN-EADD paper (reporting ~73%) evaluates using transductive protocols with external spam labels. Our implementation uses a strict causal, inductive link-prediction protocol. A direct "beat" is not claimed; instead, we demonstrate high efficacy (77.01%) under our more stringent temporal constraints.

## VERIFICATION STATUS:
- **LEAKAGE**: PASS (Chronological splitting and strict edge-time masking confirmed).
- **TGAT INFERENCE**: PASS (Forward pass and temporal attention proven via 	est_amazon_system.py).
- **FASTAPI**: PASS (Endpoints /predict/amazon and /health are functional and tested).
- **DASHBOARD**: PASS (Rebuilt to display Amazon graph logic, metrics, and live inference context).
- **COLAB**: PASS (colab/run_demo.ipynb successfully clones, loads checkpoint, and exposes API).
- **NGROK**: PASS (Configured securely via Colab Secrets, no hard-coded tokens).
- **FRESH CLONE**: PASS (Tested by removing local temp artifacts and relying on Git/LFS pulls).
- **SECURITY**: PASS (No API keys or personal paths found).
- **TESTS**: PASS (Automated pytest suite runs perfectly).
- **GITHUB**: PASS (Final models committed via Git LFS).

## FINAL COMMIT:
The repository is completely finalized, pushed, and ready for defense.
