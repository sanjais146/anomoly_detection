# Final Architecture Document

**Date:** 2026-09-09
**Project:** E-Commerce Anomaly Detection using TGAT + E10

This document strictly defines the exact architectures executed by the final API and Dashboard.

## 1. Primary Track: Amazon E-Commerce Anomaly Detection
*Focus: Unsupervised Structural Link Anomalies*

### Pipeline
1.  **Frontend / Dashboard:** A Vercel-inspired UI requesting `/analytics/amazon/anomalies`.
2.  **FastAPI (`app/main.py`):** Receives the `AmazonInteractionInput` (Reviewer ID, ASIN, Timestamp, Rating).
3.  **Data Preprocessing:** Inputs are deterministically hashed (MD5 mod 10000) to retrieve standard cold-start structural embeddings.
4.  **Temporal Graph Attention Network (`HybridAmazonModel`):**
    *   **Causal Masking:** Strict $t_{hist} < t_{target}$ boundary enforced.
    *   **Temporal Decay:** Exponential decay $w = \exp(-\tau \cdot \Delta t)$.
    *   **Aggregation:** Node representations are dynamically updated based on recent chronological graph topology.
5.  **Scoring (`amazon_demo_inference_batch`):** Anomaly Score = $1.0 - \sigma(E_u \cdot E_p)$. High scores indicate structural anomalies.
6.  **Response:** Returns anomaly probability, similarities, and learned temporal decay half-lives to the dashboard.

## 2. Historical Baseline Track: IEEE-CIS Supervised Fraud
*Focus: Supervised Tabular Classification*

### Pipeline
1.  **Frontend / Dashboard:** A dedicated baseline section comparing historical benchmark metrics.
2.  **FastAPI (`app/main.py`):** Receives `TransactionInput` via `/predict/ieee`.
3.  **Data Preprocessing:** Traditional feature engineering (tabular aggregation, no graph embeddings).
4.  **E10 CatBoost Ensemble:** Evaluates standard static and causal aggregate features.
5.  **Scoring:** Returns supervised `fraud_probability` calibrated via traditional ML methodologies.
6.  **Response:** Used strictly as a historical reference metric (62.25% F1) for project context, distinct from the Amazon TGAT unsupervised track.

## 3. The 7-Day Causal Boundary
In our evaluation (specifically relevant to E10 and TGAT evaluation protocol):
*   Information is strictly bounded.
*   **Transaction Information:** Senders, receivers, amounts.
*   **Historical Information:** Only events strictly prior to $t_{target}$ are visible.
*   **Prohibited Information:** Future interactions or labels ($t \ge t_{target}$) cannot influence the embedding of a node at time $t_{target}$.

## 4. Deployment Infrastructure
*   **Google Colab:** Single-click execution container (`run_demo.ipynb`).
*   **Git LFS:** Stores frozen model checkpoints (`amazon_tgat.pt`, `e10_base.cbm`).
*   **Ngrok:** Securely tunnels local FastAPI instance to the public internet for dashboard demonstrations.
