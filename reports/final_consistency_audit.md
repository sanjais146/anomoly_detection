# Final Project Consistency Audit

**Date:** 2026-09-09
**Project:** E-Commerce Anomaly Detection using TGAT + E10

## 1. Codebase Reality Check

| CLAIM | ACTUAL IMPLEMENTATION | CONSISTENT? | REQUIRED FIX |
| :--- | :--- | :--- | :--- |
| **Combined TGAT+E10 Pipeline** | The code executes two *separate* tracks: TGAT for Amazon (unsupervised link anomalies) and E10 for IEEE-CIS (supervised tabular fraud). TGAT embeddings are *not* fed into E10. | NO | Document the actual parallel architecture. Do not claim a merged pipeline. |
| **Model Used for Inference** | `HybridAmazonModel` (TGAT) handles `/predict/amazon`. E10 handles `/predict/ieee`. | YES | None |
| **TGAT Implementation** | `src/models/amazon_contrastive_tgat.py` implements a Causal Temporal Graph Attention Network. | YES | None |
| **TGAT Execution** | `amazon_demo_inference_batch` natively executes `model(u_idx, p_idx, t_target...)`. | YES | None |
| **Graph Construction** | Bipartite graph constructed from Amazon Electronics reviews. | YES | None |
| **Node Representation** | Users (`reviewerID`) and Products (`asin`). | YES | None |
| **Edge Representation** | Reviews (interaction events). | YES | None |
| **Timestamp Usage** | `t_target` and `edge_times` enforce causal masking (`edge_times < t_target`). | YES | None |
| **TGAT Features** | 16-d embeddings (deterministic pseudo-features for cold-start demo mode). | YES | None |
| **E10 Features** | Standard tabular transaction features (`TransactionAmt`, `card1`, etc.). | YES | None |
| **Anomaly Scoring** | TGAT: `1.0 - sigmoid(Eu · Ep)` (Link reconstruction error). | YES | None |
| **Dashboard Display** | Live visualization of TGAT anomaly distributions and E10 baselines. | YES | Update terminology to clearly separate Anomaly vs Fraud. |
| **Colab Execution** | `colab/run_demo.ipynb` clones repo, runs Uvicorn, exposes via Ngrok. | YES | Verify notebook works cleanly from a fresh runtime. |

## 2. Terminology Corrections

*   **Anomaly vs. Fraud:** Anomaly is a structurally unexpected link (TGAT score). Fraud is a confirmed malicious event with a ground-truth label (E10). We do not claim every anomaly is fraud.
*   **73% Benchmark:** The 73% GNN-EADD reference is not an "absolute ceiling". It is an external baseline evaluated on a different (transductive) protocol. Our 62.25% E10 Test F1 is our validated baseline under strictly causal evaluation.

## 3. Conclusion
The repository genuinely implements and executes TGAT for unsupervised E-Commerce anomaly detection, while preserving E10 as a historical supervised baseline. The documentation and dashboard will be aligned strictly to this reality.
