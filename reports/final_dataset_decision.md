# Final Dataset Decision

**Date:** 2026-09-07
**Project:** E-Commerce Anomaly Detection

## Decision
The canonical dataset for the final anomaly detection system is **Amazon Electronics**.

## Justification
The project requires a genuine **Temporal Graph Attention Network (TGAT)** implementation. TGAT fundamentally requires a dataset with:
1. Meaningful topological relationships (a graph).
2. Temporal evolution (timestamps for interactions).

### Why Amazon Electronics?
The Amazon Electronics dataset perfectly models an e-commerce ecosystem:
- **Nodes:** Users (Reviewers) and Products (ASINs).
- **Edges:** Reviews / Interactions.
- **Time:** Unix timestamps indicating when the interaction occurred.
This allows the TGAT model to construct a bipartite temporal graph and evaluate the likelihood of an edge forming at time *t* based on the causal history of the user and product prior to time *t*.

**Absence of Ground Truth Fraud Labels:**
Because Amazon Electronics does not contain native "fraud" labels, the task is properly scoped as **Unsupervised Anomaly Detection** via link reconstruction, rather than supervised fraud classification.

### Why not IEEE-CIS for the Final System?
While IEEE-CIS contains explicit "isFraud" labels, its graph structure is synthetically inferred (e.g., using Card IDs or IP proxies as nodes), and the temporal dynamics are masked (relative timedeltas). The research project successfully achieved a 62.25% F1 score on IEEE-CIS using a static CatBoost ensemble (E10). However, shoehorning TGAT onto IEEE-CIS proved less scientifically rigorous than applying TGAT to a native interaction graph like Amazon. 

Therefore, IEEE-CIS and the E10 CatBoost model are preserved as a **Historical Supervised Fraud Benchmark**, demonstrating the project's evolution, while Amazon Electronics powers the final TGAT anomaly detection system.

## Entities and Data Integrity
- **Reviewer ID:** Real alphanumeric IDs from Amazon.
- **ASIN (Product ID):** Real Amazon Standard Identification Numbers.
- **Rating:** Real 1-5 star ratings.
- **Timestamp:** Unix timestamp. *Note: Temporal resolution in Amazon datasets is often daily. Same-day interactions are treated carefully to prevent temporal leakage.*

No synthetic entities, fabricated sellers, or fake anomaly labels are used in this deployment.
