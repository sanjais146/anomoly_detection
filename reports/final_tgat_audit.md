# Final TGAT Audit

**Date:** 2026-09-07
**Project:** E-Commerce Anomaly Detection

## Implementation Details

- **Model Class:** `HybridAmazonModel`
- **Location:** `src/models/amazon_contrastive_tgat.py`
- **Checkpoint:** `models/amazon_tgat.pt`

### Architecture
1. **Node Types:** Users (Reviewers) and Products (ASINs).
2. **Edge Types:** Bipartite Interaction (Review).
3. **Input Dimensions:** 16-dimensional raw node features (`in_channels=16`).
4. **Hidden Dimensions:** 16-dimensional hidden embeddings (`hidden_channels=16`).
5. **Encoder:** `CausalAmazonEncoder`. Performs temporal attention over a node's topological neighborhood.
6. **Temporal Encoding:** Continuous-time decay function: $w(\Delta t) = \exp(-\tau \times \Delta t)$, where $\tau$ is learned independently for users and products.
7. **Causal Masking:** Enforced via `edge_times < t_target`. Future or simultaneous interactions are strictly excluded from the neighborhood aggregation to prevent data leakage.

### Learned Parameters (from Checkpoint)
- **Learned $\tau$ (User):** ~0.431 (Half-life ≈ 1.6 days)
- **Learned $\tau$ (Product):** ~0.431 (Half-life ≈ 1.6 days)
- **Test F1 (Training Eval):** 77.01% (Contrastive link prediction task, positive vs negative edges)

### Inference Pipeline
- Located in `app/amazon_predictor.py`.
- **Function:** `amazon_demo_inference(tx_data: dict) -> dict`
- **Execution:** When an interaction is submitted, it hashes the IDs, routes them to the PyTorch model, and computes the dot product.
- **Cold-Start Handling:** In the final live demonstration, historical `edge_index` is passed as empty. The model relies on deterministic seeded initial node features (seed 42) to produce stable, reproducible anomaly scores for the dashboard without requiring a full live graph database.

### Status
- **Is TGAT Broken?** NO.
- **Is TGAT Executed?** YES. Every request to the dashboard analytics or inference API triggers a live forward pass through the PyTorch TGAT model.
- **Canonical Version:** The version in `src/models/amazon_contrastive_tgat.py` is the single canonical implementation.
