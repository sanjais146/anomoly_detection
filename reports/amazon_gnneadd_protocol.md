# GNN-EADD Paper Protocol Audit

## 1. Paper Overview
**Methodology:** GNN-EADD (Graph Neural Network for E-commerce Anomaly Detection via Dual-stage learning).
The paper proposes a dual-stage framework for detecting anomalies in e-commerce networks (specifically Amazon review graphs).

## 2. Methodology Details
- **Dataset:** Amazon Review Graph (users reviewing products).
- **Nodes:** Users, Products.
- **Edges:** Reviews (Interactions).
- **Node Features:** Text embeddings of reviews, or structural features.
- **Edge Features:** Timestamps, ratings.
- **Anomaly Definition (Paper):** Anomalies are defined as spammers, fraudsters, or anomalous products (node-level classification) or fake reviews (edge-level classification). Often relies on injected anomalies or external spam-labeling heuristics if ground truth is absent.
- **Graph Construction:** Bipartite graph of User → Product.
- **Temporal Information:** Uses timestamps to order interactions.
- **Learning Protocol:** 
  - Stage 1: Contrastive self-supervised learning for robust node representations (handling distribution shift).
  - Stage 2: Supervised fine-tuning or anomaly scoring using the learned representations.
- **Evaluation:** Evaluated often in a transductive setting (entire graph structure is known, but labels are masked).

## 3. Our Implementation vs Base Paper

| Aspect | GNN-EADD Paper Protocol | Our TGAT Implementation |
|---|---|---|
| **Task** | Node/Edge Anomaly Classification | Edge Anomaly Detection (Link Reconstruction Error) |
| **Dataset** | Amazon (varies by sub-category) | Amazon Electronics (1.7GB reviews) |
| **Supervision** | Semi-supervised / Dual-stage | Self-supervised / Contrastive |
| **Temporal logic** | Dual-stage distribution handling | Strict causal `t_hist < t_target` temporal decay |
| **Leakage constraint**| Transductive graph often allowed | Strictly Inductive (no future edges) |

**Conclusion:** Our implementation is inspired by the dataset and contrastive approach of GNN-EADD, but strictly enforces our causal inductive boundary (no future edges). We use contrastive link prediction (reconstruction error) as the anomaly score, as the raw Amazon dataset lacks explicit ground-truth "fraud" labels. We **do not** claim exact reproduction of the 73% F1 score, as the tasks and protocols differ fundamentally (transductive vs causal inductive).
