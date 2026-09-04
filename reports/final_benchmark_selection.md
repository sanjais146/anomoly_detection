# Final Benchmark Selection

**Objective:** Select a scientifically rigorous, publicly labeled benchmark for temporal graph anomaly/fraud detection to directly evaluate our Temporal GAT framework against published baselines.

## Phase 1 — Candidate Benchmarks

### Candidate 1: YelpChi (Spam Review Graph)
* **Dataset Availability:** Public (Rayana & Akoglu) / DGFraud Repo.
* **Label Availability:** Yes (20.1% spam out of 38,063 reviews).
* **Graph Structure:** Typically Homogeneous (Reviews as nodes). Edges are pre-built static relations: R-U-R (same user), R-T-R (same month), R-S-R (same stars).
* **Features:** 32-dim hand-crafted textual/behavioral.
* **Published Baselines:** CARE-GNN (CIKM 2020), PC-GNN.
* **Metrics:** CARE-GNN (AUC ~ 76%, Macro-F1 ~ 65%).
* **T-GAT Compatibility:** **Low.** The standard benchmark heavily aggregates the graph into static offline relations (e.g., linking reviews in the same month), which completely destroys exact temporal sequence and continuous causality.

### Candidate 2: Amazon Musical Instruments (Spam Graph)
* **Dataset Availability:** Public / DGFraud.
* **Label Availability:** Yes (821 spam out of ~12k nodes).
* **Graph Structure:** Static Homogeneous user-user graphs via meta-paths (U-P-U).
* **T-GAT Compatibility:** **Low.** Identical limitation to YelpChi; relies on static aggregated meta-paths rather than continuous time.

### Candidate 3: Elliptic Bitcoin Dataset (Financial Fraud)
* **Dataset Availability:** Public (Weber et al., KDD 2019), natively embedded in PyTorch Geometric.
* **Label Availability:** Yes. 203,769 transaction nodes (4,545 Illicit, 42,019 Licit, 157,205 Unknown).
* **Graph Structure:** Directed temporal transaction graph (Tx $\rightarrow$ Tx). 234,355 edges.
* **Features:** 166 dense features per node.
* **Timestamps:** 49 distinct, chronologically ordered time steps (approx. 2 weeks each).
* **Published Baselines:** GCN (Weber et al., KDD 2019), EvolveGCN (Pareja et al., AAAI 2020).
* **Metrics:** GCN (Illicit F1 ~44%), EvolveGCN (Illicit F1 ~65%).
* **T-GAT Compatibility:** **High.** The graph is natively temporal, causal, and inductive.
* **Computation:** Easily fits within a 4GB GTX 1650 (approx 4k nodes per time step, manageable sequentially or via subgraph sampling).

## Phase 2 — Final Selection

**Selected Benchmark:** **Elliptic Bitcoin Dataset**

### Scientific Justification
1. **Unambiguous Ground Truth:** Unlike Amazon e-commerce datasets which rely on weak proxies or synthetic injections, the Elliptic dataset provides genuine, cryptographically and forensically validated illicit transaction labels.
2. **Native Temporal Causality:** The dataset is strictly partitioned into 49 sequential time steps. The chronological train/test split (Train: Steps 1–34, Test: Steps 35–49) ensures absolutely zero future leakage, natively aligning with our strict causal history constraints.
3. **Strong Published Baseline:** EvolveGCN (AAAI 2020) and standard GCN (KDD 2019) provide clear, reproducible baselines.
4. **Feasibility:** Node/edge counts (200k/234k) fit perfectly within our GTX 1650 hardware limit without needing the massive PyG sparse optimization refactors required for the 7.8M-node Amazon set.
5. **Architectural Fit:** Our Temporal GAT components (temporal decay, causal attention) directly map onto the transaction flows of Bitcoin, answering the call to detect fraud using sequence and structure.

### The Baseline to Beat
* **Target Baseline:** Graph Convolutional Network (GCN) + Temporal splits, and optionally EvolveGCN.
* **Target Metric:** Illicit Class F1-Score on the Test Split (Time steps 35–49).
