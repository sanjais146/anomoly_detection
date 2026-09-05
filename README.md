# Amazon E-Commerce Anomaly Detection using TGAT

This repository contains the official implementation for the **Amazon E-Commerce Anomaly Detection** project, utilizing Temporal Graph Attention Networks (TGAT) to identify anomalous behaviors in e-commerce review graphs.

## 1. Primary Implementation: Amazon TGAT

The primary system models the **Amazon Electronics** dataset as a heterogeneous temporal graph (User → Reviews → Product). Because raw Amazon review datasets lack explicit ground-truth "fraud" labels, we evaluate anomalies via **self-supervised link prediction**.

### Architecture
- **Nodes:** Amazon Users (`reviewerID`) and Products (`asin`).
- **Edges:** Interaction events (Reviews) containing timestamps and ratings.
- **Model:** `HybridAmazonModel` with `CausalAmazonEncoder`.
- **Temporal Modeling:** Strictly causal (`t_hist < t_target`). Edge influence decays exponentially based on a learned parameter `τ`.
- **Anomaly Scoring:** Contrastive link reconstruction error. Highly improbable edges (low cosine similarity) are flagged as anomalies.

### Evaluation & Results (Amazon Dataset)
The dataset is split chronologically (70% Train, 15% Val, 15% Test) to prevent future data leakage. Synthetic unobserved edges are injected as negative "anomalous" samples for evaluation.

- **Test F1 Score:** 77.01%
- **Test AUROC:** 0.7997
- **Test Precision:** 71.83%
- **Test Recall:** 83.00%

*Note on Benchmarks:* The GNN-EADD paper reports ~73% on Amazon datasets using transductive protocols with external spam labels. Our 77.01% result is obtained under a strictly causal, inductive link-prediction protocol. While not directly comparable, it proves the efficacy of causal temporal attention on this dataset.

---

## 2. Historical Baseline: IEEE-CIS Transaction Fraud

To maintain a comprehensive record of our research, the previous **IEEE-CIS Transaction Fraud Detection** system is preserved as a separate baseline track.

- **Architecture:** E10 Static Causal CatBoost Ensemble (508-dimensional feature space).
- **Causal Protocol:** 7-day label availability boundary (simulating chargeback delay).
- **Test F1 Score:** 62.25% (Frozen and verified).

---

## 3. Project Structure

```
├── app/
│   ├── main.py                 # FastAPI application (Amazon & IEEE endpoints)
│   ├── amazon_predictor.py     # Inference adapter for Amazon TGAT
│   └── predictor.py            # Inference adapter for E10 baseline
├── src/
│   ├── train_amazon_tgat.py    # Training script for Amazon TGAT
│   └── models/
│       ├── amazon_contrastive_tgat.py
│       └── tgat_final.py
├── pipeline/
│   └── features/
│       └── amazon_graph_builder.py
├── frontend/
│   └── index.html              # E-Commerce Anomaly Detection Dashboard
├── models/
│   └── amazon_tgat.pt          # Frozen Amazon TGAT Checkpoint
└── reports/
    ├── amazon_tgat_analysis.md # Amazon performance report
    └── amazon_gnneadd_protocol.md # Base paper methodology comparison
```

## 4. Live Dashboard & API Deployment

The project features a full research-grade dashboard displaying the Amazon graph architecture, live TGAT inference, and comparative metrics.

### Running Locally
```bash
pip install -r requirements.txt
python app/main.py
```
Visit `http://127.0.0.1:8000` to view the Command Center dashboard.

### Google Colab Deployment
A completely self-contained deployment notebook is provided in `colab/run_demo.ipynb`. It will automatically:
1. Clone this repository.
2. Pull the frozen `.pt` and `.cbm` models via Git LFS.
3. Start the FastAPI backend.
4. Expose the dashboard globally using `ngrok`.
