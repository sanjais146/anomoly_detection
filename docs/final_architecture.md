# Final System Architecture

## Overview
The final system is an **E-Commerce Anomaly Detection Pipeline** powered by a **Temporal Graph Attention Network (TGAT)**. It analyzes user-product interactions to detect structurally unexpected topological links in a bipartite e-commerce graph.

## Components

1. **Dataset Pipeline (Amazon Electronics)**
   - Extracts Reviewer IDs (Users), ASINs (Products), Ratings, and Unix Timestamps.
   - Represents the data as a bipartite interaction graph.

2. **Temporal Graph Attention Network (PyTorch)**
   - **Model:** `HybridAmazonModel`
   - **Mechanism:** Takes target User and Product nodes, pulls their historical temporal neighborhoods.
   - **Causal Filter:** Strict enforcement that `historical_timestamp < target_timestamp`.
   - **Decay:** Applies exponential decay $w(\Delta t) = \exp(-\tau \times \Delta t)$ using independently learned $\tau$ parameters for users and products.

3. **Anomaly Scorer**
   - Computes the dot product of the temporal embeddings.
   - Outputs a continuous likelihood score via a sigmoid activation.
   - Defines `Anomaly Score = 1 - Likelihood`.

4. **FastAPI Backend (`app/`)**
   - Serves the PyTorch model via REST endpoints (`/predict/amazon`, `/analytics/amazon/anomalies`).
   - Ensures deterministic execution for cold-start demo environments.

5. **Live Dashboard (`frontend/`)**
   - Built with vanilla JS, Chart.js, and Vis.js.
   - Dark-mode, Vercel-inspired analytics command center.
   - Connects directly to the FastAPI backend to visualize real inference results, timeline KPIs, and score distributions.

6. **Deployment (`colab/run_demo.ipynb`)**
   - A single-click Google Colab environment.
   - Handles dependency resolution, model loading, backend startup, and ngrok public tunneling for live demonstrations.
