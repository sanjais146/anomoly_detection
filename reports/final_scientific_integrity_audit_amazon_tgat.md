# Final Scientific Integrity Audit: Amazon TGAT

## 1. Model & Training Provenance
- **Exact dataset used for TGAT training**: Amazon Electronics (eviews_Electronics.json.gz from Snap/Julian McAuley)
- **Exact number of training interactions**: 20,000 interactions chronologically sampled from the raw 1.77 GB file.
- **Entities**: 19,236 Users and 2,012 Products. 
- **Methodology**: Contrastive Link Prediction utilizing 	_hist < t_target strict temporal causal masking. No synthetic user/product entities were generated; anomalous edges were sampled strictly as unobserved historical connections.

## 2. Dashboard Analytics & Reporting
- **Source of Dashboard Sample**: The 100-record sample serving the dashboard charts (data/amazon/raw/sample_reviews.json) is a securely downloaded contiguous byte-range extraction from the official public Snap dataset. It is NOT a synthetic mock.
- **Charts using the 100-record sample**:
  1. Temporal Interaction Timeline
  2. Amazon Review Rating Distribution
  3. Bipartite Amazon Review Graph (Initial render)
  *All three charts are explicitly labeled in the UI as: "Representative sample of 100 real Amazon Electronics reviews".*
- **Charts using the live TGAT Model**:
  1. The "Analyze Amazon Interaction" live inference panel.
  2. The appended User-Product links added to the Graph Analytics Vis.js network after a user runs a prediction.
- **Metrics from Verified Reports**:
  - F1 = 77.01%, Recall = 83.00%, AUROC = 0.7997 (Sourced directly from eports/amazon_tgat_results.json)
  - τ = 0.431 (Sourced from the trained causal checkpoint).
  - E10 Baseline F1 = 62.25% (Sourced from legacy IEEE-CIS test reports).

## 3. Synthetic Data Elimination
- A comprehensive Select-String audit was performed on the frontend and backend codebase.
- The Math.random() invocations previously used for formatting placeholder UI embedding arrays were entirely removed.
- **Result**: Zero synthetic visualization data exists in the application. Every plotted coordinate, network edge, and classification derives from genuine datasets or the authentic inference engine.

## 4. Verification Tests
- All endpoints (GET /health, GET /analytics/amazon, POST /predict/amazon) were verified.
- The test suite (pytest tests/ -v) passed with **7/7** passing tests validating the TGAT checkpoint, causal logic, and routing.
