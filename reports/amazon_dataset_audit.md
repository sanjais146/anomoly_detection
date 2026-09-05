# Amazon Dataset & Implementation Audit

## 1. Amazon Dataset Availability
- **Files Found:** data/amazon/raw/meta_Electronics.json.gz (186MB) and data/amazon/raw/reviews_Electronics.json.gz (1.77GB).
- **Entities Present:** Users (reviewerID), Products (asin).
- **Interactions:** Reviews containing rating (overall), helpfulness votes, and unixReviewTime.
- **Labels:** No explicit fraud labels exist in the raw data.

## 2. Graph Construction
- **Script:** pipeline/features/amazon_graph_builder.py
- **Nodes:** user and product.
- **Edges:** user -> reviews -> product.
- **Edge Attributes:** Timestamp, Rating, Helpful_ratio.
- **Temporal Constraint:** Handled gracefully via chronological sorting and train/val/test masking based on time percentiles (70/15/15).

## 3. TGAT Model Architecture
- **Script:** src/models/amazon_contrastive_tgat.py
- **Design:** CausalAmazonEncoder implementing causal temporal attention with learned decay 	au. HybridAmazonModel with Contrastive and Reconstruction loss.
- **Anomaly Detection:** Self-supervised link prediction. The anomaly score is derived from the negative contrastive score / reconstruction error.

## 4. Next Steps for Full Implementation
1. **Train Model & Save Checkpoint:** Create src/train_amazon_tgat.py to run the full training loop on a chronologically split subset of the Amazon graph, injecting synthetic anomalies for the test set evaluation, and saving models/amazon_tgat.pt.
2. **Inference Pipeline:** Create pp/amazon_predictor.py to expose the trained Amazon TGAT model for the FastAPI backend.
3. **API Update:** Update pp/main.py to route the primary e-commerce anomaly detection to the Amazon model, keeping E10 as a separate baseline.
4. **Dashboard:** Overhaul rontend/index.html to reflect the Amazon User/Product/Seller graph architecture and metrics.
5. **Colab & Readme:** Update Colab notebook to download the Amazon subset (if needed) and run the new endpoint. Rewrite the README.

**Verdict:** The Amazon dataset is genuinely present, and the graph construction + causal TGAT architecture is fundamentally sound. The anomaly definition is unsupervised/self-supervised due to the lack of ground-truth fraud labels, which is scientifically rigorous and honest.
