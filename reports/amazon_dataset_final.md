# Amazon Dataset Documentation

## 1. Dataset Provenance
- **Dataset Name**: Amazon Product Data (Electronics Category).
- **Source**: Julian McAuley, UCSD (Standard Amazon review datasets).
- **Version**: Pre-2018 JSON format.

## 2. Files Utilized
- data/amazon/raw/reviews_Electronics.json.gz (1.77 GB)
- data/amazon/raw/meta_Electronics.json.gz (186 MB)

## 3. Supported Entities & Metadata
The dataset natively supports the following entities used in our graph:
- **Users**: Identified by eviewerID.
- **Products**: Identified by sin.
- **Reviews (Interactions)**: Edges between Users and Products.
  - overall (Rating: 1.0 to 5.0)
  - unixReviewTime (Timestamp in seconds)
  - helpful (Votes array [helpful, total])

**Note on Sellers**: The raw meta_Electronics file contains some brand/seller information, but to maintain strict graph density and reliability in the primary evaluation, the TGAT model operates strictly on the bipartite User ↔ Product review graph.

## 4. Anomaly Context
- Explicit "fraud" or "anomaly" ground-truth labels do not exist in this dataset.
- Anomalies are derived via self-supervised link prediction (reconstruction error). Unobserved interactions (synthetic edges) are treated as anomalies to evaluate the model's ability to discern genuine topological interactions from aberrant ones.
