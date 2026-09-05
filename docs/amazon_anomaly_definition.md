# E-Commerce Anomaly Definition (Amazon Graph)

## 1. What is an anomaly?
In the context of the Amazon Electronics review dataset, an anomaly is a **highly improbable interaction** between a User and a Product, given their historical behaviors. Examples include:
- Review bombing (sudden influx of reviews on a product from disconnected users).
- Account hijacking (a user suddenly interacting with products completely outside their historical distribution).
- Spam/Bot activity (high-velocity, low-context interactions).

## 2. Which entity is classified?
The classification occurs at the **Edge (Interaction) level**. We score the `User → Product` review event.

## 3. What is the ground truth?
The raw Amazon Electronics dataset **does not** contain explicit "fraud" labels. 
To preserve scientific honesty and avoid fabricating data, we treat this as an **unsupervised/self-supervised anomaly detection** task.
- **Normal interactions (Positives):** The actual observed reviews in the dataset.
- **Anomalous interactions (Negatives):** Synthetically generated counterfactual edges (users interacting with products they have no genuine affinity for), representing injected anomalies for evaluation.
- The model must assign high probabilities to genuine links and low probabilities to anomalous links.

## 4. What features make an entity anomalous?
An interaction is flagged as anomalous if the temporal graph context of the user and the product do not align.
- **User History:** Past products reviewed, ratings, and timestamps.
- **Product History:** Past users who reviewed it, their ratings, and timestamps.
- **Feature Space:** Ratings, helpfulness votes, and temporal decay (`exp(-τ * Δt)`).

## 5. How does TGAT detect/represent abnormal temporal behavior?
TGAT builds temporal embeddings for the User and Product by aggregating their respective historical interactions. 
- **Causal Masking:** Only interactions where `t_hist < t_target` are used.
- **Temporal Decay:** Recent interactions are weighted higher via a learned parameter `τ`.
- **Contrastive Anomaly Score:** The model computes the dot product of the temporal User embedding and Product embedding. A low score (high reconstruction error) indicates that the interaction is anomalous compared to the learned temporal graph distribution.
