# Viva Preparation & Explanations

## 1. What is an e-commerce anomaly?
In the context of our Amazon implementation, an anomaly is a highly improbable interaction (a review) between a User and a Product, given their historical temporal graph behavior. It captures potential spam, review bombing, or hijacked accounts where the topological context contradicts the interaction.

## 2. Why Amazon?
The Amazon Electronics review graph is a widely recognized standard in academic literature (e.g., GNN-EADD). It provides dense, chronological interactions (Users reviewing Products) critical for demonstrating the capabilities of Temporal Graph Attention Networks in a real-world e-commerce setting.

## 3. Why TGAT?
Temporal Graph Attention Networks (TGAT) naturally model the continuous-time dynamic nature of e-commerce interactions. Unlike static graphs or tabular models, TGAT computes embeddings that dynamically evolve, aggregating neighborhood features while applying causal temporal decay to prioritize recent activity.

## 4. What are the graph nodes and edges?
- **Nodes**: Users (`reviewerID`) and Products (`asin`).
- **Edges**: Interactions (Reviews). Features include the timestamp, the rating (`overall`), and the helpfulness ratio.

## 5. What is temporal attention?
Temporal attention is a mechanism where the model learns to weight historical neighbors based not just on topological similarity, but on time. We implement an exponential decay function `w = exp(-τ * Δt)`, where `τ` is learned via backpropagation, to gracefully forget older interactions.

## 6. What is the anomaly score?
The anomaly score is the **contrastive link reconstruction error**. The model predicts the probability of an interaction occurring. High probability (high similarity between temporal embeddings) = genuine interaction. Low probability = anomalous interaction. We score it as `1.0 - probability`.

## 7. How are anomalies labeled?
The raw Amazon dataset lacks explicit "fraud" labels. Therefore, we utilize **self-supervised link prediction**. Genuine interactions act as positives, and synthetically generated counterfactual edges (interactions that never occurred) are injected as anomalies during evaluation.

## 8. How do you prevent leakage?
We enforce a strict **causal protocol**.
1. **Chronological Splitting:** The data is sorted by timestamp before splitting into Train (70%), Val (15%), and Test (15%).
2. **Causal Masking:** When evaluating an interaction at time `t`, the TGAT aggregator strictly masks out any neighbor interactions where `t_hist >= t`.

## 9. How was the threshold selected?
The threshold (0.52) was selected dynamically on the validation set by maximizing the F1 score across a linear sweep of possible threshold values.

## 10. What is the 77.01% result?
It is the Test F1 score achieved by the TGAT model on the Amazon Electronics graph distinguishing genuine, chronologically valid interactions from synthetically injected anomalous interactions, strictly without seeing future edges.

## 11. Is it comparable to GNN-EADD? Why not claim that you beat the paper?
No, it is not directly comparable. GNN-EADD utilizes a transductive protocol (the graph structure is fully known, labels are masked) often combined with external heuristic spam labels. Our implementation uses a strict causal, inductive protocol (no future edges visible) evaluating link reconstruction error. Claiming we "beat" it would be scientifically dishonest; we simply demonstrate high efficacy (77.01%) under our more stringent temporal constraints.

## 12. Why does IEEE-CIS still exist? What is E10?
The IEEE-CIS Transaction Fraud project was our original parallel baseline. **E10** is our static causal CatBoost ensemble trained on 508 tabular features, which achieved 62.25% Test F1. It is preserved because it represents a complete, rigorously evaluated supervised fraud detection baseline utilizing a simulated 7-day chargeback delay protocol.

## 13. How does the live demo work?
The FastAPI backend receives an interaction request (`reviewerID`, `asin`, timestamp). The `amazon_predictor.py` proxies the node embeddings through the frozen TGAT checkpoint (`amazon_tgat.pt`), calculates the temporal decay and contrastive similarity, and returns the anomaly probability to the frontend dashboard.

## 14. Why Google Colab and ngrok?
Google Colab provides a reproducible, cloud-based, GPU-capable Linux environment, bypassing local dependency hell. `ngrok` bridges the Colab runtime to a public URL, allowing the dashboard to be presented seamlessly to external evaluators or users without complicated network routing.

## 15. What are the limitations?
1. The anomaly definition relies on link reconstruction rather than ground-truth financial fraud labels (due to dataset limitations).
2. For latency reasons, the live demo approximates node embeddings rather than querying a live graph database (like Neo4j) to dynamically retrieve large ego-networks in real-time.
