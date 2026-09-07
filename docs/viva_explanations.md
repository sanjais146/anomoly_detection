# Viva Explanations & Defenses

This document provides structured answers for the final project viva.

## Elevator Pitches

### 30 Seconds: What is the project?
"This project is an E-Commerce Anomaly Detection system. Instead of relying on static rules, it uses a Temporal Graph Attention Network (TGAT) to analyze the evolving bipartite graph of user-product interactions—specifically Amazon reviews. The system mathematically flags interactions that are topologically unexpected given the causal history of the entities involved."

### 1 Minute: How does TGAT detect anomalies?
"TGAT treats the e-commerce platform as a massive network. When a new interaction occurs between a user and a product, TGAT looks at their strictly prior historical neighborhoods. It applies a learned continuous-time exponential decay so recent events matter more than old ones. It creates temporal embeddings for both the user and product, and takes their dot product. If the dot product is high, the link was expected. If it's extremely low, the link contradicts the historical graph structure, and we flag it as an anomaly—such as a compromised account or review bombing."

### 2 Minutes: Explain the graph and temporal architecture.
"The core architecture is a `HybridAmazonModel` in PyTorch. The graph is bipartite: Users and Products. The edges are time-stamped interactions. 
To process this, our `CausalAmazonEncoder` enforces strict causal masking—meaning it absolutely prevents the model from looking at future or simultaneous edges to avoid data leakage. 
For the edges it is allowed to see, it applies a decay factor $\tau$, which the model learned during training. For Amazon Electronics, the network learned a half-life of about 1.6 days, meaning e-commerce structures change rapidly. Finally, the network aggregates these temporal weights to form a context vector, which is used to score the probability of the new interaction."

### 5 Minutes: System Overview
*(Combine the 2-minute pitch with the following)*
"The complete system goes from raw data to live dashboard. We trained the TGAT on a 20,000-interaction Amazon Electronics graph using a contrastive link prediction objective. 
In deployment, we serve the frozen PyTorch checkpoint via a FastAPI backend. For the demonstration, we use a cold-start batch mode on a representative 100-record sample to demonstrate the inference pipeline live. 
The dashboard itself is a modern, Vercel-inspired command center built with Chart.js and Vis.js, connecting directly to the API to visualize the score distributions, temporal decay curves, and the anomalous interactions. We also retain our earlier static CatBoost experiments on the IEEE-CIS dataset as a historical supervised baseline, demonstrating the evolution of our research from static classification to temporal graph anomaly detection."

---

## Hostile Questions & Defenses

**Q: Why TGAT? Why not just use a simpler model like Random Forest or CatBoost?**
A: "We actually *did* use CatBoost (our E10 benchmark) for supervised fraud classification on the IEEE-CIS dataset, achieving 62.25% F1. However, static models ignore the rich, evolving structural topology of e-commerce. Fraud and anomalies in e-commerce—like review rings or account takeovers—are inherently relational and temporal. TGAT naturally models these structural dynamics via temporal neighborhoods, which tabular models cannot do natively."

**Q: What exactly is an anomaly in your system?**
A: "An anomaly is mathematically defined as a topologically unexpected user-product link. It is scored as $1 - \sigma(E_u \cdot E_p)$. If the model's understanding of the user's history and the product's history strongly suggests they shouldn't interact, but they do, the score approaches 1.0."

**Q: Why use Amazon Electronics? Why not IEEE-CIS for everything?**
A: "IEEE-CIS is excellent for supervised classification because it has explicit `isFraud` labels, but its graph structure must be synthetically inferred (e.g., matching Card IDs) and its timestamps are masked `TimeDeltas`. Amazon Electronics provides a genuine bipartite graph (User -> Product) with real Unix timestamps, making it scientifically far superior for training and evaluating a Temporal Graph Attention Network. Since Amazon lacks explicit fraud labels, we properly framed the final TGAT pipeline as unsupervised anomaly detection."

**Q: How is the threshold chosen? Are you just hacking the numbers to look good?**
A: "No. The threshold is strictly documented. In full training graph evaluation, the threshold was calibrated to 0.52 to separate positive links from corrupted negative links. In our live dashboard, which operates in a 'cold-start' demo mode without caching the entire historical graph in memory, the base score distribution shifts upward. To maintain a scientifically honest demonstration, we empirically calibrated the demo threshold to 0.6668 (the 75th percentile of the demo distribution) and explicitly labeled it as such in the UI. We do not conflate the two."

**Q: How do you prevent temporal leakage?**
A: "We enforce strict causal masking in the `CausalAmazonEncoder`. The condition `edge_times < target_time` is mathematically enforced during tensor operations. Simultaneous or future interactions are multiplied by zero before aggregation. The UI proudly displays '0.00% Future Leakage' because it is enforced at the PyTorch level."

**Q: Why is the E10 62.25% result not called a TGAT score?**
A: "Because that would be scientifically dishonest. The 62.25% F1 belongs to our historical static CatBoost ensemble applied to the IEEE-CIS dataset. TGAT operates on Amazon Electronics as an unsupervised anomaly detector. The final system explicitly separates the Primary Final System (Amazon TGAT) from Historical Supporting Research (IEEE-CIS E10)."

**Q: Where are the limitations of your system?**
A: "The primary limitation is temporal resolution. Amazon datasets generally provide timestamps at a daily resolution (00:00:00). We cannot infer sub-day causality. Secondly, our live demo currently operates in cold-start mode, using deterministic node features rather than querying a live massive graph database, which would be required for a full production deployment."
