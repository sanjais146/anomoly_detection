# Final Anomaly Definition

**Date:** 2026-09-07
**Project:** E-Commerce Anomaly Detection

## Concept
In this system, an anomaly is defined mathematically as an **unexpected topological link**. 

Given an e-commerce graph where users interact with products over time, an anomaly occurs when a user interacts with a product that the model strongly predicts they should *not* interact with, given the historical behavior (neighborhood) of both the user and the product.

## Mathematical Derivation
The TGAT model is trained using a contrastive link-prediction objective. 
1. The `CausalAmazonEncoder` processes the temporal neighborhood of User $u$ and Product $p$ strictly before time $t$, producing embeddings $E_u$ and $E_p$.
2. The dot product $S = E_u \cdot E_p$ represents the structural similarity or expected likelihood of the interaction.
3. During training, genuine observed links (positives) are pushed to have high similarity, while random corruptions (negatives) are pushed to have low similarity.

**Anomaly Probability:**
We define the anomaly probability as the complement of the expected likelihood:
$$ P(Anomaly) = 1 - \sigma(E_u \cdot E_p) $$
Where $\sigma$ is the sigmoid function. 

- A **high dot product** (high similarity) means the link is expected based on historical graph patterns. The anomaly score approaches **0**.
- A **low dot product** means the link contradicts historical patterns. The anomaly score approaches **1**.

## Decision Rule
The continuous anomaly probability is converted into a binary decision using a threshold:

$$ \text{Decision} = \begin{cases} \text{ANOMALY} & \text{if } P(Anomaly) \geq \tau \\ \text{NORMAL} & \text{if } P(Anomaly) < \tau \end{cases} $$

### Threshold Calibration ($\tau$)
- **Training Mode ($\tau = 0.52$):** Calibrated on the full training graph with rich historical edge context.
- **Demo Mode ($\tau = 0.6668$):** Calibrated at the 75th percentile of the cold-start distribution (where historical edges are empty). Used exclusively for the live demonstration dashboard to flag the most topologically unusual ~25% of isolated interactions.

## What This Means Practically
If a user account is compromised and suddenly reviews 50 unrelated electronics on the same day, the lack of historical neighborhood overlap between the user's past behavior and the targeted products will result in a low dot product, yielding a high anomaly score. This identifies potential review bombing, account takeovers, or coordinated inauthentic behavior without needing explicit fraud labels.
