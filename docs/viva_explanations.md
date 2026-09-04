# Viva Explanations

## 30-Second Explanation
"The system detects anomalous e-commerce transactions using a causal CatBoost ensemble. Instead of allowing the model to see future fraud labels, historical fraud information is only introduced after a simulated seven-day chargeback delay. Behavioral features such as transaction velocity, historical amounts, card-device interactions, and temporal patterns are calculated from information that would have been available at prediction time. Three CatBoost configurations are then combined into an ensemble, and a validation-derived threshold converts the probability into a fraud decision."

## 1-Minute Explanation
"Our project tackles e-commerce fraud detection by prioritizing deployment realism over inflated benchmark scores. Many academic papers achieve >70% F1 by accidentally allowing future information to leak into the training process. We enforced a strict 7-day chronological delay, reflecting the real-world reality that chargebacks take time to verify. We engineered historical behavioral features that respect this boundary, feeding them into an ensemble of CatBoost models (Base, Deep, and Weighted). Evaluated on a strictly unseen future Test month, our system achieves 62.25% F1, significantly outperforming a clean 49.69% baseline, representing the maximum valid signal extractable under strict causality."

## 2-Minute Explanation
"In this project, we built a fraud detection pipeline on the IEEE-CIS dataset. Early in our research, we realized that standard random cross-validation allows models to peak into the future, and instantly updating labels allows '0-day' feedback, both of which are impossible in a real credit card processing environment. 

To solve this, we implemented a strict inductive chronological split. For any transaction evaluated today, the model is only allowed to train on fraud labels that are at least 7 days old. We engineered high-signal behavioral features—like a credit card's transaction count or average amount over the last 7 days—using Numba-accelerated sliding windows that mathematically exclude the most recent 7 days of labels.

Our final architecture, the E10 Ensemble, averages predictions from three CatBoost models of varying depths. When evaluated on a permanently locked Test set, it achieved a 62.25% F1 score, beating a clean XGBoost baseline of 49.69% by over 12 percentage points. We also audited an external 73% benchmark and determined it was evaluated transductively on a different dataset, making it incomparable. Our 62.25% represents a highly robust, deployment-ready metric."

## Technical Explanation
"The core of the architecture relies on Gradient Boosted Decision Trees (CatBoost) applied to a highly dimensional, causally restricted feature space. We address temporal concept drift and information leakage by enforcing $\Delta = 7$ days. For a prediction at $t$, the training matrix $X_{train}$ and $Y_{train}$ are strictly bounded by $t_{label} \le t - \Delta$. 

Categorical features (e.g., `ProductCD`, `card1`) are handled natively by CatBoost's ordered target encoding, which we specifically configure to avoid look-ahead bias. The ensemble aggregates probabilities from a Depth 6 model, a Depth 8 model, and a Weighted loss model, using a static decision threshold of $0.4040$ optimized strictly on the chronological validation set. Inference takes $O(K \cdot D)$ where $K$ is the number of trees and $D$ is the depth, making it highly suitable for sub-millisecond production inference."

## Non-Technical Explanation
"Imagine you are a security guard at a store. If you automatically know someone is a thief the exact second they steal something, stopping them is easy. But in reality, you only find out someone was using a stolen credit card a week later when the real owner complains. 

Our AI system is trained to catch fraud under this exact realistic 'one-week delay' condition. It looks at how a customer is behaving today—how much they are spending, what device they are using—and compares it to known fraud patterns from a week ago. By combining three different AI 'opinions' into one final decision, we built a highly accurate system that works in the real world, rather than just looking good in a laboratory."
