# Final IEEE-CIS Research Plan
**Objective:** Develop a causally constrained temporal graph architecture for e-commerce fraud detection that integrates transaction-level tabular information with strictly historical entity interactions and learnable temporal decay.

## 1. Baselines
* XGBoost (Tabular upper bound)
* Simple MLP

## 2. Temporal Graph Architecture (T-GAT Final)
* Nodes: Transactions
* Entities: Cards, Devices
* Temporal Decay: exp(-tau * delta_t)
* Causal Mask: 	_hist < t_target

## 3. Regularization Strategy (Overfitting Prevention)
* High Dropout (0.5)
* Weight Decay (1e-4)
* Reduced Hidden Dimension (e.g., 32 or 16)
* Attention-only (removing GRU which previously overfit)
* Early Stopping on Validation F1
