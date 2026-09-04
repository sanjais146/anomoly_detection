# Final Academic Summary

### Research Problem
E-commerce fraud detection is characterized by extreme class imbalance and adversarial concept drift. Real-world systems must wait weeks for chargebacks to confirm fraud. Traditional transductive evaluation models ignore this delay, inadvertently leaking future knowledge backwards and inflating performance.

### Dataset
IEEE-CIS Fraud Detection (590,540 transactions, 6 months).

### Baseline
A clean, 31-feature XGBoost model evaluated chronologically (49.69% Test F1).

### Proposed Approaches
We investigated rich causal behavioral aggregations using CatBoost, and Temporal Graph Attention Networks (T-GAT) to model entity interactions. 

### T-GAT Investigation
T-GAT was hypothesized to capture temporal behavioral patterns. However, dense graph embeddings failed to provide complementary orthogonal signal when the tabular models were already equipped with explicitly engineered causal interaction velocities.

### Leakage Discovery
An intermediate model (E2-B) achieved 69.22% Test F1 but was invalidated after discovering 0-day label feedback. Continuous fraud counters allowed validation transactions to instantly access labels of transactions occurring moments prior, violating real-world chargeback constraints.

### Corrected Protocol
We introduced 7-Day Delayed-Label Induction ($\Delta$-Induction), mathematically restricting historical fraud labels to those occurring at least 7 days before the target transaction.

### Final Model
E6 7-Day Delayed-Induction CatBoost.

### Final Results
**56.48% Test F1** (+6.79 percentage points over the baseline).

### Main Contribution
Establishing a mathematically rigorous, deployment-realistic evaluation protocol for IEEE-CIS, and demonstrating that robust causal tabular feature engineering provides superior, physically realizable predictive power compared to transductive dense graph networks.

### Limitations
Blinded by the 7-day label delay, the model suffers from natural concept drift and struggles to instantly identify entirely novel adversarial fraud strategies emerging in the test window.

### Future Work
Potential future directions (not implemented in this repository) include:
- Adaptive delayed-label learning
- Concept-drift detection
- Online/continual learning
- Richer temporal graph representations
- Better calibration
- Cost-sensitive learning
- Specialized fraud-ring detection
- Alternative temporal GNN architectures
