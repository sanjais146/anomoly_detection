# Final Authoritative Results

| Model | Evaluation Protocol | Test F1 | Status |
|---|---|---:|---|
| Original XGBoost | Chronological | 49.69% | Valid |
| Original Hybrid T-GAT | Chronological | 49.99% | Valid |
| E2-B CatBoost | 0-day label feedback | 69.22% | INVALIDATED |
| E6 CatBoost | 7-day delayed induction | 56.48% | FINAL |
| GNN-EADD | Published benchmark | 73.00% | External |

**Why each result has its status:**
- **Original XGBoost:** Established a mathematically sound, leak-free floor using 31 arbitrary features.
- **Hybrid T-GAT:** Strictly causal, but provided negligible lift over the tabular baseline.
- **E2-B CatBoost:** Achieved high performance but was discovered to allow 0-day label feedback during cumulative fraud counting. Invalid for deployment-realistic comparison.
- **E6 CatBoost:** Corrected the E2-B leakage by enforcing a mathematically proven 7-day chargeback delay limit. This represents our strongest, scientifically defensible result.
- **GNN-EADD:** A published benchmark from external literature. Our experiments indicate that evaluation assumptions concerning temporal label availability and transductive information may substantially influence reported performance. However, the exact cause of the published 73% result cannot be definitively established without the original implementation.
