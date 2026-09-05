# E8 Error Analysis & Final Recommendation

## Results Summary
Best Temporal Graph Validation F1: 0.4615 (E8-A (No Decay))
Target Baseline (E7-D CatBoost): 0.6114

## Analysis

The end-to-end temporal graph model **failed to surpass** the heavily optimized causal CatBoost baseline.

### Why did it underperform?
1. **Neighborhood Sparsity/Noise**: The ego-graph captures raw transactions, but forces a deep neural network to learn aggregations (means, standard deviations, velocities) on the fly. CatBoost, utilizing mathematically pre-calculated aggregations (z-scores, velocity ratios in E7), had a direct structural advantage.
2. **Tabular Handling**: Deep learning models notoriously struggle against gradient boosted trees on mixed tabular data. The base features contained missing variables and numeric approximations of categoricals that CatBoost handles implicitly via oblivious trees, while PyTorch struggled with normalized numerical representations.
3. **Information Saturation**: The explicit temporal behavioral features added in E7 already extract the highest-value signal (concept drift, bursts).

### Stop Condition Met
Validation F1 did not exceed 61.14%. We correctly halted experimentation and did NOT touch the Test set.

## Final Recommendation
Do not replace CatBoost with T-GAT. The explicit, leak-free behavioral features combined with CatBoost provide superior, robust tabular performance. 
We stand by the final academic result established in E6/E7.
