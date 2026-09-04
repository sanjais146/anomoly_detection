# E10 Audit of E9 Implementation

## Objective
Verify the absolute causal integrity, isolation boundaries, and reproducibility of the E9 final optimization.

## Audit Findings
1. **Train/Validation/Test Boundaries:**
   - E9 explicitly dropped the Test set indices from memory immediately upon loading `IEEEFullFeaturesV2`.
   - `df_raw = df_raw.iloc[valid_idx]` guarantees that no Test data entered feature scaling, category encoding, or target statistics.

2. **Feature-Generation Ordering:**
   - Missingness features were derived exclusively from available columns at transaction time.
   - Temporal interactions (`c1`, `c1_addr`, `c1_dev`) correctly preserved original dataframe indices after generating chronological ego-graphs to avoid alignment corruption.

3. **7-Day Label Delay:**
   - Enforced rigorously in `_generate_e9_delayed` via `delta_seconds = 7 * 86400`.
   - The label of historical transaction $j$ is ONLY aggregated into the behavioral profile of target $i$ if $t_i - t_j \ge 604800$.

4. **Same-Timestamp & Target Exclusion:**
   - The backward lookup explicitly requires $t_{hist} < t_{target}$. Same-second transactions are excluded.

5. **Threshold & Ensemble Selection:**
   - Probabilities from `clf_base`, `clf_deep`, and `clf_weight` were averaged (`probs_ens = (probs_a + probs_b + probs_c) / 3.0`).
   - `best_threshold` strictly optimized over `y_va` and `probs_ens`. No test data leaked into probability calibration.

## Conclusion
The E9 implementation is mathematically honest, deployment-realistic, and causally valid. It strictly conforms to the IEEE-CIS chronological setup.
