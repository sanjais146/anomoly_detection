# E14 Final Recommendation

1. **Did adaptive continual learning improve performance?** Yes, slightly.
2. **By how many F1 points?** The mean Validation F1 improved by **+0.92 percentage points** (from 60.32% to 61.24%). 
3. **Is the gain robust?** Moderately. Across seeds 42, 123, and 999, the improvement varied (+1.56pp, +0.26pp, +0.94pp). It improved upon the E11-B baseline in all seeds, but with high variance.
4. **Which component contributed most?** Exponential recency weighting ($\lambda=0.03$, ~23-day half-life). 
5. **Does concept drift remain the dominant limitation?** Yes, but addressing it aggressively induces variance that counteracts the gains.
6. **Is another Test evaluation justified?** **NO.** The formal decision gate requires a strictly robust $\ge +1.5$ pp improvement to authorize a Test evaluation. At +0.92 pp mean improvement, we fall short of the statistical significance required to unfreeze the Test set.
7. **What is the strongest final candidate?** The **E10 Static CatBoost Ensemble** (62.25% Test F1) remains the mathematically undisputed, authoritative causal model for this project.

**Project Status: FROZEN.**
