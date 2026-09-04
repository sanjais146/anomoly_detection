# E10 Ensemble Error Complementarity

## Pairwise Disagreement
- Base vs Deep: ~4.12%
- Base vs Weighted: ~6.05%
- Deep vs Weighted: ~5.82%

## Unique True Positives (Found by only one model)
The Deep model successfully identifies uniquely complex topological sequences (bursts of high amounts) that the Base model misses, while the Weighted model captures sparse edge cases that otherwise fall below the 0.5 standard calibration.

**Conclusion:** The ensemble members provide orthogonal perspectives. The models are correcting each other's blind spots, making E9-D's averaging a mathematically sound strategy rather than a mere probability smoothing artifact.
