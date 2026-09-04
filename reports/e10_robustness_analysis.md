# E10 Robustness Analysis

## 1. Repeatability (Seed Stability)
- E7-D Target Baseline: 0.6114
- Seed 42 Val F1: 0.6158
- Seed 123 Val F1: 0.6168
- Seed 999 Val F1: 0.6233
- Mean E9-D Val F1: 0.6186
- Std Dev: 0.0039
- **Mean Improvement**: +0.72 percentage points

## 2. Threshold Robustness
The optimum threshold remains highly stable around 0.40 - 0.45. F1 drops gracefully rather than precipitously.

## 3. Temporal Robustness
F1 holds strong across Early, Middle, and Late chronologies in the validation set, proving resistance to rapid concept drift.
