# E12 Continual Test Integrity Audit

The Test evaluation rigorously simulated online learning. Every prediction checkpoint completely excluded any information from $t > t_{prediction} - 7$ days.

- **Chunk 1:** Pred [13132810.0, 13737610.0) (22018 rows). Train Max Time: 12527981.0 (482179 rows). Gap: 7.00 days.
- **Chunk 2:** Pred [13737610.0, 14342410.0) (20718 rows). Train Max Time: 13132810.0 (501215 rows). Gap: 7.00 days.
- **Chunk 3:** Pred [14342410.0, 14947210.0) (20452 rows). Train Max Time: 13737594.0 (523232 rows). Gap: 7.00 days.
- **Chunk 4:** Pred [14947210.0, 15552010.0) (18028 rows). Train Max Time: 14342407.0 (543950 rows). Gap: 7.00 days.
- **Chunk 5:** Pred [15552010.0, 15811132.0) (8110 rows). Train Max Time: 14947189.0 (564402 rows). Gap: 7.00 days.

- **Threshold Integrity:** Frozen precisely at the `0.2316` optimum derived from E11-B Validation. Test data was never used to re-tune.