# PROJECT AUDIT REPORT
**IEEE-CIS E-Commerce Anomaly Detection — Pre-Implementation Audit**

## 1. ACTUAL PROJECT OBJECTIVE
The project is supervised binary anomaly/fraud classification on the IEEE-CIS e-commerce transaction dataset. The scientific contribution is a rigorous 7-day causal evaluation protocol, not a new model architecture.

**Defensible framing:** SUPERVISED ANOMALY DETECTION — detects transactions whose behavior is anomalous relative to historical entity baselines, with fraud labels as ground-truth supervision.

## 2. DATASET ENTITIES (Code-Verified)
- Transactions: TransactionID, TransactionDT, TransactionAmt, isFraud
- Card Identity: card1+card2 composite (card_id), card3-card6
- Address: addr1, addr2
- Email: P_emaildomain, R_emaildomain
- Device: DeviceInfo+id_31 composite (device_id), DeviceType
- Product: ProductCD
- Identity signals: id_01–id_38 (anonymized)

## 3. TGAT STATUS — DEFINITIVE
EXISTS in src/ (tgat.py, tgat_final.py, best_tgat_final.pt, graph_builder.py)
NOT IN FINAL DEPLOYMENT PIPELINE
Reason: E3/E8 experiments showed TGAT degraded vs tabular CatBoost with equivalent features.
Correct statement: TGAT was researched and implemented; final E10 uses CatBoost with graph-equivalent behavioral features.

## 4. E10 ARCHITECTURE (Exact)
- Base: CatBoost depth=6, lr=0.10, 600 iterations
- Deep: CatBoost depth=8, lr=0.08, l2_leaf_reg=5
- Weighted: CatBoost depth=6, scale_pos_weight
- Ensemble: average of 3 probabilities
- Threshold: 0.4040 (validation-frozen)
- Features: 508 dimensions (V1-339, C1-14, D1-15, categoricals, delayed behavioral)

## 5. WHAT CAN CHANGE
- frontend/ — full UI rewrite
- README.md — rewrite
- docs/ — update
- app/main.py — add endpoints carefully

## 6. WHAT IS FROZEN
- models/e10_*.cbm — never touch
- E10 Test F1 = 62.25%
- 7-day causal protocol
- threshold = 0.4040
- all reports/
- all experiments/
