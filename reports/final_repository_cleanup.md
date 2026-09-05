# Final Repository Cleanup Report

## 1. File Changes
- **TOTAL FILES BEFORE**: 147 tracked + 87 untracked = 234 files
- **FILES DELETED**: 87 (All untracked temp scripts, caches, duplicate models, legacy ZIPs)
- **FILES ARCHIVED**: 45 (Legacy tests for E2-E14, ablation JSONs, legacy PNG matrices moved to elease/historical/)
- **FILES RETAINED**: 102 tracked core files

## 2. Amazon TGAT Verification
- **AMAZON DATASET USED**: data/amazon/raw/reviews_Electronics.json.gz
- **REAL AMAZON TRAINING**: YES
- **POSITIVE DATA**: Real Amazon interactions from the dataset parsing function
- **NEGATIVE SAMPLING**: Synthetically generated unobserved edges between real users and products
- **SYNTHETIC DATASET**: NO (Real data used for entities and positive topology)
- **TGAT CHECKPOINT**: models/amazon_tgat.pt (timestamp verified matching the training run)
- **77.01% RESULT**: YES (Produced natively by src/train_amazon_tgat.py)

## 3. Deployment Components
- **TESTS**: PASS (	est_amazon_system.py tests causal masking, model inference, and FastAPI)
- **COLAB**: PASS (colab/run_demo.ipynb is standalone and ngrok works)
- **DASHBOARD**: PASS (rontend/index.html runs entirely standalone, style.css/script.js removed)

## 4. Notes
- The IEEE-CIS baseline (E10) remains preserved in models/ and eports/ as requested.
- Causal constraints (	_hist < t_target) verified intact.
