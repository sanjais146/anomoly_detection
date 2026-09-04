import numpy as np
import pandas as pd
import pytest

from pipeline.features.e6_delayed_causal import generate_e6_features

def test_delayed_induction():
    sec_per_day = 86400
    
    # Create mock dataset
    # Entity 1 makes transactions on days 10, 20, 30, 40
    # True Fraud Labels:
    # Day 10: Fraud
    # Day 20: Legit
    # Day 30: Fraud
    # Day 40: Target transaction we want to predict
    
    df = pd.DataFrame({
        'TransactionID': [1, 2, 3, 4],
        'TransactionDT': [10*sec_per_day, 20*sec_per_day, 30*sec_per_day, 40*sec_per_day],
        'TransactionAmt': [100.0, 100.0, 100.0, 100.0],
        'isFraud': [1, 0, 1, 0],
        'card1': [101, 101, 101, 101]
    })
    
    # 1. Delta = 14 days
    # At Day 40, history = Day 10, 20, 30.
    # Label cutoff = Day 40 - 14 = Day 26.
    # Therefore, Day 10 (Fraud) and Day 20 (Legit) are visible labels.
    # Day 30 (Fraud) is NOT visible yet (40 - 30 = 10 < 14).
    # Expected fraud_count_all at Day 40 = 1 (from Day 10).
    # Expected tx_count_all at Day 40 = 3 (Day 10, 20, 30 are all < 40).
    
    feat_14d = generate_e6_features(df, ['card1'], 'c1', delta_days=14)
    
    row_day40_14 = feat_14d.iloc[3]
    assert row_day40_14['c1_tx_count_all'] == 3, "Behavioral count should see all past"
    assert row_day40_14['c1_fraud_count_all'] == 1, "Label count should only see >14 days past"
    
    # 2. Delta = 7 days
    # At Day 40, label cutoff = Day 33.
    # Day 30 (Fraud) IS visible (40 - 30 = 10 >= 7).
    # Expected fraud_count_all at Day 40 = 2 (Day 10, Day 30).
    
    feat_7d = generate_e6_features(df, ['card1'], 'c1', delta_days=7)
    
    row_day40_7 = feat_7d.iloc[3]
    assert row_day40_7['c1_fraud_count_all'] == 2, "Label count should see >7 days past"
    
    # 3. Delta = 0 days (instantaneous labels, but still strictly past)
    # Expected fraud_count_all = 2.
    feat_0d = generate_e6_features(df, ['card1'], 'c1', delta_days=0)
    row_day40_0 = feat_0d.iloc[3]
    assert row_day40_0['c1_fraud_count_all'] == 2
    
    print("All E6 delayed-label induction tests passed.")

if __name__ == "__main__":
    test_delayed_induction()
