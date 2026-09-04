import pytest
import numpy as np
import pandas as pd
from pipeline.features.multi_entity_causal import generate_causal_history

def test_strict_inductive_causality():
    # Mock data
    # Train End Time: 100
    # Entity E1
    # t=10: amt=100, fraud=1 (Train)
    # t=20: amt=200, fraud=0 (Train)
    # t=20: amt=300, fraud=0 (Train, SAME TS)
    # t=110: amt=400, fraud=1 (Val - Label should be HIDDEN)
    # t=120: amt=500, fraud=0 (Val)
    
    df = pd.DataFrame({
        'TransactionID': [1, 2, 3, 4, 5],
        'card1': ['E1', 'E1', 'E1', 'E1', 'E1'],
        'TransactionDT': [10, 20, 20, 110, 120],
        'TransactionAmt': [100.0, 200.0, 300.0, 400.0, 500.0],
        'isFraud': [1, 0, 0, 1, 0]
    })
    
    train_end_time = 100
    res = generate_causal_history(df, 'card1', train_end_time)
    
    # 1. No target self inclusion & No same-timestamp leakage
    # At t=20 (row 1 and 2), they should only see t=10
    assert res.loc[1, 'card1_all_tx_count'] == 1
    assert res.loc[1, 'card1_all_fraud_rate'] == 1.0
    assert res.loc[2, 'card1_all_tx_count'] == 1
    
    # 2. At t=110 (row 3, Val)
    # Sees t=10, 20, 20. Total past_cnt = 3. Past fraud = 1.
    assert res.loc[3, 'card1_all_tx_count'] == 3
    assert res.loc[3, 'card1_all_fraud_rate'] == 1.0 / 3.0
    
    # 3. Strict Inductive Check: At t=120 (row 4, Val)
    # Sees t=10, 20, 20, 110. Total past_cnt = 4. 
    # But t=110 label is HIDDEN because 110 > 100.
    # So past_fraud_cnt = 1 (from t=10), past_label_cnt = 3 (from t=10, 20, 20).
    assert res.loc[4, 'card1_all_tx_count'] == 4
    # Fraud rate = 1 / 3
    assert res.loc[4, 'card1_all_fraud_rate'] == 1.0 / 3.0
    
    # 4. Time since last fraud for t=120
    # Should point to t=10, NOT t=110 because t=110 is hidden.
    assert res.loc[4, 'card1_t_since_last_fraud'] == 120 - 10
    
    # Time since last tx for t=120
    # Should point to t=110, because behavior is known instantly!
    assert res.loc[4, 'card1_t_since_last_tx'] == 120 - 110
    
    print("All E2-C integrity tests PASSED!")

if __name__ == '__main__':
    test_strict_inductive_causality()
