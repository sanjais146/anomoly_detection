import pytest
import numpy as np
import pandas as pd

def test_e12_test_leakage_and_delay():
    # Simulate a full timeline: Train (0-60), Val (60-90), Test (90-120)
    df = pd.DataFrame({
        'TransactionDT': np.arange(0, 120 * 86400, 86400)
    })
    
    W_7D = 7 * 86400
    test_start = 90 * 86400
    
    # Chunk 1 of Test: Day 90 to 97
    t_k = test_start
    t_next = t_k + W_7D
    
    idx_pred = df[(df['TransactionDT'] >= t_k) & (df['TransactionDT'] < t_next)].index
    idx_train = df[df['TransactionDT'] <= t_k - W_7D].index
    
    # 1. Prediction strictly starts at test boundary
    assert min(df.loc[idx_pred, 'TransactionDT']) == 90 * 86400
    
    # 2. Training strictly stops at t_k - 7 days (Day 83)
    assert max(df.loc[idx_train, 'TransactionDT']) == 83 * 86400
    
    # 3. Intersection is empty (No same-timestamp leakage)
    assert len(set(idx_pred).intersection(set(idx_train))) == 0
    
    # 4. No future transactions (Day > 90) leak into training
    assert max(df.loc[idx_train, 'TransactionDT']) < 90 * 86400

if __name__ == "__main__":
    pytest.main([__file__])
