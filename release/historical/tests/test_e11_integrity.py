import pytest
import numpy as np
import pandas as pd

def test_e11_chronological_split():
    # Simulate transaction times
    df = pd.DataFrame({
        'TransactionDT': np.arange(0, 100 * 86400, 86400) # 100 days
    })
    
    W_7D = 7 * 86400
    val_start = 70 * 86400
    
    # Predict chunk 1: Day 70 to 76
    t_k = val_start
    t_next = t_k + W_7D
    
    # Target prediction indices
    idx_pred = df[(df['TransactionDT'] >= t_k) & (df['TransactionDT'] < t_next)].index
    
    # Allowed training indices (must be at least 7 days before chunk start)
    idx_train = df[df['TransactionDT'] <= t_k - W_7D].index
    
    assert min(df.loc[idx_pred, 'TransactionDT']) == 70 * 86400
    assert max(df.loc[idx_train, 'TransactionDT']) == 63 * 86400
    assert len(set(idx_pred).intersection(set(idx_train))) == 0

if __name__ == "__main__":
    pytest.main([__file__])
