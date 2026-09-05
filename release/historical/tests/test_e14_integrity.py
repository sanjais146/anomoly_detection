import pytest
import numpy as np
import pandas as pd

def test_e14_adaptive_leakage():
    # Simulate a chronological timeline
    # Train: 0-60, Val: 60-90
    df = pd.DataFrame({
        'TransactionDT': np.arange(0, 90 * 86400, 86400),
        'Label': np.random.randint(0, 2, 90)
    })
    
    W_7D = 7 * 86400
    val_start = 60 * 86400
    
    # Chunk 2 of Val: Day 67 to Day 74
    t_start = val_start + W_7D
    t_end = t_start + W_7D
    
    idx_pred = df[(df['TransactionDT'] >= t_start) & (df['TransactionDT'] < t_end)].index
    
    # 1. Base training pool availability
    max_train_time = t_start - W_7D
    idx_train = df[df['TransactionDT'] <= max_train_time].index
    
    assert max(df.loc[idx_train, 'TransactionDT']) == 60 * 86400
    assert len(set(idx_pred).intersection(set(idx_train))) == 0
    
    # 2. Adaptive window (e.g., 30 days)
    W_30D = 30 * 86400
    idx_train_30d = df[(df['TransactionDT'] <= max_train_time) & (df['TransactionDT'] > max_train_time - W_30D)].index
    assert min(df.loc[idx_train_30d, 'TransactionDT']) > max_train_time - W_30D
    
    # 3. Dynamic Thresholding (using previous chunk)
    # The previous chunk was Day 60 to Day 67. We can only use labels up to Day 60 to score it!
    # Wait, if we are at Day 67, the labels for Day 60-67 are NOT available yet!
    # So we can't use Chunk 1's true labels to tune the threshold for Chunk 2!
    # We can only use the most recent *labeled* chunk. At Day 67, labels are available up to Day 60.
    # Therefore, the most recent tunable chunk is Day 53 to Day 60.
    recent_tunable_end = t_start - W_7D
    recent_tunable_start = recent_tunable_end - W_7D
    idx_tunable = df[(df['TransactionDT'] >= recent_tunable_start) & (df['TransactionDT'] < recent_tunable_end)].index
    
    assert max(df.loc[idx_tunable, 'TransactionDT']) < t_start - W_7D

if __name__ == "__main__":
    pytest.main([__file__])
