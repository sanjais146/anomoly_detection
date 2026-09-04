import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")

def test_e7_strict_causality():
    from pipeline.features.e6_delayed_causal import generate_e6_features
    
    df = pd.DataFrame({
        'TransactionID': [1, 2, 3, 4, 5, 6, 7],
        'TransactionDT': [0, 10, 10, 86400*7 + 1, 86400*8, 86400*15, 86400*20],
        'isFraud':       [1, 0, 1, 1,           0,         1,        0],
        'entity':        ['A', 'A', 'A', 'A', 'A', 'A', 'A'],
        'TransactionAmt':[10, 20, 30, 40, 50, 60, 70]
    })
    
    features = generate_e6_features(df, ['entity'], 'ent', delta_days=7)
    
    # 1st row has id=1, dt=0. Row 2 has id=2, dt=10.
    assert features.iloc[1]['ent_tx_count_all'] == 1.0  
    assert features.iloc[1]['ent_fraud_count_all'] == 0.0 
    
    # Row 3 (t=10, id=3): Same timestamp as id=2. Should NOT see id=2.
    assert features.iloc[2]['ent_tx_count_all'] == 1.0 
    
    # Row 4 (t=7 days + 1s, id=4): Now t=0 label is >= 7 days old!
    assert features.iloc[3]['ent_fraud_count_all'] == 1.0 
    
    # Row 6 (t=15 days, id=6): Should see labels from t=0, t=10 (both), t=7d
    assert features.iloc[5]['ent_fraud_count_all'] == 3.0 
    
    assert features.iloc[3]['ent_tx_count_all'] == 3.0

if __name__ == "__main__":
    pytest.main([__file__])
