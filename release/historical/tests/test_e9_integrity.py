import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from pipeline.features.e9_features import generate_e9_features

def test_e9_causality():
    df = pd.DataFrame({
        'TransactionID': [1, 2, 3, 4],
        'TransactionDT': [0, 86400, 86400*8, 86400*10],
        'isFraud': [1, 0, 1, 0],
        'card1': ['A', 'A', 'A', 'A'],
        'TransactionAmt': [10, 20, 30, 40]
    })
    
    feats = generate_e9_features(df, ['card1'], 'c1', delta_days=7)
    
    # Row 1 (t=86400): sees t=0, but delta=7 days means fraud is hidden
    assert feats.iloc[1]['c1_tx_all'] == 1.0
    assert feats.iloc[1]['c1_fc_all'] == 0.0 # Hidden!
    
    # Row 2 (t=8 days): sees t=0 fraud (since 8 days > 7 days)
    assert feats.iloc[2]['c1_tx_all'] == 2.0
    assert feats.iloc[2]['c1_fc_all'] == 1.0
    
    # Row 3 (t=10 days): sees t=0 fraud, t=8 is hidden (10-8 = 2 < 7)
    assert feats.iloc[3]['c1_fc_all'] == 1.0

if __name__ == "__main__":
    pytest.main([__file__])
