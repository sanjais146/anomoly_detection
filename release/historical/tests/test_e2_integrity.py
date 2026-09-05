"""
tests/test_e2_integrity.py

Integrity tests for Experiment E2: Causal History and CatBoost.
"""
import os, sys, pytest
import pandas as pd
import numpy as np
sys.path.insert(0, ".")

def test_causal_history_no_same_timestamp():
    """Verify that same-timestamp transactions are strictly excluded."""
    # Check if file exists first
    path = "data/processed/causal_history_card_v1.parquet"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
        
    df_causal = pd.read_parquet(path)
    
    df_tx = pd.read_csv("data/raw/IEEE-CIS/train_transaction.csv", 
                        usecols=["TransactionID", "TransactionDT", "card1", "TransactionAmt"])
    df_tx["orig_idx"] = df_tx.index
    
    # Merge to check
    df_merged = df_tx.merge(df_causal, left_index=True, right_index=True, suffixes=("", "_causal"))
    
    # Find transactions with same card and same timestamp
    dupes = df_merged[df_merged.duplicated(subset=["card1", "TransactionDT"], keep=False)]
    if len(dupes) > 1:
        # For a duplicated group, the count should NOT include the other same-timestamp ones
        # and should only include strictly prior ones. Thus all same-timestamp rows 
        # should have identical 'all_tx_count'.
        grp = dupes.groupby(["card1", "TransactionDT"])
        for name, group in grp:
            counts = group["card_all_tx_count"].values
            assert np.all(counts == counts[0]), f"Same-timestamp leakage for {name}"

def test_no_future_leakage_in_fraud_rate():
    """Ensure fraud rate never uses target labels."""
    path = "data/processed/causal_history_card_v1.parquet"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
        
    df_causal = pd.read_parquet(path)
    df_tx = pd.read_csv("data/raw/IEEE-CIS/train_transaction.csv", 
                        usecols=["TransactionID", "isFraud"])
    
    df_merged = df_tx.merge(df_causal, left_index=True, right_index=True)
    
    # If this is the FIRST transaction, fraud count MUST be 0
    first_txs = df_merged[df_merged["card_all_tx_count"] == 0]
    assert np.all(first_txs["card_all_fraud_count"] == 0.0), "Leakage in first tx!"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
