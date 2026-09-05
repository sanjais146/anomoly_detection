import pytest
import numpy as np
import sys
sys.path.insert(0, ".")

def test_e10_temporal_validation_splits():
    # Verify that chronologically splitting validation data doesn't mix time ranges
    val_times = np.array([100, 200, 300, 400, 500, 600])
    
    # Split into 3
    chunk_size = len(val_times) // 3
    early = val_times[:chunk_size]
    middle = val_times[chunk_size:2*chunk_size]
    late = val_times[2*chunk_size:]
    
    assert max(early) < min(middle)
    assert max(middle) < min(late)
    
if __name__ == "__main__":
    pytest.main([__file__])
