import pytest
import numpy as np
import torch
import sys
sys.path.insert(0, ".")

def test_e8_causal_neighborhoods():
    # Mock data
    # t = [0, 10, 10, 86400*7+1]
    timestamps = np.array([0, 10, 10, 86400*7 + 1])
    entities = np.array([1, 1, 1, 1])
    labels = np.array([1, 0, 1, 1])
    
    # Simple neighborhood builder logic we'll use in the Dataset
    def build_neighborhood(target_idx, t_arr, ent_arr, lab_arr, delta_t=7*86400):
        t_target = t_arr[target_idx]
        ent_target = ent_arr[target_idx]
        
        hist_indices = []
        hist_labels = []
        
        for i in range(target_idx - 1, -1, -1):
            if ent_arr[i] == ent_target and t_arr[i] < t_target:
                hist_indices.append(i)
                # Apply 7-day delay
                if t_target - t_arr[i] >= delta_t:
                    hist_labels.append(lab_arr[i])
                else:
                    hist_labels.append(-1) # Masked
        return hist_indices, hist_labels

    # Target 0: t=0
    idx, lab = build_neighborhood(0, timestamps, entities, labels)
    assert len(idx) == 0

    # Target 1: t=10
    idx, lab = build_neighborhood(1, timestamps, entities, labels)
    assert len(idx) == 1 # Only row 0
    assert idx[0] == 0
    assert lab[0] == -1 # Masked due to delay

    # Target 2: t=10 (same timestamp)
    idx, lab = build_neighborhood(2, timestamps, entities, labels)
    assert len(idx) == 1 # Should NOT see row 1 because t_arr[1] == 10 >= t_target (10)
    assert idx[0] == 0

    # Target 3: t=7 days + 1s
    idx, lab = build_neighborhood(3, timestamps, entities, labels)
    assert len(idx) == 3 # Should see row 2, 1, 0
    assert idx[2] == 0
    assert lab[2] == 1 # Unmasked because 7d passed since t=0

    # Test future leaks
    assert all(i < 3 for i in idx)

if __name__ == "__main__":
    pytest.main([__file__])
