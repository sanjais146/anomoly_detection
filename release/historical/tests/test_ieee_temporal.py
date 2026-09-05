import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.features.ieee_graph_builder import calculate_temporal_edge_weights

class TestTemporalGraph(unittest.TestCase):
    
    def test_temporal_weighting_decay(self):
        target_t = torch.tensor([1000, 1000, 1000])
        # Edges occurred at 1000 (simultaneous), 900 (recent), 100 (old)
        edge_t = torch.tensor([1000, 900, 100])
        
        tau = 0.01
        weights = calculate_temporal_edge_weights(target_t, edge_t, tau)
        
        self.assertAlmostEqual(weights[0].item(), 1.0, places=4) # e^0 = 1
        self.assertTrue(weights[1].item() > weights[2].item()) # Recent > Old
        self.assertTrue(weights[2].item() > 0)
        
    def test_future_leakage_masking(self):
        target_t = torch.tensor([1000])
        edge_t = torch.tensor([1500]) # Occurs in the future!
        
        tau = 0.01
        weights = calculate_temporal_edge_weights(target_t, edge_t, tau)
        
        # Must be exactly 0
        self.assertEqual(weights[0].item(), 0.0)

if __name__ == '__main__':
    unittest.main()
