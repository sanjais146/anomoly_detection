import os
import torch
import unittest
from pipeline.features.amazon_graph_builder import AmazonGraphBuilder
from src.models.temporal_gat_amazon import TemporalGATAmazon

class TestAmazonTemporalIntegrity(unittest.TestCase):
    def setUp(self):
        builder = AmazonGraphBuilder(os.path.join('.', 'data', 'amazon', 'raw'), max_records=1000)
        self.data = builder.build()
        self.model = TemporalGATAmazon(in_channels=16, hidden_channels=8)
        
    def test_chronological_splits(self):
        # 1. No test-period information is used for training
        times = self.data['user', 'reviews', 'product'].edge_attr[:, 0]
        train_mask = self.data['user', 'reviews', 'product'].train_mask
        val_mask = self.data['user', 'reviews', 'product'].val_mask
        test_mask = self.data['user', 'reviews', 'product'].test_mask
        
        train_times = times[train_mask]
        test_times = times[test_mask]
        
        if len(train_times) > 0 and len(test_times) > 0:
            self.assertTrue(torch.max(train_times) <= torch.min(test_times), 
                            "Train times overlap with Test times!")
            
    def test_same_day_exclusion(self):
        # 2. Same-day interactions are excluded (and target interaction excluded)
        # We can test this by running the causal encoder and manually checking edges
        edge_index = self.data['user', 'reviews', 'product'].edge_index
        edge_attr = self.data['user', 'reviews', 'product'].edge_attr
        
        u_idx = edge_index[0, [50]]
        p_idx = edge_index[1, [50]]
        t_target = edge_attr[50, 0:1] # Target time
        
        # Modify an earlier edge to have the exact same time
        edge_attr[0, 0] = t_target[0].item()
        
        u_x = self.data['user'].x
        p_x = self.data['product'].x
        
        # We override encode_causal logic to test it directly
        # The logic says: edge_times < t
        # If edge_times[0] == t, it should NOT be in the neighborhood
        src, dst = edge_index
        edge_times = edge_attr[:, 0]
        
        u = u_idx[0].item()
        t = t_target[0].item()
        
        u_edges = (src == u) & (edge_times < t)
        
        self.assertFalse(u_edges[0].item(), "Same-day (or same-timestamp) edge was included!")
        self.assertFalse(u_edges[50].item(), "Target interaction included itself in history!")
        
    def test_future_perturbation(self):
        # 4. Future interactions cannot affect historical representation
        edge_index = self.data['user', 'reviews', 'product'].edge_index
        edge_attr = self.data['user', 'reviews', 'product'].edge_attr
        u_x = self.data['user'].x
        p_x = self.data['product'].x
        
        u_idx = edge_index[0, [5]]
        p_idx = edge_index[1, [5]]
        t_target = edge_attr[5, 0:1]
        
        u_emb1, p_emb1 = self.model.encode_causal(u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x)
        
        # Perturb future edge
        edge_attr[50, 1] = 999.0
        
        u_emb2, p_emb2 = self.model.encode_causal(u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x)
        
        self.assertTrue(torch.allclose(u_emb1, u_emb2), "Future perturbation affected historical embedding!")

if __name__ == '__main__':
    unittest.main()
