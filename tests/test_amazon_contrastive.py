import os
import torch
import unittest
import numpy as np
from pipeline.features.amazon_graph_builder import AmazonGraphBuilder
from src.train_amazon_contrastive import sample_negatives

class TestAmazonContrastive(unittest.TestCase):
    def setUp(self):
        builder = AmazonGraphBuilder(os.path.join('.', 'data', 'amazon', 'raw'), max_records=1000)
        self.data = builder.build()
        
    def test_unobserved_negatives(self):
        # Verify sampled negatives are not observed interactions
        edge_index = self.data['user', 'reviews', 'product'].edge_index
        edge_attr = self.data['user', 'reviews', 'product'].edge_attr
        num_products = self.data['product'].num_nodes
        
        u_idx = edge_index[0, :10]
        p_idx = edge_index[1, :10]
        t_target = edge_attr[:10, 0]
        
        neg_u, neg_p = sample_negatives(u_idx, p_idx, t_target, edge_index, edge_attr, num_products, k=3)
        
        # Check against entire edge_index
        src, dst = edge_index
        for i in range(len(neg_u)):
            nu = neg_u[i].item()
            np_id = neg_p[i].item()
            u_edges = dst[src == nu]
            self.assertNotIn(np_id, u_edges.tolist(), f"Sampled negative ({nu}, {np_id}) is actually a true positive edge!")
            
    def test_train_val_separation(self):
        train_mask = self.data['user', 'reviews', 'product'].train_mask
        val_mask = self.data['user', 'reviews', 'product'].val_mask
        
        overlap = torch.sum(train_mask & val_mask)
        self.assertEqual(overlap.item(), 0, "Train and Validation masks overlap!")

if __name__ == '__main__':
    unittest.main()
