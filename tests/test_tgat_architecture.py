import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.tgat_v2 import TGAT, TemporalEdgeWeighting

class TestTGATArchitecture(unittest.TestCase):
    
    def setUp(self):
        self.in_channels_tx = 16
        self.hidden_dim = 32
        self.num_cards = 100
        self.num_devices = 50
        
        self.model = TGAT(self.in_channels_tx, self.hidden_dim, self.num_cards, self.num_devices)
        
        # Create dummy HeteroData components
        self.x_dict = {
            'transaction': torch.randn(5, self.in_channels_tx, requires_grad=True),
            'card': torch.tensor([[1], [2], [3]], dtype=torch.long),
            'device': torch.tensor([[1], [2]], dtype=torch.long)
        }
        
        self.edge_index_dict = {
            ('transaction', 'rev_initiates', 'card'): torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            ('card', 'initiates', 'transaction'): torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        }
        
        # Transaction 0 occurs at t=100. Edge occurred at t=50 (valid).
        # Transaction 1 occurs at t=100. Edge occurred at t=100 (same time).
        # Transaction 2 occurs at t=100. Edge occurred at t=150 (future).
        self.edge_time_dict = {
            ('card', 'initiates', 'transaction'): torch.tensor([50, 100, 150], dtype=torch.long)
        }
        
        self.target_time_dict = {
            'transaction': torch.tensor([100, 100, 100, 100, 100], dtype=torch.long)
        }

    def test_forward_pass_and_shape(self):
        logits = self.model(self.x_dict, self.edge_index_dict, self.edge_time_dict, self.target_time_dict)
        self.assertEqual(logits.shape, (5, 1))
        self.assertTrue(torch.isfinite(logits).all())
        
    def test_gradient_flow(self):
        logits = self.model(self.x_dict, self.edge_index_dict, self.edge_time_dict, self.target_time_dict)
        loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        loss.backward()
        
        # Check gradients flow through Tx projection (inputs)
        self.assertIsNotNone(self.x_dict['transaction'].grad)
        
        # Check gradients flow through GRU
        self.assertIsNotNone(self.model.gru.weight_ih)
        self.assertTrue(torch.abs(self.model.gru.weight_ih.grad).sum() > 0)
        
        # Check gradients flow through TemporalGATConv
        self.assertIsNotNone(self.model.conv_card_to_tx.lin_src.weight.grad)
        self.assertTrue(torch.abs(self.model.conv_card_to_tx.lin_src.weight.grad).sum() > 0)
        
    def test_temporal_weighting_logic(self):
        weighting = TemporalEdgeWeighting(tau=0.01, allow_same_timestamp=False)
        target_t = torch.tensor([100, 100, 100, 100])
        edge_t = torch.tensor([50, 90, 100, 150]) # old, recent, same-time, future
        
        w = weighting(edge_t, target_t)
        
        # Valid historical edges
        self.assertTrue(w[0] > 0 and w[1] > 0)
        
        # Decay (Larger Delta_t produces lower weight) -> 50 < 90 in target-edge (50 vs 10)
        # weight(50) = exp(-0.01*50) = ~0.60
        # weight(90) = exp(-0.01*10) = ~0.90
        self.assertTrue(w[1] > w[0])
        
        # Same-time and future edges excluded by default
        self.assertEqual(w[2].item(), 0.0)
        self.assertEqual(w[3].item(), 0.0)

import torch.nn.functional as F

if __name__ == '__main__':
    unittest.main()
