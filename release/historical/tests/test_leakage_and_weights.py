import torch
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.tgat_v2 import TGAT, TemporalEdgeWeighting

class TestDeepArchitecture(unittest.TestCase):
    
    def test_issue_5_temporal_weight_attention(self):
        # Create identical GAT inputs but different delta_t
        in_channels = 8
        hidden_dim = 16
        heads = 1
        model = TGAT(in_channels, hidden_dim, num_cards=10, num_devices=10, tau=0.01)
        
        # Two target transactions, both reading from the same card.
        # Tx 0 reads from Card 0 at Delta_t = 10
        # Tx 1 reads from Card 0 at Delta_t = 100
        x_dict = {
            'transaction': torch.ones(2, in_channels),
            'card': torch.zeros(1, dtype=torch.long),
            'device': torch.zeros(1, dtype=torch.long)
        }
        
        edge_index_dict = {
            ('card', 'initiates', 'transaction'): torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
            ('transaction', 'rev_initiates', 'card'): torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
        }
        
        # Edge times
        # Card -> Tx0 occurred at 90 (Target is 100 -> delta = 10)
        # Card -> Tx1 occurred at 0 (Target is 100 -> delta = 100)
        edge_time_dict = {
            ('card', 'initiates', 'transaction'): torch.tensor([90, 0], dtype=torch.long),
            ('transaction', 'rev_initiates', 'card'): torch.tensor([90, 0], dtype=torch.long)
        }
        
        target_time_dict = {
            'transaction': torch.tensor([100, 100], dtype=torch.long)
        }
        
        # Forward pass
        # Instead of the full model which has multiple hops and randomness, 
        # let's just test the TemporalGATConv explicitly.
        conv = model.conv_card_to_tx
        conv.eval() # Turn off dropout
        
        w = model.temporal_weighting(edge_time_dict[('card', 'initiates', 'transaction')], target_time_dict['transaction'])
        
        card_x = torch.ones(1, hidden_dim)
        tx_x = torch.ones(2, hidden_dim)
        
        out = conv((card_x, tx_x), edge_index_dict[('card', 'initiates', 'transaction')], edge_weight_temporal=w)
        
        # Out for Tx 0 should be larger than Out for Tx 1 because Delta_t is smaller
        mag_0 = torch.norm(out[0]).item()
        mag_1 = torch.norm(out[1]).item()
        
        self.assertTrue(mag_0 > mag_1, "Message magnitude did not decay with larger Delta_t!")

if __name__ == '__main__':
    unittest.main()
