import torch
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.tgat_v2 import TGAT

class TestIssue6Leakage(unittest.TestCase):
    def test_target_self_leakage(self):
        in_channels = 8
        hidden_dim = 16
        model = TGAT(in_channels, hidden_dim, num_cards=2, num_devices=2)
        
        # 1 Target Transaction (t=100) and 1 Card.
        # The Transaction initiates the Card.
        tx_features_1 = torch.ones(1, in_channels) # We want to see if this leaks back to itself
        
        x_dict = {
            'transaction': tx_features_1,
            'card': torch.zeros(1, dtype=torch.long),
            'device': torch.zeros(1, dtype=torch.long)
        }
        
        edge_index_dict = {
            ('transaction', 'rev_initiates', 'card'): torch.tensor([[0], [0]], dtype=torch.long),
            ('card', 'initiates', 'transaction'): torch.tensor([[0], [0]], dtype=torch.long)
        }
        
        edge_time_dict = {
            ('transaction', 'rev_initiates', 'card'): torch.tensor([100], dtype=torch.long),
            ('card', 'initiates', 'transaction'): torch.tensor([100], dtype=torch.long)
        }
        target_time_dict = {'transaction': torch.tensor([100], dtype=torch.long)}
        
        # If allow_same_timestamp=False, Hop 2 (Card->Tx) will mask the edge to 0 weight, preventing leakage.
        # BUT let's check Hop 1 (Tx->Card) explicitly.
        model.eval()
        
        tx_x = model.tx_proj(x_dict['transaction'])
        card_x_init = model.card_emb(x_dict['card'].squeeze(-1))
        
        # Hop 1
        ei_t2c = edge_index_dict[('transaction', 'rev_initiates', 'card')]
        card_x = model.conv_tx_to_card((tx_x, card_x_init), ei_t2c)
        
        # Because we didn't pass target_time or edge_weight_temporal to Hop 1,
        # the transaction's features freely updated the card.
        # Let's change the transaction features and see if Card embedding changes.
        tx_features_2 = torch.zeros(1, in_channels)
        tx_x_2 = model.tx_proj(tx_features_2)
        card_x_2 = model.conv_tx_to_card((tx_x_2, card_x_init), ei_t2c)
        
        self.assertFalse(torch.allclose(card_x, card_x_2), "Hop 1 leaked! The target transaction altered the card state.")
        
        # Even though Hop 2 masks the edge (so the final logits might be safe if Delta_t=0 is blocked),
        # if the Target Transaction happened at T=100 and it reads from a Card that has a future transaction at T=150,
        # Hop 1 currently leaks the T=150 transaction into the Card, which then leaks into T=100 during Hop 2!
        # This confirms Issue 6 and 7!

if __name__ == '__main__':
    unittest.main()
