import torch
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.tgat_v2 import TGAT

class TestCausalLeakage(unittest.TestCase):
    
    def setUp(self):
        self.in_channels = 8
        self.hidden_dim = 16
        self.model = TGAT(self.in_channels, self.hidden_dim, num_cards=2, num_devices=2, K=2)
        self.model.eval()
        
    def test_A_future_feature_perturbation(self):
        # Target Tx at t=100.
        # Historical Tx at t=50.
        # Future Tx at t=150.
        # All share Card 1.
        
        x_dict = {
            'transaction': torch.randn(3, self.in_channels)
        }
        
        # Tx 0: Hist, Tx 1: Target, Tx 2: Future
        # Edge indices for Tx -> Card
        edge_index_dict = {
            ('card', 'initiates', 'transaction'): torch.tensor([[1, 1, 1], [0, 1, 2]], dtype=torch.long),
            ('transaction', 'rev_initiates', 'card'): torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.long)
        }
        
        edge_time_dict = {
            ('transaction', 'rev_initiates', 'card'): torch.tensor([50, 100, 150], dtype=torch.long)
        }
        
        target_time_dict = {
            'transaction': torch.tensor([50, 100, 150], dtype=torch.long)
        }
        
        batch_tx_indices = torch.tensor([1], dtype=torch.long) # We are predicting for Target (Tx 1)
        
        # Base Output
        out_base = self.model(batch_tx_indices, x_dict, edge_index_dict, edge_time_dict, target_time_dict)
        
        # Perturb Future Tx (Tx 2)
        x_dict_perturbed = {
            'transaction': x_dict['transaction'].clone()
        }
        x_dict_perturbed['transaction'][2] += 100.0
        
        out_perturbed = self.model(batch_tx_indices, x_dict_perturbed, edge_index_dict, edge_time_dict, target_time_dict)
        
        self.assertTrue(torch.allclose(out_base, out_perturbed), "Future feature leaked into target prediction!")
        
    def test_B_target_self_leakage(self):
        # Change target's own features. Only the target's initial features should affect it.
        # It shouldn't leak through the card history.
        # We verify that if we isolate the GRU card representation, it remains unchanged when target features change.
        
        x_dict = {
            'transaction': torch.randn(2, self.in_channels) # Tx 0: Hist (t=50), Tx 1: Target (t=100)
        }
        
        edge_index_dict = {
            ('card', 'initiates', 'transaction'): torch.tensor([[1, 1], [0, 1]], dtype=torch.long),
            ('transaction', 'rev_initiates', 'card'): torch.tensor([[0, 1], [1, 1]], dtype=torch.long)
        }
        edge_time_dict = {
            ('transaction', 'rev_initiates', 'card'): torch.tensor([50, 100], dtype=torch.long)
        }
        target_time_dict = {
            'transaction': torch.tensor([50, 100], dtype=torch.long)
        }
        
        # Test just the card extractor for target Tx 1
        query_cards = torch.tensor([1], dtype=torch.long)
        query_times = torch.tensor([100], dtype=torch.long)
        
        card_out_base = self.model.card_seq_extractor(
            x_dict['transaction'], query_cards, query_times,
            edge_index_dict[('transaction', 'rev_initiates', 'card')],
            edge_time_dict[('transaction', 'rev_initiates', 'card')]
        )
        
        x_dict_perturbed = {
            'transaction': x_dict['transaction'].clone()
        }
        x_dict_perturbed['transaction'][1] += 100.0 # Perturb target
        
        card_out_perturbed = self.model.card_seq_extractor(
            x_dict_perturbed['transaction'], query_cards, query_times,
            edge_index_dict[('transaction', 'rev_initiates', 'card')],
            edge_time_dict[('transaction', 'rev_initiates', 'card')]
        )
        
        self.assertTrue(torch.allclose(card_out_base, card_out_perturbed), "Target's own features leaked into its historical context!")

    def test_C_same_time_exclusion(self):
        # Tx 0 and Tx 1 occur at t=100. Predicting for Tx 1 should exclude Tx 0.
        x_dict = {
            'transaction': torch.randn(2, self.in_channels)
        }
        edge_index_dict = {
            ('card', 'initiates', 'transaction'): torch.tensor([[1, 1], [0, 1]], dtype=torch.long),
            ('transaction', 'rev_initiates', 'card'): torch.tensor([[0, 1], [1, 1]], dtype=torch.long)
        }
        edge_time_dict = {
            ('transaction', 'rev_initiates', 'card'): torch.tensor([100, 100], dtype=torch.long)
        }
        target_time_dict = {
            'transaction': torch.tensor([100, 100], dtype=torch.long)
        }
        
        query_cards = torch.tensor([1], dtype=torch.long)
        query_times = torch.tensor([100], dtype=torch.long)
        
        card_out = self.model.card_seq_extractor(
            x_dict['transaction'], query_cards, query_times,
            edge_index_dict[('transaction', 'rev_initiates', 'card')],
            edge_time_dict[('transaction', 'rev_initiates', 'card')]
        )
        
        # Since both are at t=100 and Delta_t > 0 is enforced, the sequence is empty.
        # The output should be 0.
        self.assertTrue(torch.allclose(card_out, torch.zeros_like(card_out)), "Same-time transaction was not excluded!")

if __name__ == '__main__':
    unittest.main()
