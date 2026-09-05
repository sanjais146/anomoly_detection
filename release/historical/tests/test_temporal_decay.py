import torch
import unittest
import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.tgat_v2 import TemporalDecay

class TestTemporalDecay(unittest.TestCase):
    
    def test_decay_scale(self):
        decay = TemporalDecay(init_tau=0.1) # Softplus to ~0.1
        
        sec_per_day = 86400.0
        # Delta_t in seconds
        delta_0 = torch.tensor([0.0]) # Same time (should be masked to 0)
        delta_1 = torch.tensor([1.0 * sec_per_day])
        delta_7 = torch.tensor([7.0 * sec_per_day])
        delta_30 = torch.tensor([30.0 * sec_per_day])
        delta_100 = torch.tensor([100.0 * sec_per_day])
        
        w0 = decay(delta_0).item()
        w1 = decay(delta_1).item()
        w7 = decay(delta_7).item()
        w30 = decay(delta_30).item()
        w100 = decay(delta_100).item()
        
        self.assertEqual(w0, 0.0, "Delta_t = 0 was not masked.")
        self.assertTrue(w1 > w7 > w30 > w100, "Decay is not monotonically decreasing.")
        
        # Verify specific expected value for 100 days (approx exp(-10))
        # Wait, if init_tau is exactly 0.1, tau is softplus(raw_tau) ~ 0.1
        # w100 = exp(-0.1 * 100) = exp(-10) = 4.5e-5
        # It should be > 0 and less than w30
        self.assertTrue(w100 > 0.0)
        
    def test_tau_gradient_and_positivity(self):
        decay = TemporalDecay(init_tau=0.1)
        
        # Force raw_tau negative
        decay.raw_tau.data = torch.tensor(-10.0)
        
        delta = torch.tensor([86400.0])
        w = decay(delta)
        
        # tau = softplus(-10) + eps ~ eps (1e-6)
        # weight = exp(-eps * 1) ~ 1.0
        self.assertTrue(w.item() > 0.99, "Tau did not stay positive!")
        
        w.backward()
        self.assertIsNotNone(decay.raw_tau.grad)
        
if __name__ == '__main__':
    unittest.main()
