import os
import unittest
import numpy as np
from pipeline.features.ieee_graph_builder_final import IEEEGraphBuilderFinal

class TestFinalIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        builder = IEEEGraphBuilderFinal(K=5)
        cls.X, cls.y, cls.times, cls.train_idx, cls.val_idx, cls.test_idx, cls.card_histories = builder.load_and_preprocess()
        
    def test_chronological_splits(self):
        train_times = self.times[self.train_idx]
        val_times = self.times[self.val_idx]
        test_times = self.times[self.test_idx]
        
        # Max train time <= Min val time
        self.assertLessEqual(train_times.max(), val_times.min())
        # Max val time <= Min test time
        self.assertLessEqual(val_times.max(), test_times.min())
        
    def test_strict_historical_causality(self):
        # Sample some transactions that have histories
        for i in np.random.choice(len(self.times), 1000):
            hist_indices = self.card_histories[i]
            target_time = self.times[i]
            
            for h_idx in hist_indices:
                if h_idx != -1: # -1 is padding
                    hist_time = self.times[h_idx]
                    # STRICTLY less than (<), preventing future or same-timestamp leakage
                    self.assertLess(hist_time, target_time, "Found non-strict history (future or same-timestamp)!")
                    
    def test_target_excluded_from_history(self):
        for i in np.random.choice(len(self.times), 1000):
            hist_indices = self.card_histories[i]
            self.assertNotIn(i, hist_indices, "Target transaction appears in its own history!")

if __name__ == '__main__':
    unittest.main()
