import math
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor/cgt'))
from paper_protocol import CONFIG, build_model, degree_statistical_parity, pretraining_loss


class ProtocolTests(unittest.TestCase):
    def test_multiclass_macro_frequency_gap(self):
        # Low frequencies (2/3,1/3,0), high (1/3,0,2/3): mean gap = 4/9.
        out = degree_statistical_parity([0,0,1,0,2,2], [0,1,2,0,1,2],
                                       np.arange(6), np.ones(6, bool), fraction=.5)
        self.assertAlmostEqual(out['delta_sp'], 4/9)
        self.assertAlmostEqual(out['delta_sp_percent'], 400/9)

    def test_groups_are_selected_only_from_test_candidates(self):
        out = degree_statistical_parity(np.zeros(12), np.zeros(12),
                                       [-100,100] + list(range(10)),
                                       np.array([False,False]+[True]*10))
        self.assertEqual(out['low_node_ids'], [2,3])
        self.assertEqual(out['high_node_ids'], [10,11])

    def test_ties_rounding_and_no_overlap(self):
        out = degree_statistical_parity(np.zeros(11), np.zeros(11),
                                       np.ones(11), np.ones(11, bool))
        self.assertEqual(out['group_size'], 2)
        self.assertEqual(out['low_node_ids'], [0,1])
        self.assertEqual(out['high_node_ids'], [9,10])

    def test_parity_is_not_accuracy_difference(self):
        # The two groups have different accuracy but identical predictions.
        out = degree_statistical_parity(np.zeros(10), [0]*5+[1]*5,
                                       np.arange(10), np.ones(10, bool))
        self.assertEqual(out['delta_sp'], 0.)

    def test_empty_groups_fail_explicitly(self):
        with self.assertRaises(ValueError):
            degree_statistical_parity([0,0], [0,0], [1,2], np.ones(2, bool))

    def test_paper_loss_values_and_gradients(self):
        h = torch.eye(2, dtype=torch.float64, requires_grad=True)
        reconstructed = torch.tensor([[3.,0.],[0.,4.]], dtype=torch.float64, requires_grad=True)
        sampled = torch.tensor([[.75,.25],[.25,.75]], dtype=torch.float64, requires_grad=True)
        total, terms = pretraining_loss(h, reconstructed, sampled,
                                       torch.zeros_like(reconstructed), torch.eye(2, dtype=torch.float64),
                                       torch.zeros((2,2,2), dtype=torch.float64))
        # Two transitions * ||I||_F^2=4; feature norm 5/N=2.5; BCE=-4*log(.75).
        expected = .5*4 + .5*2.5 - 4*math.log(.75)
        self.assertAlmostEqual(total.item(), expected)
        self.assertAlmostEqual(terms['augmentation'].item(), -4*math.log(.75))
        total.backward()
        self.assertGreater(reconstructed.grad.abs().sum().item(), 0)
        self.assertGreater(sampled.grad.abs().sum().item(), 0)

    def test_actual_architecture_and_probability_coefficients(self):
        model = build_model(8, 3)
        self.assertEqual(len(model.layers), 4)
        self.assertTrue(all(x.num_heads == 4 and x.dropout == .1 for x in model.layers))
        self.assertEqual(model.gcn.anfa, .2)  # D coefficient, not A coefficient
        self.assertEqual(model.gcn.beta, .8)
        # Optional dropout must preserve the previous runner's default.
        from models import GraphTransformerLayer
        self.assertEqual(GraphTransformerLayer(64,64,4,1,1).dropout, .5)


if __name__ == '__main__':
    unittest.main()
