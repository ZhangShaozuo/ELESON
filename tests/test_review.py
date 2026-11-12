import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from eleson.model import ELESON, objective


class ReviewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_equations_and_batch_denominators(self):
        y = torch.tensor([0,0,0,1,1,2])
        x = torch.zeros(6,20,7)
        zc = torch.tensor([[0.],[1.],[2.],[2.],[4.],[9.]], requires_grad=True)
        e = torch.tensor([[1.,2.,3.]]).repeat(6,1).requires_grad_()
        out = dict(evidence=e, probability=e/e.sum(-1,keepdim=True), zc=zc,
            reconstruction=x[:,:,:6].clone(), combined_logits=torch.zeros(6,3),
            control_logits=torch.zeros(6,3), behavior_logits=torch.zeros(6,2))
        loss, terms = objective(out,x,y,torch.zeros(6,dtype=torch.long))
        expected_con = ((2/3)*3+1*2)/6
        self.assertAlmostEqual(float(terms['constraint']),expected_con,places=6)
        self.assertAlmostEqual(float(terms['behavior_ce']),math.log(2)*3/6,places=6)
        alpha = np.array([2.,3.,4.]); total=alpha.sum()
        var = np.sum(alpha*(total-alpha)/(total**2*(total+1)))
        self.assertAlmostEqual(float(terms['dirichlet_variance']),var,places=6)
        expected = F.nll_loss(out['probability'].log(),y).item()+var+2*math.log(3)+.3*expected_con+.4*math.log(2)*.5
        self.assertAlmostEqual(float(loss.detach()),expected,places=6)
        self.assertAlmostEqual(float(terms['optimization_loss']-terms['encoder_objective']),.8*math.log(2)*.5,places=6)
        _, legacy = objective(out,x,y,torch.zeros(6,dtype=torch.long),con_reduction='class',adv_reduction='conveyor')
        self.assertAlmostEqual(float(legacy['constraint']),((2/3)+1)/2,places=6)
        self.assertAlmostEqual(float(legacy['behavior_ce']),math.log(2),places=6)

    def test_configured_backbones_and_checkpoint_roundtrip(self):
        for backbone in ('convlstm','cnn_lstm'):
            torch.manual_seed(12)
            m = ELESON(samples=20,feature_dim=7,hidden=4,frame_samples=5,
                kernel_size=5,encoder_width=13,generator_widths=[11,17],
                evidence_widths=[9,8],adversary_width=5,intervention_width=6,backbone=backbone)
            x = torch.randn(3,20,7)
            out = m(x,auxiliary=True)
            loss,_ = objective(out,x,torch.tensor([0,1,2]),torch.tensor([0,1,0]))
            loss.backward()
            m.eval()
            restored = ELESON(**m.config)
            restored.load_state_dict(m.state_dict())
            restored.eval()
            torch.testing.assert_close(m(x)['confidence'],restored(x)['confidence'])

    def test_invalid_kernel_rejected(self):
        with self.assertRaises(ValueError):
            ELESON(kernel_size=2)


if __name__ == '__main__':
    unittest.main()
