import math,unittest,torch
from loss_convention_control import normalized_reconstruction_loss


class LossConventionTest(unittest.TestCase):
    def test_hand_computed_normalizations_and_gradients(self):
        h=torch.eye(2,dtype=torch.float64,requires_grad=True)
        x=torch.tensor([[3.,0.],[0.,4.]],dtype=torch.float64,requires_grad=True)
        a=torch.tensor([[.75,.25],[.25,.75]],dtype=torch.float64)
        loss,terms=normalized_reconstruction_loss(h,x,a,torch.zeros_like(x),torch.eye(2,dtype=torch.float64),torch.zeros(2,2,2))
        self.assertAlmostEqual(terms['transition'].item(),.5)
        self.assertAlmostEqual(terms['feature'].item(),25/4)
        self.assertAlmostEqual(loss.item(),.5*.5+.5*25/4-4*math.log(.75))
        loss.backward();self.assertIsNotNone(x.grad);self.assertTrue(torch.isfinite(x.grad).all())


if __name__=='__main__':unittest.main()
