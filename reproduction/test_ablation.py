import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,pyro,dgl
sys.path.insert(0,str(Path(__file__).parent/'vendor/cgt'))
import models
from ablation_protocol import CASES,build_variant,assert_flags,variant_loss,fixed_bernoulli


class AblationTests(unittest.TestCase):
    def test_fixed_sampler_values_rng_and_grad_guard(self):
        p=torch.tensor([0.,.05,.2,.8,.95,1.],dtype=torch.float64).repeat(100,1)
        for seed in range(42,52):
            torch.manual_seed(seed)
            expected=pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1,probs=p).rsample()
            state=torch.get_rng_state()
            torch.manual_seed(seed);actual=fixed_bernoulli(p)
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
        with self.assertRaises(ValueError):fixed_bernoulli(p.requires_grad_())

    def test_switches_control_executed_branches_and_survive_checkpoint(self):
        torch.set_num_threads(2)
        device=torch.device('cpu')
        edges=torch.tensor([[0,1,1,2,2,3,3,4,4,0],[1,0,2,1,3,2,4,3,0,4]])
        g=dgl.graph((edges[0],edges[1]),num_nodes=5)
        x=torch.randn(5,8);adj=torch.zeros(5,5,dtype=torch.float64)
        adj[edges[0],edges[1]]=1
        B=torch.zeros(5,5);sim=torch.randn(5,5,3);phi=torch.randn(5,5,1)
        g.ndata['x']=x;g.edata['sim']=sim[edges[0],edges[1]];g.edata['phi']=phi[edges[0],edges[1]]
        for case,(aug,att_sim,att_phi) in CASES.items():
            model=build_variant(8,3,case)
            with patch.object(model.gcn,'forward',wraps=model.gcn.forward) as sampling, \
                 patch.object(models,'src_dot_dst_sim',wraps=models.src_dot_dst_sim) as sim_calls, \
                 patch.object(models,'imp_add_attn',wraps=models.imp_add_attn) as phi_calls:
                h,recon,sampled=model(g.local_var(),adj,sim,phi,B,3,1,device,device)
                self.assertEqual(sampling.call_count,int(aug))
                self.assertEqual(sim_calls.call_count,4*att_sim)
                self.assertEqual(phi_calls.call_count,4*att_phi)
                self.assertEqual(sampled is None,not bool(aug))
                loss,terms=variant_loss(h,recon,sampled,x,adj,torch.zeros(3,5,5))
                self.assertTrue(torch.isfinite(loss));loss.backward()
                self.assertIsNotNone(model.embedding_h.weight.grad)
                if not aug:self.assertEqual(terms['augmentation'].item(),0.)
            with tempfile.TemporaryDirectory() as tmp:
                file=Path(tmp)/'pretrain.pt';torch.save(model,file)
                classifier=models.Transformer_class(8,64,3,64,4,4,'cora',str(file),aug,att_sim,att_phi)
                assert_flags(classifier.model,case)


if __name__=='__main__':unittest.main()
