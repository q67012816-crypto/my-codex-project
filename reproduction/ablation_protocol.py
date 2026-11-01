"""Table-8 switch settings and fixed-probability sampling for diagnostic runs.

The existing baseline source/checkpoints are left unchanged.
"""
import torch
from torch.distributions.utils import clamp_probs
from models import GCN, Transformer
from paper_protocol import CONFIG, pretraining_loss

CASES = {'none': (0, 0, 0), 'attention': (0, 1, 1),
         'augmentation': (1, 0, 0), 'full': (1, 1, 1)}


def fixed_bernoulli(probs):
    """Forward hard sample of temperature-1 RelaxedBernoulliStraightThrough.

    Only for fixed probabilities without gradients. The original hard sample is
    round(sigmoid(logit(U)+logit(p))) = 1[U > 1-p]. Keep the same uniform RNG
    draws and endpoint clamps. No continuous surrogate is needed or provided.
    This must not be used for a future learnable augmenter.
    """
    if probs.requires_grad:
        raise ValueError('fixed sampler cannot replace a learnable distribution')
    p = clamp_probs(probs)
    u = clamp_probs(torch.rand(p.shape, dtype=p.dtype, device=p.device))
    return (u > (1-p)).to(p.dtype)


class FixedSamplingGCN(GCN):
    def forward(self, g, adj_org, x, B, edge_index, current_epoch, device_2):
        probs = (self.anfa * B + self.beta * adj_org).clamp(max=1).to(device_2)
        while True:
            sampled = fixed_bernoulli(probs).triu(1)
            sampled = sampled + sampled.T
            indices = sampled.to_sparse()._indices()
            # Preserve the upstream retry rule, including its isolated-node limitation.
            if indices.numel() and int(indices.max())+1 == g.num_nodes():
                return indices, x, sampled, True


def build_variant(in_features, classes, case, fast_sampling=True):
    aug, sim, phi = CASES[case]
    c = CONFIG
    model = Transformer(in_features, c.hidden_dim, classes, hidden_dim=c.hidden_dim,
                        num_layers=c.layers-1, num_heads=c.heads,
                        k_transition=c.transition_steps, aug_check=aug,
                        sim_check=sim, phi_check=phi, alfa=c.zeta, beta=c.xi,
                        dropout=c.dropout)
    if aug and fast_sampling:
        # Preserve initial weights and RNG: replacement consumes no lasting draws.
        with torch.random.fork_rng(devices=[]):
            sampler = FixedSamplingGCN(in_features, c.hidden_dim, c.zeta, c.xi)
        sampler.load_state_dict(model.gcn.state_dict())
        model.gcn = sampler
    assert_flags(model, case)
    return model


def assert_flags(model, case):
    aug, sim, phi = CASES[case]
    assert (model.aug_check, model.sim_check, model.phi_check) == (aug, sim, phi)
    assert len(model.layers) == CONFIG.layers
    for layer in model.layers:
        assert (layer.sim_check, layer.phi_check) == (sim, phi)
        assert (layer.attention.sim_check, layer.attention.phi_check) == (sim, phi)


def variant_loss(h, reconstructed, sampled, features, adjacency, targets):
    if sampled is not None:
        return pretraining_loss(h, reconstructed, sampled, features, adjacency, targets)
    cosine = torch.nn.functional.normalize(h, dim=1) @ torch.nn.functional.normalize(h, dim=1).T
    transition = sum((cosine-target).square().sum() for target in targets)
    feature = torch.linalg.vector_norm(reconstructed-features)/features.shape[0]
    augmentation = h.new_zeros(())
    total = CONFIG.beta1*transition + CONFIG.beta2*feature
    return total, dict(transition=transition, feature=feature, augmentation=augmentation)
