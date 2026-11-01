"""Compare reconstruction gradients on one shared, initial seed-42 forward."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('DGLBACKEND', 'pytorch')
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/cgt'))
from ablation_protocol import build_variant, variant_loss
from loss_convention_control import normalized_reconstruction_loss
from util import load_dgl


def main():
    torch.set_num_threads(2)
    data = Planetoid(root=str(ROOT / 'data'), name='Cora', split='public')[0]
    key = hashlib.sha256(
        (ROOT / 'vendor/cgt/util.py').read_bytes() + data.x.numpy().tobytes() +
        data.edge_index.numpy().tobytes() + json.dumps(dict(clusters=5, p=3), sort_keys=True).encode()
    ).hexdigest()[:16]
    cached = torch.load(ROOT / 'cache' / f'cora-paper-{key}.pt', map_location='cpu')
    logm, b, sim, phi = [cached[k] for k in ['logM', 'B', 'sim', 'phi']]
    sim = torch.nan_to_num(sim)
    graph = load_dgl(to_networkx(data, to_undirected=True), data.x, sim, phi)
    adj = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float64)
    adj[data.edge_index[0], data.edge_index[1]] = 1
    torch.manual_seed(42)
    model = build_variant(data.num_features, 7, 'full')
    model.train()
    device = torch.device('cpu')
    h, reconstructed, sampled = model(graph.local_var(), adj, sim, phi, b, 3, 1, device, device)
    report = dict(seed=42, parameter='embedding_h.weight',
                  scope='One shared initial forward; component gradients include beta=0.5; no optimizer step.',
                  profiles={})
    for name, function in [('raw_sum_flattened_L2_per_N', variant_loss),
                           ('transition_mean_feature_MSE', normalized_reconstruction_loss)]:
        _, terms = function(h, reconstructed, sampled, data.x, adj, logm)
        norms = {}
        for term in ['transition', 'feature']:
            gradient = torch.autograd.grad(.5 * terms[term], model.embedding_h.weight, retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
            norms[term] = float(torch.linalg.vector_norm(gradient))
        report['profiles'][name] = dict(
            loss_values={k: float(v) for k, v in terms.items()},
            weighted_component_gradient_norms=norms,
            transition_to_feature_gradient_norm_ratio=norms['transition'] / norms['feature'])
    baseline = json.loads((ROOT / 'runs/cora-baseline-10/seed-42/result.json').read_text())
    raw = report['profiles']['raw_sum_flattened_L2_per_N']['loss_values']
    assert all(value == baseline['history'][0][name] for name, value in raw.items())
    report['matches_baseline_first_forward_loss_terms'] = True
    (ROOT / 'results/loss-gradient-scale-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
