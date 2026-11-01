"""Small CPU CGT run, using upstream model and structural preprocessing."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor' / 'cgt'))
import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from models import Transformer, Transformer_class
from util import get_A_D, getM_logM, get_B_sim_phi, load_dgl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain-epochs', type=int, default=2)
    parser.add_argument('--finetune-epochs', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'smoke')
    args = parser.parse_args()
    if min(args.pretrain_epochs, args.finetune_epochs) < 1:
        parser.error('both training phases require at least one epoch')
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')
    data = Planetoid(root=str(ROOT / 'data'), name='Cora', split='public')[0]
    assert (data.num_nodes, data.num_features) == (2708, 1433)
    nx_g = to_networkx(data, to_undirected=True)
    adj, degree, _ = get_A_D(nx_g, data.num_nodes)
    cache = ROOT / 'cache' / 'cora-k3-upstream-v1.pt'
    if cache.exists():
        structural = torch.load(cache, map_location='cpu')
    else:
        transitions, logs = getM_logM(nx_g, data.num_nodes, kstep=3)
        B, sim, phi = get_B_sim_phi(nx_g, transitions, data.num_nodes,
                                     int(data.y.max()) + 1, data.x, kstep=3)
        structural = dict(logM=torch.tensor(np.array(logs), dtype=torch.float32),
                          B=torch.tensor(B, dtype=torch.float32),
                          sim=torch.tensor(sim, dtype=torch.float32),
                          phi=torch.tensor(phi, dtype=torch.float32))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(structural, cache)
    logM, B, sim, phi = (structural[k] for k in ['logM', 'B', 'sim', 'phi'])
    sim = torch.nan_to_num(sim, nan=0.0)
    graph = load_dgl(nx_g, data.x, sim, phi)
    adj = torch.tensor(adj)
    for name, value in [('logM', logM), ('B', B), ('sim', sim), ('phi', phi)]:
        assert torch.isfinite(value).all(), name

    # Match upstream make_masks(..., val_test_ratio=0.2), without its unused imports.
    held_out = np.random.choice(data.num_nodes, 2 * int(.2 * data.num_nodes), replace=False)
    val_idx, test_idx = np.split(held_out, 2)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros_like(val_mask)
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    train_mask = ~(val_mask | test_mask)
    assert not (val_mask & test_mask).any()
    masks = dict(train=train_mask, validation=val_mask, test=test_mask)
    model = Transformer(data.num_features, 64, 7, hidden_dim=64,
                        num_layers=4, num_heads=4, k_transition=3,
                        aug_check=1, sim_check=1, phi_check=1, alfa=.1, beta=.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad()
        h, reconstructed, sampled = model(graph, adj, sim, phi, B, 3, epoch, device, device)
        # Upstream cosine reconstruction, feature MSE and augmentation BCE losses.
        norms = torch.linalg.vector_norm(h, dim=1)
        cosine = (h @ h.T) / torch.outer(norms, norms)
        loss_m = sum((cosine - target).square().mean() for target in logM) / 3
        loss_x = F.mse_loss(reconstructed, data.x)
        loss_a = F.binary_cross_entropy_with_logits(sampled, adj)
        loss = loss_m + loss_x + loss_a
        assert torch.isfinite(loss), 'non-finite pretraining loss'
        loss.backward()
        assert model.embedding_h.weight.grad is not None
        assert torch.isfinite(model.embedding_h.weight.grad).all()
        optimizer.step()
        row = dict(phase='pretrain', epoch=epoch, loss=loss.item())
        history.append(row)
        print(json.dumps(row), flush=True)
    checkpoint = args.output / 'pretrained.pt'
    torch.save(model, checkpoint)
    classifier = Transformer_class(data.num_features, 64, 7, 64, 4, 4, 'cora',
                                   str(checkpoint), 1, 1, 1)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    best_acc, best_state, best_epoch = -1, None, None
    for epoch in range(1, args.finetune_epochs + 1):
        classifier.train()
        optimizer.zero_grad()
        scores = classifier(graph, adj, sim, phi, B, 3, epoch, device, device)
        # Preserve upstream softmax + cross_entropy behavior for this smoke run.
        loss = F.cross_entropy(scores[train_mask], data.y[train_mask])
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(classifier.MLP.mlp[0].weight.grad).all()
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            scores = classifier(graph, adj, sim, phi, B, 3, epoch, device, device)
            acc = (scores.argmax(1)[val_mask] == data.y[val_mask]).float().mean().item()
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = copy.deepcopy(classifier.state_dict())
        row = dict(phase='finetune', epoch=epoch, loss=loss.item(), validation_accuracy=acc)
        history.append(row)
        print(json.dumps(row), flush=True)
    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(graph, adj, sim, phi, B, 3, best_epoch, device, device).argmax(1)
    test_acc = (predictions[test_mask] == data.y[test_mask]).float().mean().item()
    torch.save(dict(model=best_state, predictions=predictions, labels=data.y,
                    degree=torch.tensor(degree), masks=masks), args.output / 'evaluation.pt')
    raw = ROOT / 'data' / 'Cora' / 'raw'
    raw_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(raw.glob('ind.cora.*'))}
    result = dict(model='CGT', upstream_commit='948f25477eb0253873ffdf6d38fcf53f258fdd83',
                  purpose='CPU smoke test; not a paper reproduction', dataset='Cora',
                  seed=args.seed, device='cpu', nodes=data.num_nodes, features=data.num_features,
                  directed_edges=data.num_edges, split_counts={k: int(v.sum()) for k,v in masks.items()},
                  pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
                  learning_rate=1e-4, hidden_dim=64, num_layers_argument=4, num_heads=4,
                  best_validation_epoch=best_epoch, best_validation_accuracy=best_acc,
                  test_accuracy=test_acc, delta_sp=None,
                  delta_sp_status='not implemented: Eq. (17) verified in uploaded paper; grouping tie policy pending',
                  elapsed_seconds=time.monotonic()-started, history=history,
                  versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                                pyg=torch_geometric.__version__, dgl=dgl.__version__),
                  raw_data_sha256=raw_hashes)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
