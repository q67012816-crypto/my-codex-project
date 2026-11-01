"""Validation-only coefficient search; test scores are computed only after selection."""
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
from paper_protocol import CONFIG, config_dict, degree_statistical_parity
from ablation_protocol import CASES, build_variant, assert_flags, variant_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--xi', type=float, choices=[.2,.4,.6,.8], required=True)
    parser.add_argument('--zeta', type=float, choices=[.2,.4,.6,.8], required=True)
    parser.add_argument('--case', choices=list(CASES), required=True)
    parser.add_argument('--p', type=int, choices=[3,4], default=3)
    parser.add_argument('--sampler', choices=['fast','reference'], default='fast')
    parser.add_argument('--pretrain-epochs', type=int, default=500)
    parser.add_argument('--finetune-epochs', type=int, default=200)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'validation-grid-smoke')
    args = parser.parse_args()
    if args.case != "full" or args.p != 3 or args.sampler != "fast":
        parser.error("grid fixes full model, p=3 and the verified fixed sampler")
    effective_config = {**config_dict(), "xi": args.xi, "zeta": args.zeta}
    if min(args.pretrain_epochs, args.finetune_epochs) < 1:
        parser.error('both training phases require at least one epoch')
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if (args.output / 'result.json').exists():
        parser.error('output already contains a completed run; use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    def progress(phase, epoch, total, **metrics):
        status = dict(seed=args.seed, phase=phase, epoch=epoch, total_epochs=total,
                      elapsed_seconds=time.monotonic()-started, **metrics)
        temporary = args.output / 'progress.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(args.output / 'progress.json')
    progress('preprocessing', 0, 0)
    device = torch.device('cpu')
    data = Planetoid(root=str(ROOT / 'data'), name='Cora', split='public')[0]
    assert (data.num_nodes, data.num_features) == (2708, 1433)
    nx_g = to_networkx(data, to_undirected=True)
    adj, degree, _ = get_A_D(nx_g, data.num_nodes)
    cache_key = hashlib.sha256(
        (ROOT / 'vendor/cgt/util.py').read_bytes() +
        data.x.numpy().tobytes() + data.edge_index.numpy().tobytes() +
        json.dumps(dict(clusters=CONFIG.clusters, p=CONFIG.transition_steps), sort_keys=True).encode()
    ).hexdigest()[:16]
    cache = ROOT / 'cache' / f'cora-paper-{cache_key}.pt'
    if cache.exists():
        structural = torch.load(cache, map_location='cpu')
    else:
        transitions, logs = getM_logM(nx_g, data.num_nodes, kstep=CONFIG.transition_steps)
        B, sim, phi = get_B_sim_phi(nx_g, transitions, data.num_nodes,
                                     CONFIG.clusters, data.x, kstep=CONFIG.transition_steps)
        structural = dict(logM=torch.tensor(np.array(logs), dtype=torch.float32),
                          B=torch.tensor(B, dtype=torch.float32),
                          sim=torch.tensor(sim, dtype=torch.float32),
                          phi=torch.tensor(phi, dtype=torch.float32))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(structural, cache)
    logM, B, sim, phi = (structural[k] for k in ['logM', 'B', 'sim', 'phi'])
    if args.p != len(logM):
        targets_path = ROOT / 'cache' / f'cora-targets-p{args.p}-{cache_key}.pt'
        if targets_path.exists():
            targets = torch.load(targets_path, map_location='cpu')
        else:
            _, logs = getM_logM(nx_g, data.num_nodes, kstep=args.p)
            targets = torch.tensor(np.array(logs), dtype=torch.float32)
            torch.save(targets, targets_path)
        assert torch.equal(targets[:len(logM)], logM), 'p change altered earlier transition targets'
        logM = targets
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
    model = build_variant(data.num_features, int(data.y.max()) + 1, args.case, args.sampler=='fast')
    model.afla, model.beta = args.zeta, args.xi
    model.gcn.anfa, model.gcn.beta = args.zeta, args.xi
    assert model.gcn.anfa == args.zeta and model.gcn.beta == args.xi
    print(json.dumps(dict(effective_config=effective_config, actual_layers=len(model.layers)), indent=2), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad()
        h, reconstructed, sampled = model(graph.local_var(), adj, sim, phi, B, 3, epoch, device, device)
        loss, terms = variant_loss(h, reconstructed, sampled, data.x, adj, logM)
        augmentation_has_gradient = terms['augmentation'].requires_grad
        assert torch.isfinite(loss), 'non-finite pretraining loss'
        loss.backward()
        assert model.embedding_h.weight.grad is not None
        assert torch.isfinite(model.embedding_h.weight.grad).all()
        optimizer.step()
        row = dict(phase='pretrain', epoch=epoch, loss=loss.item(),
                   **{name: value.item() for name,value in terms.items()},
                   augmentation_has_gradient=augmentation_has_gradient)
        history.append(row)
        progress('pretrain', epoch, args.pretrain_epochs, loss=loss.item())
        print(json.dumps(row), flush=True)
    checkpoint = args.output / 'pretrained.pt'
    torch.save(model, checkpoint)
    classifier = Transformer_class(data.num_features, 64, 7, 64, 4, 4, 'cora',
                                   str(checkpoint), *CASES[args.case])
    assert_flags(classifier.model, args.case)
    assert classifier.model.gcn.anfa == args.zeta and classifier.model.gcn.beta == args.xi
    optimizer = torch.optim.Adam(classifier.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    best_acc, best_state, best_epoch = -1, None, None
    for epoch in range(1, args.finetune_epochs + 1):
        classifier.train()
        optimizer.zero_grad()
        scores = classifier(graph.local_var(), adj, sim, phi, B, 3, epoch, device, device)
        # Preserve upstream softmax + cross_entropy behavior for this smoke run.
        loss = F.cross_entropy(scores[train_mask], data.y[train_mask])
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(classifier.MLP.mlp[0].weight.grad).all()
        optimizer.step()
        classifier.eval()
        with torch.no_grad():
            scores = classifier(graph.local_var(), adj, sim, phi, B, 3, epoch, device, device)
            acc = (scores.argmax(1)[val_mask] == data.y[val_mask]).float().mean().item()
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = copy.deepcopy(classifier.state_dict())
        row = dict(phase='finetune', epoch=epoch, loss=loss.item(), validation_accuracy=acc)
        history.append(row)
        progress('finetune', epoch, args.finetune_epochs, loss=loss.item(), validation_accuracy=acc)
        print(json.dumps(row), flush=True)
    classifier.load_state_dict(best_state)
    classifier.eval()
    torch.save(dict(model=classifier, masks=masks, degree=torch.tensor(degree),
                    rng_state=torch.get_rng_state(), seed=args.seed,
                    xi=args.xi, zeta=args.zeta), args.output / 'validation-selected.pt')
    raw = ROOT / 'data' / 'Cora' / 'raw'
    result = dict(model='validation-only coefficient grid', case='full', seed=args.seed,
                  xi=args.xi, zeta=args.zeta, effective_config=effective_config,
                  pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
                  best_validation_epoch=best_epoch, best_validation_accuracy=best_acc,
                  test_evaluated=False, elapsed_seconds=time.monotonic()-started,
                  cache_key=cache_key, history=history,
                  split_counts={k:int(v.sum()) for k,v in masks.items()},
                  actual_coefficients=dict(xi=classifier.model.gcn.beta,zeta=classifier.model.gcn.anfa),
                  raw_data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.glob('ind.cora.*'))},
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ['train_validation_grid.py','ablation_protocol.py','paper_protocol.py',
                                              'vendor/cgt/models.py','vendor/cgt/util.py']})
    (args.output / 'result.json').write_text(json.dumps(result,indent=2)+'\n')
    progress('completed',args.finetune_epochs,args.finetune_epochs,
             best_validation_accuracy=best_acc,test_evaluated=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
