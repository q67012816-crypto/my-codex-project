"""Verify the bounded, seed-42 reconstruction-normalization diagnostic."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('DGLBACKEND', 'pytorch')
sys.path.insert(0, str(ROOT / 'vendor/cgt'))
from ablation_protocol import assert_flags


def main():
    base = ROOT / 'runs/loss-convention-control'
    out = base / 'seed-42'
    reference = ROOT / 'runs/cora-baseline-10/seed-42'
    protocol = json.loads((base / 'protocol.json').read_text())
    result = json.loads((out / 'result.json').read_text())
    old = json.loads((reference / 'result.json').read_text())
    assert result['source_sha256'] == protocol['source_sha256']
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert result['seed'] == 42 and result['case'] == 'full'
    assert result['sampler'] == 'fast'
    assert result['reconstruction_steps'] == result['structural_transition_steps'] == 3
    for key in ['effective_config', 'raw_data_sha256', 'cache_key', 'split_counts']:
        assert result[key] == old[key], key
    for phase, count in [('pretrain', 500), ('finetune', 200)]:
        rows = [r for r in result['history'] if r['phase'] == phase]
        assert [r['epoch'] for r in rows] == list(range(1, count + 1))
        assert all(math.isfinite(r['loss']) for r in rows)
    pretrain = [r for r in result['history'] if r['phase'] == 'pretrain']
    assert all(not r['augmentation_has_gradient'] for r in pretrain)
    fine = [r for r in result['history'] if r['phase'] == 'finetune']
    best = max(fine, key=lambda r: r['validation_accuracy'])
    assert best['epoch'] == result['best_validation_epoch']
    assert best['validation_accuracy'] == result['best_validation_accuracy']

    saved = torch.load(out / 'evaluation.pt', map_location='cpu')
    prior = torch.load(reference / 'evaluation.pt', map_location='cpu')
    for name in ['train', 'validation', 'test']:
        assert torch.equal(saved['masks'][name], prior['masks'][name])
    for name in ['labels', 'degree']:
        assert torch.equal(saved[name], prior[name])
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    model = torch.load(out / 'pretrained.pt', map_location='cpu')
    assert_flags(model, 'full')
    assert model.gcn.anfa == .2 and model.gcn.beta == .8
    pred, truth, degree = [saved[k].numpy() for k in ['predictions', 'labels', 'degree']]
    test = saved['masks']['test'].numpy()
    accuracy = float((pred[test] == truth[test]).mean())
    assert math.isclose(accuracy, result['test_accuracy'], abs_tol=1e-7)
    ids = np.flatnonzero(test)
    ranked = ids[np.lexsort((ids, degree[ids]))]
    low, high = ranked[:108], ranked[-108:]
    classes = np.unique(truth[test])
    delta = float(np.mean([abs((pred[low] == c).mean() - (pred[high] == c).mean())
                           for c in classes]))
    assert math.isclose(delta, result['delta_sp'], abs_tol=1e-12)
    fairness = json.loads((out / 'fairness.json').read_text())
    assert fairness['low_node_ids'] == low.tolist()
    assert fairness['high_node_ids'] == high.tolist()
    report = dict(
        passed=True, seed=42, pretrain_epochs=500, finetune_epochs=200,
        same_masks_config_cache_data=True, actual_flags_verified=True,
        reference_accuracy_percent=100 * old['test_accuracy'],
        control_accuracy_percent=100 * result['test_accuracy'],
        accuracy_difference_points=100 * (result['test_accuracy'] - old['test_accuracy']),
        reference_delta_sp_percent=old['delta_sp_percent'],
        control_delta_sp_percent=100 * delta,
        delta_sp_difference_points=100 * delta - old['delta_sp_percent'],
        interpretation='One seed only; not a ten-run reproduction or an exact upstream training replay.',
        changes='Transition mean and feature MSE replace the raw sum and unsquared flattened L2/N; fixed augmentation BCE unchanged.')
    (base / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
