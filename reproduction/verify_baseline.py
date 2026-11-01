"""Verify completed baseline artifacts independently of the training metric code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path(__file__).parent/'runs/cora-baseline-10')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    base = args.directory
    protocol = json.loads((base/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest, name
    summary = json.loads((base/'summary.json').read_text())
    verified, accuracies, disparities, split_hashes = [], [], [], set()
    for seed in protocol['seeds']:
        out = base/f'seed-{seed}'
        if not (out/'result.json').exists():
            assert args.allow_incomplete, f'missing seed {seed}'
            continue
        r = json.loads((out/'result.json').read_text())
        data = torch.load(out/'evaluation.pt', map_location='cpu')
        groups = json.loads((out/'fairness.json').read_text())
        assert r['source_sha256'] == protocol['source_sha256']
        assert r['seed']==seed and r['pretrain_epochs']==500 and r['finetune_epochs']==200
        for phase, expected in [('pretrain',500),('finetune',200)]:
            rows = [x for x in r['history'] if x['phase']==phase]
            assert [x['epoch'] for x in rows] == list(range(1,expected+1))
            assert all(math.isfinite(x['loss']) for x in rows)
        masks = data['masks']
        train, val, test = [masks[k].numpy() for k in ['train','validation','test']]
        assert [int(x.sum()) for x in [train,val,test]] == [1626,541,541]
        assert not (train & val).any() and not (train & test).any() and not (val & test).any()
        assert (train | val | test).all()
        fingerprint = hashlib.sha256(test.tobytes()+val.tobytes()).hexdigest()
        assert fingerprint not in split_hashes, 'duplicate split'
        split_hashes.add(fingerprint)
        pred, labels, degree = [data[k].numpy() for k in ['predictions','labels','degree']]
        acc = float((pred[test]==labels[test]).mean())
        assert math.isclose(acc, r['test_accuracy'], abs_tol=1e-7)
        ids = np.flatnonzero(test)
        order = ids[np.lexsort((ids, degree[ids]))]
        low, high = order[:108], order[-108:]
        assert groups['low_node_ids']==low.tolist() and groups['high_node_ids']==high.tolist()
        classes = np.unique(labels[test])
        # Independent direct counting, not the training fairness function.
        delta = sum(abs(np.count_nonzero(pred[low]==c)/108 -
                        np.count_nonzero(pred[high]==c)/108) for c in classes)/len(classes)
        assert math.isclose(delta, r['delta_sp'], abs_tol=1e-12)
        assert math.isclose(100*delta, r['delta_sp_percent'], abs_tol=1e-10)
        validation = [x for x in r['history'] if x['phase']=='finetune']
        best = max(validation,key=lambda x:x['validation_accuracy'])
        assert r['best_validation_epoch']==best['epoch']
        assert all(torch.isfinite(v).all() for v in data['model'].values())
        accuracies.append(100*r['test_accuracy']); disparities.append(100*delta)
        verified.append(seed)
    assert len(verified)==summary['completed_runs']
    if not args.allow_incomplete:
        assert len(verified)==10 and summary['status']=='completed' and not summary['failures']
    for metric, values in [('accuracy_percent',accuracies),('delta_sp_percent',disparities)]:
        if values:
            assert math.isclose(float(np.mean(values)), summary[metric]['mean'],abs_tol=1e-10)
            if len(values)>1:
                assert math.isclose(float(np.std(values,ddof=1)),summary[metric]['std_sample'],abs_tol=1e-10)
    result = dict(passed=True, scope='full' if len(verified)==10 else 'partial',
                  verified_seeds=verified, independently_recomputed_metrics=True,
                  distinct_splits=len(split_hashes), valid_histories_and_checkpoints=True)
    (base/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
