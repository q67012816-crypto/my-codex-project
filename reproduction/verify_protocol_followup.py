"""Verify completed paired-seed and validation-grid artifacts independently."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('DGLBACKEND','pytorch')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/cgt'))
import numpy as np
from scipy import stats
import torch
from ablation_protocol import assert_flags


def independent_metrics(saved):
    masks=saved['masks']
    assert [int(masks[k].sum()) for k in ['train','validation','test']]==[1626,541,541]
    assert not any((masks[a]&masks[b]).any() for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert (masks['train']|masks['validation']|masks['test']).all()
    pred,truth,degree=[saved[k].numpy() for k in ['predictions','labels','degree']]
    test=masks['test'].numpy();ids=np.flatnonzero(test)
    assert np.array_equal(np.unique(truth[test]),np.arange(7))
    order=ids[np.lexsort((ids,degree[ids]))];low,high=order[:108],order[-108:]
    a=np.bincount(pred[low],minlength=7)/108
    b=np.bincount(pred[high],minlength=7)/108
    return 100*float((pred[test]==truth[test]).mean()),100*float(np.abs(a-b).mean())


def verify_history(r):
    for phase,n in [('pretrain',500),('finetune',200)]:
        rows=[x for x in r['history'] if x['phase']==phase]
        assert [x['epoch'] for x in rows]==list(range(1,n+1))
        assert all(math.isfinite(x['loss']) for x in rows)
    rows=[x for x in r['history'] if x['phase']=='finetune']
    best=max(rows,key=lambda x:x['validation_accuracy'])
    assert best['epoch']==r['best_validation_epoch'] and best['validation_accuracy']==r['best_validation_accuracy']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-only',action='store_true',help='Verify completed paired artifacts while the independent grid is running.')
    args=parser.parse_args()
    torch.set_num_threads(1)
    paired=ROOT/'runs/cora-paired-30'
    summary=json.loads((paired/'summary.json').read_text());protocol=json.loads((paired/'protocol.json').read_text())
    assert summary['status'] in ['completed','budget_exhausted'] and not summary['failures']
    if summary['status']=='completed':
        assert [row['seed'] for row in summary['pairs']]==list(range(42,72))
    for name,digest in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    legacy_sources=json.loads((ROOT/'runs/cora-baseline-10/protocol.json').read_text())['source_sha256']
    for name,digest in legacy_sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    columns=[[],[]];splits=set()
    for row in summary['pairs']:
        evaluations=[];values=[]
        for case in ['full','none']:
            record=row[case];directory=Path(record['output'])
            r=json.loads((directory/'result.json').read_text());verify_history(r)
            assert r['seed']==row['seed']
            if not (case=='full' and row['seed']<52):assert r['case']==case
            expected_sources=legacy_sources if case=='full' and row['seed']<52 else protocol['source_sha256']
            assert r['source_sha256']==expected_sources
            assert r['effective_config']==protocol['effective_config']
            d=torch.load(directory/'evaluation.pt',map_location='cpu')
            actual=independent_metrics(d)
            assert np.allclose(actual,[record['accuracy_percent'],record['delta_sp_percent']],rtol=0,atol=1e-10)
            assert all(torch.isfinite(v).all() for v in d['model'].values())
            model=torch.load(directory/'pretrained.pt',map_location='cpu')
            assert_flags(model,case)
            assert model.gcn.beta==.8 and model.gcn.anfa==.2
            evaluations.append(d);values.append(actual)
        for key in ['train','validation','test']:assert torch.equal(evaluations[0]['masks'][key],evaluations[1]['masks'][key])
        for key in ['labels','degree']:assert torch.equal(evaluations[0][key],evaluations[1][key])
        signature=hashlib.sha256(evaluations[0]['masks']['test'].numpy().tobytes()+evaluations[0]['masks']['validation'].numpy().tobytes()).hexdigest()
        assert signature not in splits;splits.add(signature)
        columns[0].append(values[0]);columns[1].append(values[1])
    for i,name in enumerate(['accuracy_points','delta_sp_x100_points']):
        full=np.array(columns[0])[:,i];none=np.array(columns[1])[:,i]
        ci=stats.ttest_rel(full,none).confidence_interval(confidence_level=.95)
        r=summary['paired'][name]
        assert math.isclose(float((full-none).mean()),r['mean_difference'],abs_tol=1e-10)
        assert np.allclose([ci.low,ci.high],r['confidence_interval'],rtol=0,atol=1e-10)
    result=dict(passed=True,paired=dict(verified_pairs=len(splits),target_pairs=summary['target_pairs'],
                target_reached=len(splits)==summary['target_pairs'],actual_layer_switches_verified=True,
                finite_weights_and_full_epochs_verified=True,paired_intervals_independently_verified=True))
    if args.paired_only:
        out=ROOT/'runs/ordered-followup-20261101/paired-verification.json'
        out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        return
    grid=ROOT/'runs/cora-validation-grid-seed-100'
    g=json.loads((grid/'summary.json').read_text());p=json.loads((grid/'protocol.json').read_text())
    assert g['status'] in ['completed','budget_exhausted'] and not g['failures']
    for name,digest in p['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'run_validation_grid.py').read_bytes()).hexdigest()==p['driver_sha256']
    candidates=[];reference_masks=None
    for item in g['candidates']:
        path=Path(item['output']);r=json.loads((path/'result.json').read_text());verify_history(r)
        assert r['seed']==item['seed']==p['seed']==100
        assert r['xi']==item['xi'] and r['zeta']==item['zeta']
        for key in ['best_validation_accuracy','best_validation_epoch','cache_key']:
            assert r[key]==item[key]
        assert not r['test_evaluated'] and 'test_accuracy' not in r
        assert r['source_sha256']==p['source_sha256']
        assert r['effective_config']=={**p['reference_config'],'xi':item['xi'],'zeta':item['zeta']}
        snap=torch.load(path/'validation-selected.pt',map_location='cpu')
        assert snap['model'].model.gcn.beta==item['xi'] and snap['model'].model.gcn.anfa==item['zeta']
        assert_flags(snap['model'].model,'full')
        assert all(torch.isfinite(v).all() for v in snap['model'].state_dict().values())
        if reference_masks is None:reference_masks=snap['masks']
        else:
            for key in reference_masks:assert torch.equal(reference_masks[key],snap['masks'][key])
        candidates.append(item)
    if g['status']=='completed':
        assert len(candidates)==16 and g['test_evaluated']
        choice=json.loads((grid/'selection.json').read_text())
        best=sorted(candidates,key=lambda r:(-r['best_validation_accuracy'],r['xi'],r['zeta']))[0]
        assert choice['xi']==best['xi'] and choice['zeta']==best['zeta']
        assert choice['protocol_sha256']==hashlib.sha256((grid/'protocol.json').read_bytes()).hexdigest()
        evaluation=json.loads((grid/'test-after-selection.json').read_text())
        assert evaluation['selection_file_sha256']==hashlib.sha256((grid/'selection.json').read_bytes()).hexdigest()
        assert choice['selection_utc']<=evaluation['evaluation_utc']
        for label,entry in evaluation['results'].items():
            actual_label='selected' if label=='reference' and (choice['xi'],choice['zeta'])==(.8,.2) else label
            d=torch.load(grid/f'{actual_label}-evaluation.pt',map_location='cpu')
            a,s=independent_metrics(d)
            assert np.allclose([a,s],[entry['accuracy_percent'],entry['delta_sp_percent']],rtol=0,atol=1e-10)
        result['grid']=dict(verified_candidates=16,validation_only_selection_verified=True,
                            test_after_locked_selection_verified=True,scope='one preregistered split only')
    else:
        assert not (grid/'selection.json').exists() and not (grid/'test-after-selection.json').exists()
        result['grid']=dict(verified_candidates=len(candidates),scope='partial: time budget exhausted; no selection or test result')
    lr=ROOT/'runs/finetune-lr-probe-seed42/comparison.json'
    if lr.exists():
        comparison=json.loads(lr.read_text());assert comparison['passed'] and comparison['same_initialization_and_RNG_verified']
        assert len({a['initial_state_sha256'] for a in comparison['arms']})==1
        assert len({a['rng_before_test_sha256'] for a in comparison['arms']})==1
        for item in comparison['arms']:
            path=lr.parent/f"lr-{item['lr']:g}-evaluation.pt"
            d=torch.load(path,map_location='cpu');actual=independent_metrics(d)
            assert np.allclose(actual,[item['accuracy_percent'],item['delta_sp_percent']],rtol=0,atol=1e-10)
        result['fine_tuning_lr_probe_verified']=True
    out=ROOT/'runs/ordered-followup-20261101/verification.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
