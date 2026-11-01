"""Independent verification of all 50 group/seed observations and sampler control."""
import hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('DGLBACKEND','pytorch')
sys.path.insert(0,str(ROOT/'vendor/cgt'))
import ablation_protocol  # required by trusted locally generated checkpoints


def main():
    base=ROOT/'runs/cora-ablation-10';reference=ROOT/'runs/cora-baseline-10'
    protocol=json.loads((base/'protocol.json').read_text())
    summary=json.loads((base/'summary.json').read_text())
    assert summary['status']=='completed' and summary['completed_new_runs']==40 and not summary['failures']
    assert summary['sampler_control']['passed']
    for manifest in [protocol['source_sha256'],protocol['original_baseline_sources']]:
        for n,h in manifest.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
    reference_splits={};reference_data={};verified={};metrics={}
    for group in protocol['groups']:
        rows=[];splits=set()
        for seed in protocol['seeds']:
            out=(reference if group=='full-p3' else base/group)/f'seed-{seed}'
            r=json.loads((out/'result.json').read_text())
            d=torch.load(out/'evaluation.pt',map_location='cpu')
            masks={k:v.numpy() for k,v in d['masks'].items()}
            train,val,test=[masks[k] for k in ['train','validation','test']]
            assert [int(x.sum()) for x in [train,val,test]]==[1626,541,541]
            assert not (train&val).any() and not (train&test).any() and not (val&test).any()
            assert (train|val|test).all()
            split=hashlib.sha256(val.tobytes()+test.tobytes()).hexdigest()
            if seed not in reference_splits:
                ref=torch.load(reference/f'seed-{seed}/evaluation.pt',map_location='cpu')
                reference_splits[seed]=hashlib.sha256(ref['masks']['validation'].numpy().tobytes()+ref['masks']['test'].numpy().tobytes()).hexdigest()
                reference_data[seed]=dict(labels=ref['labels'],degree=ref['degree'],
                    metadata=json.loads((reference/f'seed-{seed}/result.json').read_text()))
            assert split==reference_splits[seed] and split not in splits
            for name in ['labels','degree']:assert torch.equal(d[name],reference_data[seed][name])
            for name in ['raw_data_sha256','effective_config','cache_key']:
                assert r[name]==reference_data[seed]['metadata'][name]
            splits.add(split)
            for phase,n in [('pretrain',500),('finetune',200)]:
                records=[x for x in r['history'] if x['phase']==phase]
                assert [x['epoch'] for x in records]==list(range(1,n+1))
                assert all(math.isfinite(x['loss']) for x in records)
            fine=[x for x in r['history'] if x['phase']=='finetune']
            assert max(fine,key=lambda x:x['validation_accuracy'])['epoch']==r['best_validation_epoch']
            pred,truth,degree=[d[k].numpy() for k in ['predictions','labels','degree']]
            acc=float((pred[test]==truth[test]).mean())
            assert math.isclose(acc,r['test_accuracy'],abs_tol=1e-7)
            ids=np.flatnonzero(test);ordered=ids[np.lexsort((ids,degree[ids]))]
            low,high=ordered[:108],ordered[-108:];classes=np.unique(truth[test])
            delta=sum(abs((pred[low]==c).mean()-(pred[high]==c).mean()) for c in classes)/len(classes)
            assert math.isclose(delta,r['delta_sp'],abs_tol=1e-12)
            saved=json.loads((out/'fairness.json').read_text())
            assert saved['low_node_ids']==low.tolist() and saved['high_node_ids']==high.tolist()
            model=torch.load(out/'pretrained.pt',map_location='cpu')
            case=group.rsplit('-p',1)[0];flags=ablation_protocol.CASES[case]
            ablation_protocol.assert_flags(model,case)
            assert model.gcn.anfa==.2 and model.gcn.beta==.8
            assert all(torch.isfinite(v).all() for v in d['model'].values())
            if group!='full-p3':
                assert r['source_sha256']==protocol['source_sha256']
                assert r['reconstruction_steps']==int(group[-1]) and r['structural_transition_steps']==3
            if not flags[0]:assert all(x['augmentation']==0 for x in r['history'] if x['phase']=='pretrain')
            rows.append([seed,100*r['test_accuracy'],100*delta])
        values=np.array(rows)[:,1:];entry=summary['groups'][group]
        for i,name in enumerate(['accuracy_percent','delta_sp_percent']):
            assert math.isclose(float(values[:,i].mean()),entry[name]['mean'],abs_tol=1e-9)
            assert math.isclose(float(values[:,i].std(ddof=1)),entry[name]['std_sample'],abs_tol=1e-9)
        verified[group]=dict(runs=10,distinct_splits=len(splits),matches_reference_splits=True,actual_layer_switches_verified=True)
        metrics[group]=rows
    for entry in summary['comparisons'].values():
        after={x[0]:x for x in metrics[entry['after']]};before={x[0]:x for x in metrics[entry['before']]}
        for col,key in [(1,'test_accuracy'),(2,'delta_sp_percent')]:
            diffs=[after[s][col]-before[s][col] for s in protocol['seeds']]
            assert math.isclose(float(np.mean(diffs)),entry[key]['mean_difference'],abs_tol=1e-9)
            assert math.isclose(float(np.std(diffs,ddof=1)),entry[key]['std_sample'],abs_tol=1e-9)
            assert np.allclose(diffs,entry[key]['per_seed'],rtol=0,atol=1e-9)
    result=dict(passed=True,scope='full',group_observations=50,new_formal_runs=40,reused_baseline_runs=10,
                groups=verified,paired_comparisons_verified=True,sampler_control=summary['sampler_control'])
    (base/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
