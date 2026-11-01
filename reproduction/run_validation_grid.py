"""Sixteen coefficient candidates on one preregistered split, validation-only selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('DGLBACKEND','pytorch')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/cgt'))
import torch
import ablation_protocol

SOURCES=['train_validation_grid.py','ablation_protocol.py','paper_protocol.py','vendor/cgt/models.py','vendor/cgt/util.py']


def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def evaluate_after_selection(out,selection,completed):
    """This is the only function in the search that reads test labels for metrics."""
    assert (out/'selection.json').exists() and len(completed)==16
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_networkx
    from util import load_dgl
    from paper_protocol import degree_statistical_parity
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data=Planetoid(root=str(ROOT/'data'),name='Cora',split='public')[0]
    reference=completed[.8,.2]
    cache=ROOT/'cache'/f"cora-paper-{reference['cache_key']}.pt"
    structural=torch.load(cache,map_location='cpu')
    b,sim,phi=[structural[k] for k in ['B','sim','phi']]
    sim=torch.nan_to_num(sim,nan=0.)
    graph=load_dgl(to_networkx(data,to_undirected=True),data.x,sim,phi)
    adj=torch.zeros(data.num_nodes,data.num_nodes,dtype=torch.float64)
    adj[data.edge_index[0],data.edge_index[1]]=1
    device=torch.device('cpu')
    results={}
    for label,pair in [('selected',(selection['xi'],selection['zeta'])),('reference',(.8,.2))]:
        if label=='reference' and pair==(selection['xi'],selection['zeta']):
            results[label]=dict(results['selected']);continue
        record=completed[pair]
        snapshot=torch.load(Path(record['output'])/'validation-selected.pt',map_location='cpu')
        model=snapshot['model'];model.eval()
        torch.set_rng_state(snapshot['rng_state'])
        with torch.no_grad():
            pred=model(graph.local_var(),adj,sim,phi,b,3,record['best_validation_epoch'],device,device).argmax(1)
        mask=snapshot['masks']['test']
        correct=int((pred[mask]==data.y[mask]).sum())
        fairness=degree_statistical_parity(pred,data.y,snapshot['degree'],mask)
        torch.save(dict(predictions=pred,labels=data.y,degree=snapshot['degree'],masks=snapshot['masks']),out/f'{label}-evaluation.pt')
        results[label]=dict(xi=pair[0],zeta=pair[1],seed=record['seed'],test_nodes=int(mask.sum()),correct=correct,
                            accuracy_percent=100*correct/int(mask.sum()),delta_sp_percent=fairness['delta_sp_percent'],
                            validation_accuracy=record['best_validation_accuracy'],best_validation_epoch=record['best_validation_epoch'])
        write_json(out/f'{label}-fairness.json',fairness)
    result=dict(evaluation_utc=datetime.now(timezone.utc).isoformat(),
                selection_file_sha256=hashlib.sha256((out/'selection.json').read_bytes()).hexdigest(),
                selection_preceded_test=True,results=results,
                scope='One Cora split only; a bounded coefficient diagnostic, not ten-run reproduction or evidence of author selection bias.')
    write_json(out/'test-after-selection.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--max-minutes',type=float,default=75)
    parser.add_argument('--paired-stage',type=Path,default=ROOT/'runs/cora-paired-30/summary.json')
    args=parser.parse_args()
    prior=json.loads(args.paired_stage.read_text())
    assert prior['status'] in ['completed','budget_exhausted'], 'Finish the seed-count stage before starting the grid.'
    assert not prior['failures']
    out=ROOT/f'runs/cora-validation-grid-seed-{args.seed}'
    out.mkdir(exist_ok=True)
    if (out/'test-after-selection.json').exists():raise SystemExit('Completed search already exists; preserve results.')
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    reference_config=json.loads((ROOT/'runs/cora-baseline-10/seed-42/result.json').read_text())['effective_config']
    protocol=dict(seed=args.seed,values=[.2,.4,.6,.8],candidates=16,pretrain_epochs=500,finetune_epochs=200,
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=sources,reference_config=reference_config,
                  selection='Highest best validation accuracy; ties: ascending xi, then ascending zeta.',
                  test_policy='No test metrics for candidates; lock selection first; evaluate winner and preregistered (.8,.2) reference once.',
                  scope='Single-split bounded diagnostic on existing CPU paper-settings implementation; not the original CUDA command.',
                  max_minutes=args.max_minutes,workers=2)
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    write_json(out/'protocol.json',protocol)
    started=time.monotonic();deadline=started+args.max_minutes*60
    completed={};failures={};deferred=[]

    def validate(directory,xi,zeta):
        r=json.loads((directory/'result.json').read_text())
        assert r['source_sha256']==sources and r['seed']==args.seed
        assert r['xi']==xi and r['zeta']==zeta and not r['test_evaluated']
        assert 'test_accuracy' not in r and not (directory/'evaluation.pt').exists()
        assert r['effective_config']=={**reference_config,'xi':xi,'zeta':zeta}
        assert r['actual_coefficients']==dict(xi=xi,zeta=zeta)
        for phase,count in [('pretrain',500),('finetune',200)]:
            assert [x['epoch'] for x in r['history'] if x['phase']==phase]==list(range(1,count+1))
        fine=[x for x in r['history'] if x['phase']=='finetune']
        best=max(fine,key=lambda x:x['validation_accuracy'])
        assert best['epoch']==r['best_validation_epoch'] and best['validation_accuracy']==r['best_validation_accuracy']
        return dict(output=str(directory),seed=args.seed,xi=xi,zeta=zeta,
                    best_validation_accuracy=r['best_validation_accuracy'],best_validation_epoch=r['best_validation_epoch'],cache_key=r['cache_key'])

    def job(xi,zeta):
        directory=out/f'xi-{xi:.1f}-zeta-{zeta:.1f}'
        if (directory/'result.json').exists():return validate(directory,xi,zeta)
        if time.monotonic()+600>deadline:return None
        directory.mkdir(parents=True,exist_ok=True)
        print(json.dumps(dict(event='started',xi=xi,zeta=zeta,seed=args.seed)),flush=True)
        command=[sys.executable,'-u',str(ROOT/'train_validation_grid.py'),'--case','full','--seed',str(args.seed),
                 '--xi',str(xi),'--zeta',str(zeta),'--pretrain-epochs','500','--finetune-epochs','200','--output',str(directory)]
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',DGLBACKEND='pytorch')
        with (directory/'training.log').open('w') as stream:
            subprocess.run(command,cwd=ROOT.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,
                           timeout=min(1200,max(1,deadline-time.monotonic())),check=True)
        return validate(directory,xi,zeta)

    def summarize(final=False):
        status='completed' if len(completed)==16 and not failures else ('budget_exhausted' if final and not failures else ('failed' if final else 'running'))
        write_json(out/'summary.json',dict(status=status,completed_candidates=len(completed),required_candidates=16,
                   elapsed_seconds=time.monotonic()-started,failures=failures,deferred=deferred,
                   candidates=[completed[p] for p in sorted(completed)],test_evaluated=False))

    summarize()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[(xi,zeta) for xi in protocol['values'] for zeta in protocol['values']]
        futures={pool.submit(job,*pair):pair for pair in jobs}
        for future in as_completed(futures):
            pair=futures[future]
            try:
                record=future.result()
                if record is None:deferred.append(list(pair))
                else:completed[pair]=record
                print(json.dumps(dict(event='deferred' if record is None else 'completed',pair=pair,
                                      validation_accuracy=None if record is None else record['best_validation_accuracy'])),flush=True)
            except Exception as error:
                failures[str(pair)]=str(error);print(json.dumps(dict(event='failed',pair=pair,error=str(error))),flush=True)
            summarize()
    summarize(final=True)
    if len(completed)==16 and not failures:
        winner=sorted(completed.values(),key=lambda r:(-r['best_validation_accuracy'],r['xi'],r['zeta']))[0]
        selection=dict(**winner,selection_utc=datetime.now(timezone.utc).isoformat(),candidate_count=16,
                       protocol_sha256=hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),
                       criterion=protocol['selection'],test_evaluated_before_selection=False)
        write_json(out/'selection.json',selection)
        evaluate_after_selection(out,selection,completed)
        summary=json.loads((out/'summary.json').read_text());summary['test_evaluated']=True
        write_json(out/'summary.json',summary)
    print((out/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    main()
