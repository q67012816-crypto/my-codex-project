"""Four Table-8 cases plus an isolated SSL p=4 experiment, seeds 42–51."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv,hashlib,json,os,statistics,subprocess,sys,threading,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FLAGS={'none':(0,0,0),'attention':(0,1,1),'augmentation':(1,0,0),'full':(1,1,1)}
SOURCES=['train_ablation.py','ablation_protocol.py','paper_protocol.py','vendor/cgt/models.py','vendor/cgt/util.py']
PAPER={'none-p3':(84.81,5.88),'attention-p3':(85.54,4.85),
       'augmentation-p3':(84.95,3.53),'full-p3':(87.10,3.46)}


def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runs/cora-ablation-10')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}
    base=ROOT/'runs/cora-baseline-10'
    original_protocol=json.loads((base/'protocol.json').read_text())
    for n,h in original_protocol['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
    assert json.loads((base/'verification.json').read_text())['scope']=='full'
    protocol=dict(seeds=list(range(42,52)),pretrain_epochs=500,finetune_epochs=200,
                  source_sha256=sources,original_baseline_sources=original_protocol['source_sha256'],
                  groups=['none-p3','attention-p3','augmentation-p3','full-p3','full-p4'],
                  required_new_runs=40,structural_transition_steps=3,
                  sensitivity='p=4 changes only reconstruction targets; B/sim/phi and structural search stay at k=3',
                  flags=FLAGS,reference_reuse='full-p3 uses existing independently verified ten-seed baseline',
                  sampler='fixed Bernoulli threshold; gated by complete seed42 original-RNG equivalence check',
                  std='sample ddof=1; population std also saved',paper_targets=PAPER)
    manifest=args.output/'protocol.json'
    # JSON roundtrip turns tuples into lists, so compare canonical JSON values.
    protocol=json.loads(json.dumps(protocol))
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:parser.error('protocol differs: use a new output directory')
    write_json(manifest,protocol)
    completed={};failures={};control={};gate=threading.Event();started=time.monotonic()
    baseline={s:json.loads((base/f'seed-{s}/result.json').read_text()) for s in protocol['seeds']}

    def validate(case,p,seed,r):
        assert r['case']==case and r['reconstruction_steps']==p and r['seed']==seed
        assert r['source_sha256']==sources and r['structural_transition_steps']==3
        assert r['switches']==dict(zip(['aug','sim','phi'],FLAGS[case]))
        assert r['effective_config']==baseline[seed]['effective_config']
        for phase,n in [('pretrain',500),('finetune',200)]:
            assert [x['epoch'] for x in r['history'] if x['phase']==phase]==list(range(1,n+1))
        assert 0<=r['test_accuracy']<=1 and 0<=r['delta_sp']<=1
        return r

    def execute(case,p,seed,out):
        out.mkdir(parents=True,exist_ok=True);result=out/'result.json'
        if result.exists():return validate(case,p,seed,json.loads(result.read_text()))
        # Preserve incomplete earlier diagnostics before restarting a seed.
        present=[out/n for n in ['training.log','progress.json'] if (out/n).exists()]
        if present:
            archive=out/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
            for f in present:f.rename(archive/f.name)
        env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','DGLBACKEND':'pytorch'}
        cmd=[sys.executable,'-u',str(ROOT/'train_ablation.py'),'--case',case,'--p',str(p),'--seed',str(seed),
             '--pretrain-epochs','500','--finetune-epochs','200','--output',str(out)]
        print(json.dumps(dict(event='started',case=case,p=p,seed=seed)),flush=True)
        with (out/'training.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return validate(case,p,seed,json.loads(result.read_text()))

    def control_job():
        try:
            out=args.output/'sampler-control/seed-42'
            r=execute('full',3,42,out);ref=baseline[42]
            for key in ['history','test_accuracy','delta_sp','best_validation_epoch']:
                assert r[key]==ref[key],f'sampler equivalence failed: {key}'
            import torch
            x=torch.load(out/'evaluation.pt',map_location='cpu')
            y=torch.load(base/'seed-42/evaluation.pt',map_location='cpu')
            assert torch.equal(x['predictions'],y['predictions'])
            assert all(torch.equal(v,y['model'][k]) for k,v in x['model'].items())
            control.update(passed=True,seed=42,full_epochs=True,identical_history_predictions_weights=True)
            write_json(args.output/'sampler-equivalence.json',control)
            return r
        except Exception as exc:
            control.update(passed=False,error=str(exc));raise
        finally:gate.set()

    def run_job(case,p,seed):
        if FLAGS[case][0]:
            gate.wait()
            if not control.get('passed'):raise RuntimeError('augmentation experiments blocked by sampler equivalence failure')
        return execute(case,p,seed,args.output/f'{case}-p{p}'/f'seed-{seed}')

    def summarize():
        groups={'full-p3':baseline.copy()}
        for (case,p,seed),r in completed.items():groups.setdefault(f'{case}-p{p}',{})[seed]=r
        report=dict(status='completed' if len(completed)==40 and not failures and control.get('passed') else 'incomplete',
                    required_new_runs=40,completed_new_runs=len(completed),failures=failures,
                    sampler_control=control,elapsed_seconds=time.monotonic()-started,groups={},comparisons={})
        for name in protocol['groups']:
            rows=groups.get(name,{});entry=dict(completed_runs=len(rows),seeds=sorted(rows))
            for metric,values in [('accuracy_percent',[100*r['test_accuracy'] for r in rows.values()]),
                                  ('delta_sp_percent',[r['delta_sp_percent'] for r in rows.values()])]:
                if values:entry[metric]=dict(mean=statistics.mean(values),std_sample=statistics.stdev(values) if len(values)>1 else None,
                                             std_population=statistics.pstdev(values))
            if name in PAPER:entry['paper_accuracy_percent'],entry['paper_delta_sp_percent']=PAPER[name]
            report['groups'][name]=entry
        comparisons=[('augmentation_with_attention','full-p3','attention-p3'),
                     ('augmentation_without_attention','augmentation-p3','none-p3'),
                     ('attention_with_augmentation','full-p3','augmentation-p3'),
                     ('attention_without_augmentation','attention-p3','none-p3'),
                     ('p4_minus_p3','full-p4','full-p3')]
        for label,after,before in comparisons:
            if len(groups.get(after,{}))==len(groups.get(before,{}))==10:
                out=dict(after=after,before=before,paired_seeds=protocol['seeds'])
                for metric,factor in [('test_accuracy',100),('delta_sp_percent',1)]:
                    diffs=[factor*(groups[after][s][metric]-groups[before][s][metric]) for s in protocol['seeds']]
                    out[metric]=dict(mean_difference=statistics.mean(diffs),std_sample=statistics.stdev(diffs),per_seed=diffs)
                report['comparisons'][label]=out
        write_json(args.output/'summary.json',report)
        with (args.output/'runs.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['group','seed','accuracy_percent','delta_sp_percent','best_validation_epoch'])
            for name in protocol['groups']:
                for seed,r in sorted(groups.get(name,{}).items()):writer.writerow([name,seed,100*r['test_accuracy'],r['delta_sp_percent'],r['best_validation_epoch']])

    summarize()
    jobs=[(case,3,seed) for case in ['none','attention','augmentation'] for seed in protocol['seeds']]
    jobs += [('full',4,seed) for seed in protocol['seeds']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(control_job):None}
        futures.update({pool.submit(run_job,*job):job for job in jobs})
        for future in as_completed(futures):
            job=futures[future]
            try:
                r=future.result()
                if job is not None:completed[job]=r
                print(json.dumps(dict(event='completed',job=job,sampler_control=job is None,
                                      accuracy_percent=100*r['test_accuracy'],delta_sp_percent=r['delta_sp_percent'])),flush=True)
            except Exception as exc:
                failures[str(job)]=str(exc);print(json.dumps(dict(event='failed',job=job,error=str(exc))),flush=True)
            summarize()
    print((args.output/'summary.json').read_text(),flush=True)
    return 0 if len(completed)==40 and not failures and control.get('passed') else 1


if __name__=='__main__':sys.exit(main())
