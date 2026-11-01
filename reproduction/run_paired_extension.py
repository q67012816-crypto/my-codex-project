"""Extend the frozen full/none comparison; only the number of seeds changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
from scipy.stats import t
import torch

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def load_record(directory, case, seed, sources, config, legacy=False):
    result = json.loads((directory/'result.json').read_text())
    assert result['seed'] == seed and result['effective_config'] == config
    for phase, count in [('pretrain', 500), ('finetune', 200)]:
        rows = [row for row in result['history'] if row['phase'] == phase]
        assert [row['epoch'] for row in rows] == list(range(1, count+1))
        assert all(math.isfinite(row['loss']) for row in rows)
    if not legacy:
        assert result['source_sha256'] == sources
        assert result['case'] == case and result['reconstruction_steps'] == 3
    saved = torch.load(directory/'evaluation.pt', map_location='cpu')
    test = saved['masks']['test'].numpy()
    pred, truth, degree = [saved[name].numpy() for name in ['predictions','labels','degree']]
    ids = np.flatnonzero(test)
    assert len(ids) == 541
    ranked = ids[np.lexsort((ids, degree[ids]))]
    low, high = ranked[:108], ranked[-108:]
    classes = np.unique(truth[test])
    sp = 100*float(np.mean([abs((pred[low] == c).mean()-(pred[high] == c).mean()) for c in classes]))
    accuracy = 100*float((pred[test] == truth[test]).mean())
    assert math.isclose(accuracy, 100*result['test_accuracy'], abs_tol=1e-5)
    assert math.isclose(sp, result['delta_sp_percent'], abs_tol=1e-10)
    signature = hashlib.sha256(b''.join(saved['masks'][name].numpy().tobytes() for name in ['train','validation','test'])+
                               truth.tobytes()+degree.tobytes()).hexdigest()
    return dict(seed=seed, case=case, accuracy_percent=accuracy, delta_sp_percent=sp,
                split_labels_degree_sha256=signature, output=str(directory),
                best_validation_epoch=result['best_validation_epoch'])


def paired_summary(rows):
    output = {}
    for name, column in [('accuracy_points','accuracy_percent'),('delta_sp_x100_points','delta_sp_percent')]:
        differences = [row['full'][column]-row['none'][column] for row in rows]
        mean, sd = statistics.mean(differences), statistics.stdev(differences)
        half = float(t.ppf(.975, len(rows)-1))*sd/math.sqrt(len(rows))
        output[name] = dict(mean_difference=mean, std_sample=sd,
                            confidence_interval=[mean-half, mean+half], n=len(rows))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-seeds', type=int, choices=[30,50], default=30)
    parser.add_argument('--max-minutes', type=float, default=100)
    args = parser.parse_args()
    out = ROOT/f'runs/cora-paired-{args.total_seeds}'
    out.mkdir(exist_ok=True)
    old = ROOT/'runs/cora-ablation-10'
    prior = json.loads((old/'protocol.json').read_text())
    assert json.loads((old/'verification.json').read_text())['passed']
    assert json.loads((old/'sampler-equivalence.json').read_text())['passed']
    sources = prior['source_sha256']
    for manifest in [sources, prior['original_baseline_sources']]:
        for name, digest in manifest.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    config = json.loads((ROOT/'runs/cora-baseline-10/seed-42/result.json').read_text())['effective_config']
    protocol = dict(seeds=list(range(42,42+args.total_seeds)), cases=['full','none'],
                    source_sha256=sources, effective_config=config,
                    pretrain_epochs=500, finetune_epochs=200, structural_k=3, reconstruction_p=3,
                    change='Seed count only; frozen CPU protocol, not the unmodified official CUDA command.',
                    reuse='Seeds42-51 use verified full baseline and none ablations.',
                    max_minutes=args.max_minutes, workers=2,
                    interval='Two-sided paired Student t, sample SD; no multiplicity adjustment.')
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == protocol
    write_json(out/'protocol.json', protocol)
    completed = {}
    for seed in range(42,52):
        completed['full',seed] = load_record(ROOT/f'runs/cora-baseline-10/seed-{seed}', 'full', seed, sources, config, True)
        completed['none',seed] = load_record(old/f'none-p3/seed-{seed}', 'none', seed, sources, config)
    started = time.monotonic()
    deadline = started+args.max_minutes*60
    failures, deferred = {}, []

    def summarize(final=False):
        pairs=[]
        for seed in protocol['seeds']:
            if all((case,seed) in completed for case in protocol['cases']):
                full,none=completed['full',seed],completed['none',seed]
                assert full['split_labels_degree_sha256'] == none['split_labels_degree_sha256']
                pairs.append(dict(seed=seed,full=full,none=none))
        status = 'completed' if len(pairs)==args.total_seeds and not failures else ('budget_exhausted' if final and not failures else ('failed' if final else 'running'))
        report=dict(status=status,completed_pairs=len(pairs),target_pairs=args.total_seeds,
                    completed_new_jobs=len(completed)-20,required_new_jobs=2*(args.total_seeds-10),
                    elapsed_seconds=time.monotonic()-started,failures=failures,deferred=deferred,
                    paired=paired_summary(pairs),pairs=pairs)
        write_json(out/'summary.json',report)
        with (out/'paired-differences.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['seed','full_accuracy','none_accuracy','accuracy_difference_points','full_delta_sp','none_delta_sp','delta_sp_difference'])
            for row in pairs:
                full,none=row['full'],row['none']
                writer.writerow([row['seed'],full['accuracy_percent'],none['accuracy_percent'],full['accuracy_percent']-none['accuracy_percent'],
                                 full['delta_sp_percent'],none['delta_sp_percent'],full['delta_sp_percent']-none['delta_sp_percent']])

    def job(case,seed):
        directory=out/f'{case}-p3/seed-{seed}'
        if (directory/'result.json').exists():
            return load_record(directory,case,seed,sources,config)
        if time.monotonic()+600 > deadline:
            return None
        directory.mkdir(parents=True,exist_ok=True)
        print(json.dumps(dict(event='started',case=case,seed=seed)),flush=True)
        command=[sys.executable,'-u',str(ROOT/'train_ablation.py'),'--case',case,'--p','3','--seed',str(seed),
                 '--pretrain-epochs','500','--finetune-epochs','200','--output',str(directory)]
        environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',DGLBACKEND='pytorch')
        with (directory/'training.log').open('w') as stream:
            subprocess.run(command,cwd=ROOT.parent,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=900,check=True)
        return load_record(directory,case,seed,sources,config)

    summarize()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(job,case,seed):(case,seed) for seed in range(52,42+args.total_seeds) for case in protocol['cases']}
        for future in as_completed(futures):
            key=futures[future]
            try:
                row=future.result()
                if row is None:deferred.append(list(key))
                else:completed[key]=row
                print(json.dumps(dict(event='deferred' if row is None else 'completed',case=key[0],seed=key[1])),flush=True)
            except Exception as error:
                failures[str(key)]=str(error)
                print(json.dumps(dict(event='failed',job=key,error=str(error))),flush=True)
            summarize()
    summarize(final=True)
    print((out/'summary.json').read_text(),flush=True)


if __name__ == '__main__':
    main()
