"""Run the fixed ten-split Cora baseline; aggregate only validated full runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCES = ['train_cora.py', 'paper_protocol.py', 'vendor/cgt/models.py', 'vendor/cgt/util.py']


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/cora-baseline-10')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    protocol = dict(dataset='Cora', seeds=list(range(42,52)), pretrain_epochs=500,
                    finetune_epochs=200, source_sha256=sources,
                    purpose='CGT repository baseline with audited paper settings; known deviations retained',
                    finetune_epochs_source='repository working choice; not specified by paper',
                    split='random 60/20/20 with upstream integer rounding',
                    primary_std='sample standard deviation, ddof=1; paper does not specify ddof')
    manifest = args.output / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        parser.error('existing protocol differs; use a new experiment directory')
    atomic_json(manifest, protocol)
    completed, failures = {}, {}
    started = time.monotonic()

    def validate(seed, result):
        assert result['seed'] == seed
        assert result['pretrain_epochs'] == 500 and result['finetune_epochs'] == 200
        assert result['source_sha256'] == sources
        assert sum(x['phase']=='pretrain' for x in result['history']) == 500
        assert sum(x['phase']=='finetune' for x in result['history']) == 200
        assert 0 <= result['test_accuracy'] <= 1 and 0 <= result['delta_sp'] <= 1
        assert result['actual_attention_layers'] == 4
        return result

    def run_seed(seed):
        out = args.output / f'seed-{seed}'
        out.mkdir(exist_ok=True)
        result_path = out / 'result.json'
        if result_path.exists():
            return validate(seed, json.loads(result_path.read_text()))
        env = {**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2', 'DGLBACKEND':'pytorch'}
        command = [sys.executable, '-u', str(ROOT/'train_cora.py'), '--seed', str(seed),
                   '--pretrain-epochs', '500', '--finetune-epochs', '200', '--output', str(out)]
        print(json.dumps(dict(event='started', seed=seed)), flush=True)
        with (out/'training.log').open('w') as log:
            subprocess.run(command, cwd=ROOT.parent, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        return validate(seed, json.loads(result_path.read_text()))

    def summarize():
        runs = [completed[k] for k in sorted(completed)]
        report = dict(status='completed' if len(runs)==10 and not failures else 'incomplete',
                      completed_seeds=sorted(completed), failures=failures,
                      required_runs=10, completed_runs=len(runs), protocol=protocol,
                      elapsed_seconds=time.monotonic()-started)
        if runs:
            for key, values, target in [('accuracy_percent',[100*r['test_accuracy'] for r in runs],87.10),
                                        ('delta_sp_percent',[r['delta_sp_percent'] for r in runs],3.46)]:
                report[key] = dict(mean=statistics.mean(values),
                    std_sample=statistics.stdev(values) if len(values)>1 else None,
                    std_population=statistics.pstdev(values), paper_mean=target,
                    mean_minus_paper=statistics.mean(values)-target)
            report['remaining_deviations'] = runs[0]['remaining_deviations']
        atomic_json(args.output/'summary.json', report)
        with (args.output/'runs.csv').open('w',newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['seed','pretrain_epochs','finetune_epochs','accuracy_percent',
                             'delta_sp_percent','best_validation_epoch','elapsed_seconds'])
            for r in runs:
                writer.writerow([r['seed'],r['pretrain_epochs'],r['finetune_epochs'],100*r['test_accuracy'],
                                 r['delta_sp_percent'],r['best_validation_epoch'],r['elapsed_seconds']])

    summarize()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_seed, seed):seed for seed in protocol['seeds']}
        for future in as_completed(jobs):
            seed = jobs[future]
            try:
                completed[seed] = future.result()
                print(json.dumps(dict(event='completed',seed=seed,
                      accuracy_percent=100*completed[seed]['test_accuracy'],
                      delta_sp_percent=completed[seed]['delta_sp_percent'])), flush=True)
            except Exception as exc:
                failures[str(seed)] = str(exc)
                print(json.dumps(dict(event='failed',seed=seed,error=str(exc))), flush=True)
            summarize()
    print((args.output/'summary.json').read_text(), flush=True)
    return 0 if len(completed)==10 and not failures else 1


if __name__ == '__main__':
    sys.exit(main())
