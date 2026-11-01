"""Execute the README command unchanged in a tracked-source snapshot."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path('/workspace/Community-aware-Graph-Transformer')
OUT = ROOT / 'runs/official-readme-20261101'
COMMAND = 'python exp.py --dataset cora'


def main():
    if (OUT / 'attempt.json').exists():
        raise SystemExit('Completed attempt already exists; preserve it.')
    source = OUT / 'source'
    source.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    paths = subprocess.check_output(['git', '-C', str(UPSTREAM), 'ls-files', '-z']).decode().split('\0')
    hashes = {}
    for name in filter(None, paths):
        original = UPSTREAM / name
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        hashes[name] = hashlib.sha256(original.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == hashes[name]
    # Supply the same public input files, without altering source or arguments.
    destination = source / 'original_datasets/planetoid/Cora'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'data/Cora', destination, dirs_exist_ok=True)
    environment = dict(os.environ)
    environment.update(PATH='/workspace/venvs/cgt/bin:' + environment['PATH'],
                       PYTHONDONTWRITEBYTECODE='1', DGLBACKEND='pytorch',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    protocol = dict(command=COMMAND, cwd=str(source), upstream_commit=commit,
                    source_sha256=hashes, source_edits=False,
                    environment_note='Existing Python3.10 / PyTorch2.0.1+cpu environment; not the original CUDA environment.yml.',
                    data_note='Copy of the same public Cora inputs; no source or CLI edits.',
                    timeout_seconds=600)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    timed_out = False
    try:
        with (OUT / 'command.log').open('w') as stream:
            result = subprocess.run(['bash', '-c', COMMAND], cwd=source, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
        return_code = result.returncode
    except subprocess.TimeoutExpired:
        return_code = None
        timed_out = True
    unchanged = all(hashlib.sha256((source/n).read_bytes()).hexdigest() == h for n,h in hashes.items())
    assert unchanged
    outcome = dict(command=COMMAND, return_code=return_code, timed_out=timed_out,
                   elapsed_seconds=time.monotonic()-started, original_source_unchanged=True,
                   status='command_completed' if return_code == 0 else 'failed_before_valid_performance_result',
                   accuracy=None,
                   interpretation='A startup/environment error is not an 84% result and cannot establish paper reproducibility.')
    (OUT / 'attempt.json').write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps(outcome, indent=2), flush=True)
    print((OUT / 'command.log').read_text()[-5000:], flush=True)


if __name__ == '__main__':
    main()
