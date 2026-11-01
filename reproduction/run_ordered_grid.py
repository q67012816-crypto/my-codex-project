"""Start the bounded coefficient search only after the seed-extension stage ends."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/ordered-followup-20261101'
PAIRED=ROOT/'runs/cora-paired-30'


def write(value):
    temporary=OUT/'status.tmp'
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(OUT/'status.json')


def main():
    OUT.mkdir(exist_ok=True)
    start=(PAIRED/'protocol.json').stat().st_mtime
    deadline=start+180*60
    plan=dict(overall_training_budget_minutes=180,budget_assumption='Fixed maximum training duration: 180 minutes.',
              budget_start_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),
              stage4_target_pairs=30,stage4_max_minutes=100,
              stage5_seed=100,stage5_candidates=16,stage5_max_minutes=75,
              stage5_scope='One split pilot; validation-only choice then winner/reference test. Not a repeated paper reproduction.',
              original_command='Blocked by missing dependency and unavailable CUDA; downstream experiments explicitly retain the existing CPU protocol.')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    write(dict(status='waiting_for_seed_extension',plan=plan))
    while time.time()<deadline:
        if (OUT/'cancel-grid').exists():
            write(dict(status='cancelled_by_user'));return
        summary=json.loads((PAIRED/'summary.json').read_text())
        if summary['status']!='running':break
        time.sleep(5)
    else:
        write(dict(status='overall_budget_exhausted_before_grid'));return
    if summary['status'] not in ['completed','budget_exhausted'] or summary['failures']:
        write(dict(status='grid_not_started_due_to_seed_stage_failure',seed_status=summary['status']));return
    remaining=(deadline-time.time())/60-3
    minutes=min(75,remaining)
    if minutes<15:
        write(dict(status='insufficient_remaining_budget',remaining_minutes=remaining));return
    command=[sys.executable,'-u',str(ROOT/'run_validation_grid.py'),'--seed','100','--max-minutes',str(minutes)]
    write(dict(status='grid_running',budget_minutes=minutes,seed_pairs_completed=summary['completed_pairs']))
    print(json.dumps(dict(event='grid_started',budget_minutes=minutes)),flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',DGLBACKEND='pytorch')
    with (OUT/'grid-driver.log').open('w') as stream:
        result=subprocess.run(command,cwd=ROOT.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,
                              timeout=max(1,deadline-time.time()),check=False)
    write(dict(status='finished' if result.returncode==0 else 'grid_driver_failed',return_code=result.returncode))
    print(json.dumps(dict(event='ordered_followup_finished',return_code=result.returncode)),flush=True)


if __name__=='__main__':main()
