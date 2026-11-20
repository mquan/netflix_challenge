#!/usr/bin/env python3
"""Stop an orphaned refit child; keep active hash-bound scripts unchanged."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-nn-pool6-refit-watchdog'
QUEUE=u.ROOT/'experiments/upstream-nn-pool6-full-refit-queue'

def command(pid):
    p=subprocess.run(['ps','-p',str(pid),'-o','command='],text=True,capture_output=True)
    return p.stdout.strip() if p.returncode==0 else ''

def main(supervisor):
    REPORT.mkdir(exist_ok=False)
    plan=u.read(QUEUE/'plan.json')
    paths={'worker_sha256':u.ROOT/'scripts/upstream_nn_pool6_full_refit.py',
        'queue_sha256':u.ROOT/'scripts/upstream_nn_pool6_full_refit_queue.py',
        'declaration_sha256':u.ROOT/'experiments/upstream-native-nn-pool6-production/declaration.json'}
    while True:
        complete=(QUEUE/'completion.json').exists()
        alive=str(paths['queue_sha256']) in command(supervisor)
        changed=[key for key,p in paths.items() if u.sha(p)!=plan[key]]
        if complete:
            u.write(REPORT/'verification.json',dict(state='queue_complete_no_orphans',supervisor=supervisor));return
        if not alive or changed:
            status=u.read(QUEUE/'status.json');pid=status.get('pid')
            stopped=False
            if pid and str(paths['worker_sha256']) in command(pid) and os.getpgid(pid)==pid:
                os.killpg(pid,signal.SIGTERM);stopped=True
                for _ in range(10):
                    if not command(pid):break
                    time.sleep(1)
                if str(paths['worker_sha256']) in command(pid):os.killpg(pid,signal.SIGKILL)
            if changed and alive:os.kill(supervisor,signal.SIGTERM)
            u.write(REPORT/'verification.json',dict(state='stopped_orphan_or_source_change',
                supervisor_alive=alive,changed_bindings=changed,owned_child=pid,owned_child_stopped=stopped,
                partial_outputs_preserved=True));return
        time.sleep(5)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('supervisor',type=int);a=p.parse_args();main(a.supervisor)
