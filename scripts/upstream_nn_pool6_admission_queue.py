#!/usr/bin/env python3
"""Wait for complete fits, then admit only independently replayed pipelines."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-nn-pool6-admission-queue'
ORIGINAL=u.ROOT/'experiments/upstream-nn-pool6-full-refit-queue'
DECL=u.ROOT/'experiments/upstream-native-nn-pool6-production/declaration.json'
CHECKER=u.ROOT/'scripts/verify_upstream_nn_pool6_full_refit.py'

def main():
    REPORT.mkdir(exist_ok=False)
    original=u.read(ORIGINAL/'plan.json')
    paths={'queue_sha256':u.ROOT/'scripts/upstream_nn_pool6_full_refit_queue.py',
        'worker_sha256':u.ROOT/'scripts/upstream_nn_pool6_full_refit.py','declaration_sha256':DECL}
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),CHECKER]},
        rule='No assembly, score promotion or recovered outcomes before all five independent serialized-pipeline admissions. Verify original execution hashes again after completion.'))
    while not (ORIGINAL/'completion.json').exists():
        for key,p in paths.items():
            if u.sha(p)!=original[key]:raise ValueError('Frozen active execution source changed')
        time.sleep(10)
    for key,p in paths.items():
        if u.sha(p)!=original[key]:raise ValueError('Frozen completed execution source changed')
    lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    def interrupted(signum,frame):raise KeyboardInterrupt('Admission supervisor signal')
    signal.signal(signal.SIGTERM,interrupted)
    for entry in u.read(DECL)['families']:
        if entry['reuse_existing_full_base']:continue
        name=entry['development']
        with (REPORT/(name+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,str(CHECKER),name],cwd=u.ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                u.write(REPORT/'status.json',dict(state='independent_serialized_pipeline_replay',development=name,pid=child.pid))
                child.wait()
            except BaseException:
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                child.wait();raise
            if child.returncode:raise RuntimeError('Admission failed; refits and evidence preserved')
    for key,p in paths.items():
        if u.sha(p)!=original[key]:raise ValueError('Frozen completed execution source changed')
    u.write(REPORT/'completion.json',dict(state='complete_all_five_independent_full_pipeline_admissions',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),qualifying_labels_read=False,
        assembly_calibration_and_frozen_qualifying_audit_remaining=True))

if __name__=='__main__':main()
