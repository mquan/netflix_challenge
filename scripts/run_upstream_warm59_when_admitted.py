#!/usr/bin/env python3
"""Serialize the declared warm59 diagnostic behind existing meta fitting."""
import fcntl
from pathlib import Path
import subprocess
import sys
import upstream_warm59_screen as experiment

u=experiment.u

if __name__=='__main__':
    report=experiment.REPORT
    with (report/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=u.ROOT/'experiments/upstream-source-extension/run.lock'
        with path.open('a') as shared:
            u.write(report/'status.json',dict(state='waiting_for_existing_meta_fit_lock',observed=experiment.resources.stamp()))
            fcntl.flock(shared,fcntl.LOCK_EX)
            for script in ['upstream_warm59_screen.py','verify_upstream_warm59_screen.py']:
                with (report/(script+'.log')).open('x') as log:
                    subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            u.write(report/'status.json',dict(state='complete_ridge_diagnostic_and_independent_replay',observed=experiment.resources.stamp()))
