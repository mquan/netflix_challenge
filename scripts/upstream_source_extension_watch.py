#!/usr/bin/env python3
"""Wait outside the shared fitting lock; lock only complete-source preparation."""
import fcntl
import time
import upstream_source_extension as experiment

u=experiment.u
REPORT=experiment.REPORT


def step():
    ready=experiment.inventory()
    parent_ready=(experiment.parent.REPORT/'independent-verification.json').exists()
    if not all(ready.values()) or not parent_ready:
        u.write(REPORT/'status.json',dict(state='waiting_for_complete_source_jobs',ready=ready,parent_ready=parent_ready,
            fitting_lock_held=False))
        return False
    with (REPORT/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            u.write(REPORT/'status.json',dict(state='waiting_for_prepare_lock',ready=ready,fitting_lock_held=False))
            return False
        experiment.prepare()
        u.write(REPORT/'status.json',dict(state='prepared',nonlinear_training_started=False))
        return True


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not step():time.sleep(30)


if __name__=='__main__':run()
