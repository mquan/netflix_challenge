#!/usr/bin/env python3
"""Recover only reference-transform supervision; frozen numerical worker unchanged."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import upstream_knn3pair2_virtual_queue as resource
import upstream_split3eight_scaling_parity as parity
import upstream_split3eight_compact_queue as queue
u=parity.u
REPORT=u.ROOT/'experiments/upstream-split3eight-scaling-recovery'

def main():
    if REPORT.exists():raise ValueError('Existing recovery evidence preserved')
    REPORT.mkdir()
    failure=parity.REPORT/'failure.json'
    if u.read(failure)['error']!='ModuleNotFoundError("No module named \'upstream_split3eight_virtual_queue\'")':
        raise ValueError('Only the exact supervision import failure is authorized')
    if (parity.REPORT/'verification.json').exists():raise ValueError('Complete parity evidence already exists')
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(parity.__file__),
        Path(resource.__file__),failure,parity.REPORT/'plan.json',parity.REPORT/'cache-verification.json']}
    u.write(REPORT/'plan.json',dict(created=parity.stamp(),bindings=bindings,
        original_child_pid=31196,original_child_stopped=True,models_fitted=0,
        reason='Frozen run wrapper referenced a nonexistent renamed resource helper after launching its transform child. Stop only owned incomplete transform child and preserve original failure. Import verified existing resource helper BEFORE launching the unchanged frozen --worker. Reuse complete exact caches, same all-row reference statistics and every-bit checks, original total24GiB guard and shared epoch lock. No model/data/budget/source rewrite.',
        qualifying_labels_read=False))
    with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as slot:
        u.write(REPORT/'status.json',dict(state='waiting_for_complete_current_raw_epoch',observed=parity.stamp()))
        fcntl.flock(slot,fcntl.LOCK_EX)
        for name,digest in bindings.items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Recovery source binding changed')
        with (REPORT/'reference-worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(parity.__file__)),'--worker'],cwd=u.ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            high=0
            try:
                while child.poll() is None:
                    own=resource.model.base.preceding.group_rss(child.pid)
                    other=resource.model.base.preceding.resources.other_rss({child.pid})
                    free=parity.shutil.disk_usage(u.ROOT).free
                    high=high+1 if own+other>24*2**30 else 0
                    u.write(REPORT/'status.json',dict(state='unchanged_full_reference_transform',observed=parity.stamp(),
                        pid=child.pid,own_rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free))
                    if high>=3 or free<8*2**30:raise RuntimeError('Original resource guard; preserve all evidence')
                    time.sleep(5)
            except BaseException:
                resource.model.base.preceding.stop(child)
                raise
            if child.returncode:raise RuntimeError('Unchanged full reference worker failed')
        proof=u.read(parity.REPORT/'verification.json')
        if proof['columns']!=406 or proof['rows']!=1128164:raise ValueError('Full parity scope differs')
    u.write(REPORT/'reference-completion.json',dict(observed=parity.stamp(),
        verification_sha256=u.sha(parity.REPORT/'verification.json'),shared_epoch_lock_released=True))
    with (REPORT/'neural-queue.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(Path(queue.__file__))],cwd=u.ROOT,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    u.write(REPORT/'completion.json',dict(observed=parity.stamp(),
        study_completion_sha256=u.sha(queue.REPORT/'completion.json'),qualifying_labels_read=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        REPORT.mkdir(parents=True,exist_ok=True)
        u.write(REPORT/'failure.json',dict(error=repr(error),artifacts_preserved=True));raise
