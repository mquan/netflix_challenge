#!/usr/bin/env python3
"""Serialize paired item154 preparation and queue its known-selected full NN."""
import fcntl
from pathlib import Path
import subprocess
import sys
import upstream_item154_stream_production as production

u=production.u
REPORT=u.ROOT/'experiments/upstream-item154-production-queue'

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing full-fit queue evidence preserved')
        proof,counts,_=production.declaration()
        paths=[Path(__file__),u.ROOT/'scripts/upstream_item154_stream_bank.py',Path(production.__file__),u.ROOT/'scripts/verify_upstream_item154_stream_production.py']
        hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
        u.write(REPORT/'plan.json',dict(created=production.resources.stamp(),development_rmse=proof['rmse'],epochs=counts,
            weights=[.5,.5],code_sha256=hashes,qualifying_labels_read=False,
            rule='Prepare and verify streamed paired context under the shared fit lock, then fit/replay full new item154 NN with known-fold median budgets. Reuse original full tree. Freeze fixed mean before new canonical-label audit.'))
        u.write(REPORT/'status.json',dict(state='waiting_for_shared_preparation_slot'))
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            with (REPORT/'bank.log').open('x') as log:
                subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/upstream_item154_stream_bank.py')],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for name,digest in hashes.items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Declared full-fit code changed')
        with (REPORT/'production.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(production.__file__))],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        u.write(REPORT/'status.json',dict(state='verified_stream_bank_and_full_controller_started',pid=child.pid,observed=production.resources.stamp()))

if __name__=='__main__':run()
