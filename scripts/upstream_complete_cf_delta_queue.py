#!/usr/bin/env python3
"""Wait for bounded CPU headroom, then admit/replay/fit/replay the full delta."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory
import upstream_cf96_primary_lane as identity

REPORT=u.ROOT/'experiments/upstream-complete-cf-delta-queue'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers,current,reserve=0,baseline=None):
    return (sum(w['bytes'] for w in workers.values())+reserve<24*2**30 and
        sum(w['rss_bytes'] for w in workers.values())+reserve<24*2**30 and
        current['compressor_bytes']<5*2**30 and
        (baseline is None or current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20) and
        shutil.disk_usage(u.ROOT).free>=18.75*2**30)


def main():
    if REPORT.exists():raise ValueError('Existing delta queue retained')
    REPORT.mkdir();child=None;captured=None
    files=[Path(__file__),Path(resources.__file__),Path(memory.__file__),Path(identity.__file__),
        u.ROOT/'OWN_COMPLETE_CF_DELTA_PLAN.md',u.ROOT/'scripts/upstream_complete_cf_delta.py',
        u.ROOT/'scripts/verify_upstream_complete_cf_delta.py',
        u.ROOT/'experiments/upstream-cf96-complete-bank/independent-verification.json',
        u.ROOT/'experiments/upstream-remote15h-ext-return/s3-tx50n-v1/local-verification.json',
        u.ROOT/'experiments/upstream-own391-paired-data/independent-verification.json']
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,created=stamp(),cpu_only=True,
        rule='Reserve3GiB before every phase; own3GiB/research24GiB/compressor5GiB/swapgrowth128MiB/disk18.75GiB. Never acquire GPU/primary lanes or signal peers. No automatic retries. Preserve owned child RAM on monitor interruption or resource guard.'))
    u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    def check():
        for k,d in bindings.items():
            if u.sha(u.ROOT/k)!=d:raise ValueError('Frozen delta queue source changed: '+k)
    def interrupted(signum,frame):raise KeyboardInterrupt('Delta monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            phases=[('admit','upstream_complete_cf_delta.py','admit'),
                ('independent_novelty','verify_upstream_complete_cf_delta.py','novelty'),
                ('fit','upstream_complete_cf_delta.py','fit'),
                ('independent_fit','verify_upstream_complete_cf_delta.py','fit')]
            baseline=None
            for phase,script,action in phases:
                while True:
                    check();workers=resources.footprints(os.getpid());current=memory.memory()
                    if safe(workers,current,reserve=3*2**30,baseline=baseline):break
                    u.write(REPORT/'status.json',dict(state='waiting_three_GiB_and_compressor_below_five',
                        next_phase=phase,workers=workers,memory=current,observed=stamp()))
                    time.sleep(10)
                if baseline is None:baseline=current
                path=u.ROOT/'scripts'/script
                with (REPORT/(phase+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(path),action],cwd=u.ROOT,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    captured=identity.process(child.pid)
                    if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                        raise ValueError('Dedicated actual delta child required')
                    u.write(REPORT/'child-ownership.json',captured)
                    while child.poll() is None:
                        check();workers=resources.footprints(child.pid);current=memory.memory()
                        own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                        if not safe(workers,current,baseline=baseline) or max(own['bytes'],own['rss_bytes'])>3*2**30:
                            raise RuntimeError('Delta resource guard; owned child RAM preserved')
                        u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,
                            workers=workers,memory=current,observed=stamp()))
                        time.sleep(3)
                    if child.returncode:raise RuntimeError('Delta phase failed; no automatic retry: '+phase)
                    child=None;captured=None
            proof=u.ROOT/'experiments/upstream-complete-cf-delta/independent-verification.json'
            if u.read(proof)['state']!='complete_independent_expanded_CF95_TX50n_known_ridge_and_canonical_scores':
                raise ValueError('Complete independent fixed study required')
            u.write(REPORT/'completion.json',dict(state='complete_full_delta_admission_fixed_ridge_and_independent_replay',
                proof_sha256=u.sha(proof),observed=stamp(),qualifying_labels_read=False))
            u.write(REPORT/'status.json',dict(state='complete',observed=stamp()))
    except BaseException as error:
        if child is not None and child.poll() is None and (captured is None or identity.same_owner(captured,identity.process(child.pid))):
            os.killpg(child.pid,signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=True,
            no_automatic_retry=True,observed=stamp()))
        raise


if __name__=='__main__':main()
