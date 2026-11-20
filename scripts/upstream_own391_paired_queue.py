#!/usr/bin/env python3
"""Guarded matched391 neural arm, after the already queued original1600 recipe."""
import argparse
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
import upstream_cf96_primary_lane as identities


def directory(variant):return u.ROOT/('experiments/upstream-own391-paired-'+variant+'-queue')
def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers,current,baseline,reserve=0):
    return (sum(v['bytes'] for v in workers.values())+reserve<24*2**30 and
        sum(v['rss_bytes'] for v in workers.values())+reserve<24*2**30 and
        current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and
        current['compressor_bytes']<5*2**30 and shutil.disk_usage(u.ROOT).free>=18.75*2**30)


def watchdog(variant,supervisor):
    report=directory(variant);owner=u.read(report/'monitor.json')
    if owner['pid']!=supervisor:raise ValueError('Original neural monitor identity required')
    while not (report/'owner-release.json').exists():
        if not identities.same_owner(owner,identities.process(supervisor)):
            if (report/'owner-release.json').exists():return
            path=report/'child-ownership.json';captured=u.read(path) if path.exists() else None;stopped=False
            if captured and identities.same_owner(captured,identities.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            u.write(report/'watchdog-recovery.json',dict(state='monitor_lost_original_neural_child_RAM_preserved',child=captured,stopped=stopped,observed=stamp()))
            return
        time.sleep(2)


def main(variant,original1600_owner):
    report=directory(variant)
    if report.exists():raise ValueError('Existing complete/partial paired neural queue retained')
    data_proof=u.ROOT/'experiments/upstream-own391-paired-data/independent-verification.json'
    if u.read(data_proof)['state']!='complete_independent_matched_B391_all_operators_statistics_ridge_and_stored_predictions':
        raise ValueError('Complete matched paired391 data proof required before queueing')
    report.mkdir();child=None;captured=None;watch=None;lane=None
    u.write(report/'monitor.json',identities.process(os.getpid()))
    files=[Path(__file__),Path(resources.__file__),Path(memory.__file__),Path(identities.__file__),data_proof,
        u.ROOT/'OWN391_PAIRED_NEURAL_PLAN.md']+[u.ROOT/'scripts'/name for name in [
        'upstream_own391_paired_data.py','verify_upstream_own391_paired_data.py','upstream_exact391_scaler.py',
        'upstream_own391_paired_mps.py','verify_upstream_own391_paired_mps.py','verify_upstream_own391_paired_canonical.py']]
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    native_owner=identities.process(original1600_owner)
    native_complete=u.ROOT/'experiments/upstream-high-capacity-successor/completion.json'
    if not native_complete.exists() and (not native_owner or 'upstream_high_capacity_successor.py' not in native_owner['command']):
        raise ValueError('Exact original1600 queue must be live or independently complete')
    u.write(report/'plan.json',dict(variant=variant,created=stamp(),source_sha256=bindings,original1600_owner=native_owner,
        rule='Both neural arms wait original1600 completion to preserve its queue priority. Same391 pairedB raw/all90 learned contexts,618inputs,5userfolds/3seeds/max40pat4/batch8192/originalAdamW. H128 primary,H512 declared control. Reserve16GiB before nonblockingGPUlock;16GiB worker,24GiB research physical/RSS,5GiB compressor,128MiB swapgrowth,18.75GiB disk. Full actual eligible scalers/models/predictions and separate canonical score required. Actual child RAM preserved on guard/owner loss; no retries or numerical recipe changes. No qualifying model inference or outcome read.'))
    def check():
        for key,digest in bindings.items():
            if u.sha(u.ROOT/key)!=digest:raise ValueError('Frozen paired neural execution changed: '+key)
    def interrupted(signum,frame):raise KeyboardInterrupt('Original matched neural monitor signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while not native_complete.exists():
            check()
            if not identities.same_owner(native_owner,identities.process(original1600_owner)):
                if native_complete.exists():break
                raise ValueError('Original1600 owner terminal without full completion; no automatic replacement')
            u.write(report/'status.json',dict(state='waiting_original1600_priority',owner=original1600_owner,observed=stamp()));time.sleep(10)
        if u.read(native_complete)['state']!='complete_original1600_and_independent_full_output_verification':raise ValueError('Full original1600 independent completion required')
        bindings[str(native_complete.relative_to(u.ROOT))]=u.sha(native_complete)
        lane=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a')
        baseline=memory.memory()
        while True:
            check();workers=resources.footprints();current=memory.memory()
            if safe(workers,current,baseline,reserve=16*2**30):
                try:
                    fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    workers=resources.footprints();current=memory.memory()
                    if safe(workers,current,baseline,reserve=16*2**30):break
                    fcntl.flock(lane,fcntl.LOCK_UN)
                except BlockingIOError:pass
            u.write(report/'status.json',dict(state='waiting_sixteen_GiB_then_nonblocking_GPU',workers=workers,memory=current,observed=stamp()));time.sleep(10)
        with (report/'watchdog.log').open('x') as log:
            watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--variant',variant,'--watch',str(os.getpid())],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        phases=[('prepare','upstream_own391_paired_mps.py',[]),('fit','upstream_own391_paired_mps.py',['--worker']),
            ('independent_actual_replay','verify_upstream_own391_paired_mps.py',[]),
            ('separate_canonical_scores','verify_upstream_own391_paired_canonical.py',[])]
        for phase,script,args in phases:
            check();path=u.ROOT/'scripts'/script
            with (report/(phase+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(path),'--variant',variant,*args],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                captured=identities.process(child.pid)
                if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                    if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                    raise ValueError('Dedicated actual neural child identity required')
                u.write(report/'child-ownership.json',captured)
                while child.poll() is None:
                    check();workers=resources.footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                    if not safe(workers,current,baseline) or max(own['bytes'],own['rss_bytes'])>16*2**30 or watch.poll() is not None:
                        raise RuntimeError('Original neural resource/watchdog guard; preserve exact child RAM')
                    u.write(report/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,observed=stamp()));time.sleep(3)
                if child.returncode:raise RuntimeError('Original neural phase failed and preserved; no retry: '+phase)
                child=None;captured=None
        proof=u.ROOT/('experiments/upstream-own391-paired-'+variant+'-mps/independent-canonical-verification.json')
        if u.read(proof)['state']!='complete_separate_matched_B391_canonical_neural_and_fixed_half_scores':raise ValueError('Complete original saved-model and canonical proof required')
        u.write(report/'completion.json',dict(state='complete_matched_B391_neural_arm_actual_fits_replay_and_canonical_scores',variant=variant,canonical_verification_sha256=u.sha(proof),qualifying_predictions_generated=False,observed=stamp()))
        u.write(report/'status.json',dict(state='complete',observed=stamp()));u.write(report/'owner-release.json',dict(state='all_owned_neural_phases_terminal_after_independent_verification'))
    except BaseException as error:
        if child is not None and child.poll() is None and (captured is None or identities.same_owner(captured,identities.process(child.pid))):os.killpg(child.pid,signal.SIGSTOP)
        u.write(report/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=True,no_automatic_retry=True,observed=stamp()));raise
    finally:
        if lane is not None:lane.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['h128','h512'],default='h128');parser.add_argument('--original1600-owner',type=int,default=61480);parser.add_argument('--watch',type=int);args=parser.parse_args()
    watchdog(args.variant,args.watch) if args.watch else main(args.variant,args.original1600_owner)
