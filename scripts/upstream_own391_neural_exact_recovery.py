#!/usr/bin/env python3
"""Adopt the exact original H128 process, then replay every saved pipeline."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_own391_paired_queue as original

u,identity,resources,memory=original.u,original.identities,original.resources,original.memory
REPORT=u.ROOT/'experiments/upstream-own391-neural-exact-recovery'
OLD=original.directory('h128')


def watchdog(pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=pid:raise ValueError('Exact monitor required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(pid)):
            if (REPORT/'owner-release.json').exists():return
            captured=u.read(REPORT/'child-ownership.json');stopped=False
            if identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            u.write(REPORT/'watchdog-recovery.json',dict(child=captured,actual_child_RAM_preserved=stopped,observed=original.stamp()))
            return
        time.sleep(2)


def admission(workers,current,baseline,pid):
    # Already-preserved worker RAM counts toward its original 16GiB reservation.
    reserved={k:dict(v) for k,v in workers.items()}
    own=reserved.get(str(pid))
    if own is None:return False
    if max(own['bytes'],own['rss_bytes'])>16*2**30:return False
    own['bytes']=max(own['bytes'],16*2**30)
    own['rss_bytes']=max(own['rss_bytes'],16*2**30)
    return original.safe(reserved,current,baseline) and current['compressor_bytes']<4.5*2**30


def main():
    if REPORT.exists():raise ValueError('Existing exact recovery retained')
    failure=u.read(OLD/'failure.json');captured=failure['child'];current=identity.process(captured['pid'])
    old_owner=u.read(OLD/'monitor.json')
    if (failure['error']!="RuntimeError('Original neural resource/watchdog guard; preserve exact child RAM')" or
        not failure['actual_child_RAM_preserved'] or not failure['no_automatic_retry'] or
        identity.same_owner(old_owner,identity.process(old_owner['pid'])) or
        not identity.same_owner(captured,current) or 'T' not in current['state'] or
        current['pgid']!=current['pid'] or
        str(u.ROOT/'scripts/upstream_own391_paired_mps.py')+' --variant h128 --worker' not in current['command'] or
        (OLD/'completion.json').exists()):raise ValueError('Only exact stopped original orphan H128 fit may continue')
    frozen=dict(u.read(OLD/'plan.json')['source_sha256'])
    for path in [Path(__file__),u.ROOT/'OWN391_EXACT_NEURAL_RECOVERY.md',OLD/'plan.json',OLD/'failure.json',OLD/'monitor.json',
        u.ROOT/'experiments/upstream-high-capacity-successor/completion.json']:
        key=str(path.relative_to(u.ROOT));digest=u.sha(path)
        if key in frozen and frozen[key]!=digest:raise ValueError('Conflicting original frozen expectation')
        frozen[key]=digest
    baseline=u.read(OLD/'status.json')['memory']
    REPORT.mkdir();watch=None;child=None;lane=None
    u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    u.write(REPORT/'child-ownership.json',captured)
    u.write(REPORT/'plan.json',dict(source_sha256=frozen,original_child=captured,baseline=baseline,created=original.stamp(),
        rule='Resume SAME stopped H128 process only, retaining tensors/optimizer/RNG and all original numerical sources. CF intake remains stopped. Three stable comp<4.5GiB observations, account already allocated RAM toward original16GiB reservation, nonblocking original GPU lane then recheck. Original own16/research24/comp5/swapgrowth128MiB/disk18.75 unchanged. Baseline is last recorded original preguard observation, not reset at admission. Full actual model/scaler/input replay then separate canonical verifier. Original failure/plan never overwritten. Supplemental original queue completion only after full proof for existing dependent waiters. No new fitting worker or automatic retry.'))
    def check():
        for rel,digest in frozen.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen source/evidence changed: '+rel)
    def interrupted(signum,frame):raise KeyboardInterrupt('Exact neural monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def capacity(phase,adopt=False):
        good=0
        while True:
            check();workers=resources.footprints(captured['pid']);current=memory.memory()
            ready=admission(workers,current,baseline,captured['pid']) if adopt else original.safe(workers,current,baseline,reserve=16*2**30) and current['compressor_bytes']<4.5*2**30
            good=good+1 if ready else 0
            if watch.poll() is not None:raise RuntimeError('Exact recovery watchdog unavailable')
            if good>=3:return
            u.write(REPORT/'status.json',dict(state='waiting_stable_original_neural_capacity',next_phase=phase,
                good_samples=good,workers=workers,memory=current,observed=original.stamp()));time.sleep(10)
    def monitor(phase):
        check();workers=resources.footprints(captured['pid']);current=memory.memory()
        own=workers.get(str(captured['pid']))
        if own is None:
            if not identity.same_owner(captured,identity.process(captured['pid'])):return
            raise ValueError('Cannot account exact live neural worker')
        if not original.safe(workers,current,baseline) or max(own['bytes'],own['rss_bytes'])>16*2**30 or watch.poll() is not None:
            u.write(REPORT/'guard-sample.json',dict(workers=workers,memory=current,watchdog_exit=watch.poll(),observed=original.stamp()))
            raise RuntimeError('Original exact neural resource guard; preserve same RAM')
        u.write(REPORT/'status.json',dict(state=phase,pid=captured['pid'],workers=workers,memory=current,observed=original.stamp()))
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            lane=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a')
            while True:
                capacity('same_original_fit_continuation',True)
                try:
                    fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if admission(resources.footprints(captured['pid']),memory.memory(),baseline,captured['pid']):break
                    fcntl.flock(lane,fcntl.LOCK_UN)
                except BlockingIOError:pass
                time.sleep(10)
            current=identity.process(captured['pid'])
            if not identity.same_owner(captured,current) or 'T' not in current['state']:raise ValueError('Exact original stopped fit required')
            if watch.poll() is not None:raise RuntimeError('Watchdog lost before continuation')
            os.killpg(captured['pid'],signal.SIGCONT)
            u.write(REPORT/'continuation.json',dict(same_actual_process=captured,restarted=False,observed=original.stamp()))
            while identity.same_owner(captured,identity.process(captured['pid'])):monitor('same_original_H128_fit_continued');time.sleep(3)
            fit=u.ROOT/'experiments/upstream-own391-paired-h128-mps/fit-completion.json'
            if not fit.exists() or u.read(fit)['state']!='complete_all_five_original_budget_matched_B391_neural_fits':raise ValueError('Exact original fit terminal without complete receipt; no retry')
            u.write(REPORT/'adopted-fit-terminal.json',dict(fit_completion_sha256=u.sha(fit),exit_status_available=False,full_independent_replay_required=True))
            for phase,script in [('independent_actual_replay','verify_upstream_own391_paired_mps.py'),('separate_canonical_scores','verify_upstream_own391_paired_canonical.py')]:
                capacity(phase);path=u.ROOT/'scripts'/script
                with (REPORT/(phase+'.log')).open('x') as log:
                    captured=None
                    child=subprocess.Popen([sys.executable,'-u',str(path),'--variant','h128'],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    captured=identity.process(child.pid)
                    if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                        raise ValueError('Dedicated actual verification child required')
                    u.write(REPORT/'child-ownership.json',captured)
                    while child.poll() is None:monitor(phase);time.sleep(3)
                    if child.returncode:raise RuntimeError('Independent phase failed; no retry: '+phase)
                    child=None
            proof=u.ROOT/'experiments/upstream-own391-paired-h128-mps/independent-canonical-verification.json'
            if u.read(proof)['state']!='complete_separate_matched_B391_canonical_neural_and_fixed_half_scores':raise ValueError('Full independent canonical proof required')
            completion=dict(state='complete_matched_B391_neural_arm_actual_fits_replay_and_canonical_scores',variant='h128',
                canonical_verification_sha256=u.sha(proof),qualifying_predictions_generated=False,observed=original.stamp(),
                exact_recovery_plan_sha256=u.sha(REPORT/'plan.json'),original_guard_failure_preserved=True)
            u.write(REPORT/'completion.json',completion)
            if (OLD/'completion.json').exists():raise ValueError('Never overwrite original completion')
            u.write(OLD/'completion.json',completion)
            u.write(REPORT/'status.json',dict(state='complete',observed=original.stamp()))
            u.write(REPORT/'owner-release.json',dict(state='all_owned_workers_terminal_after_full_proof'))
    except BaseException as error:
        stopped=False
        if captured and identity.same_owner(captured,identity.process(captured['pid'])):
            os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
        elif child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGSTOP);stopped=True
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=stopped,no_automatic_retry=True,observed=original.stamp()))
        raise
    finally:
        if lane is not None:lane.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args()
    watchdog(args.watch) if args.watch else main()
