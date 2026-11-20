#!/usr/bin/env python3
"""Adopt the exact suspended CF/TX intake worker after original1600 releases RAM."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_complete_cf_delta_queue as original

u,identity,resources,memory=original.u,original.identity,original.resources,original.memory
REPORT=u.ROOT/'experiments/upstream-complete-cf-delta-recovery'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def watchdog(owner_pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Exact recovery owner required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(owner_pid)):
            if (REPORT/'owner-release.json').exists():return
            captured=u.read(REPORT/'child-ownership.json')
            stopped=False
            if identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            u.write(REPORT/'watchdog-recovery.json',dict(reason='Exact recovery monitor lost',
                captured=captured,actual_child_RAM_preserved=stopped,observed=stamp()))
            return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing recovery evidence and live state preserved')
    old=original.REPORT
    failure=u.read(old/'failure.json');captured=failure['child'];actual=identity.process(captured['pid'])
    old_owner=u.read(old/'monitor.json');frozen=u.read(old/'plan.json')['source_sha256']
    expected=u.ROOT/'scripts/upstream_complete_cf_delta.py'
    if (failure['error']!="RuntimeError('Delta resource guard; owned child RAM preserved')" or
        not failure['actual_child_RAM_preserved'] or not failure['no_automatic_retry'] or
        identity.same_owner(old_owner,identity.process(old_owner['pid'])) or
        not identity.same_owner(captured,actual) or 'T' not in actual['state'] or
        actual['pgid']!=actual['pid'] or str(expected)+' admit' not in actual['command']):
        raise ValueError('Only the exact suspended orphan intake worker may be adopted')
    for rel,digest in frozen.items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original immutable delta source changed')
    if (u.ROOT/'experiments/upstream-complete-cf-delta').exists():
        raise ValueError('This recovery expects source intake before any study artifacts or fit')
    REPORT.mkdir();new_child=None;watch=None
    u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    u.write(REPORT/'child-ownership.json',captured)
    files=[Path(__file__),Path(original.__file__),old/'plan.json',old/'failure.json',old/'monitor.json']
    bindings=dict(frozen)
    bindings.update({str(p.relative_to(u.ROOT)):u.sha(p) for p in files})
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,adopted_child=captured,created=stamp(),
        original_monitor_confirmed_terminal=old_owner,
        rule='Wait independently complete original1600, then three consecutive compressor-below4.5GiB/reserve3GiB observations. Resume SAME actual suspended intake process, preserving original source, RAM and guards. Continuous own3GiB/research24GiB/compressor5GiB/swapgrowth128MiB/disk18.75GiB checks. No new intake launch or retry. Actual independent novelty, fixed ridge and independent fit replay follow. Dedicated child/watchdog identity checks; no GPU/primary locks or peer signals.'))
    def check():
        for rel,digest in bindings.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen recovery/source evidence changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('Delta recovery monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    native=u.ROOT/'experiments/upstream-high-capacity-successor/completion.json'
    baseline=None
    def wait_capacity(phase):
        good=0
        while True:
            check();workers=resources.footprints(os.getpid());current=memory.memory()
            ready=(original.safe(workers,current,reserve=3*2**30,baseline=baseline) and
                current['compressor_bytes']<4.5*2**30)
            good=good+1 if ready else 0
            if good>=3:return current
            u.write(REPORT/'status.json',dict(state='waiting_stable_original_resource_headroom',
                next_phase=phase,good_samples=good,workers=workers,memory=current,observed=stamp()))
            time.sleep(10)
    def monitor(phase):
        check();workers=resources.footprints(captured['pid']);current=memory.memory()
        own=workers.get(str(captured['pid']),dict(bytes=0,rss_bytes=0))
        if (not original.safe(workers,current,baseline=baseline) or
            max(own['bytes'],own['rss_bytes'])>3*2**30 or watch.poll() is not None):
            raise RuntimeError('Recovery resource/watchdog guard; same owned child RAM preserved')
        u.write(REPORT/'status.json',dict(state=phase,pid=captured['pid'],workers=workers,memory=current,observed=stamp()))
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],
                    cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while not native.exists():
                check()
                current=identity.process(captured['pid'])
                if not identity.same_owner(captured,current) or 'T' not in current['state']:
                    raise ValueError('Original suspended intake identity/state changed while waiting')
                if watch.poll() is not None:raise RuntimeError('Recovery watchdog lost while waiting')
                u.write(REPORT/'status.json',dict(state='waiting_independent_original1600_completion',
                    suspended_child=captured,observed=stamp()));time.sleep(10)
            if u.read(native)['state']!='complete_original1600_and_independent_full_output_verification':
                raise ValueError('Complete original1600 proof required before resuming')
            bindings[str(native.relative_to(u.ROOT))]=u.sha(native)
            baseline=wait_capacity('resume_same_intake')
            current=identity.process(captured['pid'])
            if not identity.same_owner(captured,current) or 'T' not in current['state']:
                raise ValueError('Same stopped intake required immediately before continuation')
            if watch.poll() is not None:raise RuntimeError('Watchdog unavailable before continuation')
            os.killpg(captured['pid'],signal.SIGCONT)
            u.write(REPORT/'continuation.json',dict(same_actual_process=captured,continued=stamp(),restarted=False))
            while identity.same_owner(captured,identity.process(captured['pid'])):
                monitor('same_original_intake_resumed');time.sleep(3)
            im=u.ROOT/'experiments/upstream-complete-cf-delta/input-manifest.json'
            if not im.exists() or u.read(im)['state']!='complete_all488_scalar_pairs_pending_independent_replay':
                raise ValueError('Adopted intake terminal without complete manifest; no restart')
            u.write(REPORT/'adopted-intake-terminal.json',dict(input_manifest_sha256=u.sha(im),
                exit_status_available=False,completion_to_be_proved_by_full_independent_replay=True,observed=stamp()))
            for phase,script,action in [('independent_novelty','verify_upstream_complete_cf_delta.py','novelty'),
                ('fit','upstream_complete_cf_delta.py','fit'),('independent_fit','verify_upstream_complete_cf_delta.py','fit')]:
                wait_capacity(phase);path=u.ROOT/'scripts'/script
                with (REPORT/(phase+'.log')).open('x') as log:
                    captured=None
                    new_child=subprocess.Popen([sys.executable,'-u',str(path),action],cwd=u.ROOT,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    captured=identity.process(new_child.pid)
                    if not captured or captured['pgid']!=new_child.pid or str(path) not in captured['command']:
                        if new_child.poll() is None:os.killpg(new_child.pid,signal.SIGSTOP)
                        raise ValueError('Actual dedicated recovery child required')
                    u.write(REPORT/'child-ownership.json',captured)
                    while new_child.poll() is None:monitor(phase);time.sleep(3)
                    if new_child.returncode:raise RuntimeError('Recovery phase failed; no retry: '+phase)
                    new_child=None
            proof=u.ROOT/'experiments/upstream-complete-cf-delta/independent-verification.json'
            if u.read(proof)['state']!='complete_independent_expanded_CF95_TX50n_known_ridge_and_canonical_scores':
                raise ValueError('Complete independent actual-fit/canonical proof required')
            u.write(REPORT/'completion.json',dict(state='complete_same_intake_adoption_novelty_fixed_ridge_and_independent_replay',
                proof_sha256=u.sha(proof),observed=stamp(),qualifying_labels_read=False))
            u.write(REPORT/'status.json',dict(state='complete',observed=stamp()))
            u.write(REPORT/'owner-release.json',dict(state='all_owned_workers_terminal_after_full_independent_replay'))
    except BaseException as error:
        if new_child is not None and new_child.poll() is None:
            if captured is None or identity.same_owner(captured,identity.process(new_child.pid)):
                os.killpg(new_child.pid,signal.SIGSTOP)
        elif captured and identity.same_owner(captured,identity.process(captured['pid'])):
            os.killpg(captured['pid'],signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=True,
            no_automatic_retry=True,observed=stamp()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args()
    watchdog(args.watch) if args.watch else main()
