#!/usr/bin/env python3
"""Adopt the preserved declaration with reversible read-cache control."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_own391_cpu_tree_queue as original
import upstream_read_cache_policy as cache

u,identity,resources,memory=original.u,original.identity,original.resources,original.memory
REPORT=u.ROOT/'experiments/upstream-own391-cpu-tree-recovery'


def watchdog(pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=pid:raise ValueError('Exact recovery owner required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(pid)):
            if (REPORT/'owner-release.json').exists():return
            captured=u.read(REPORT/'child-ownership.json');stopped=False
            if identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            cache.restore_after_owner_loss(REPORT)
            cache.write(REPORT/'watchdog-recovery.json',dict(child=captured,actual_group_RAM_preserved=stopped,
                prior_journaled_cache_policies_restored=True,observed=original.stamp()));return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing same-process recovery retained')
    old=original.REPORT;failure=u.read(old/'failure.json');captured=failure['child'];current=identity.process(captured['pid'])
    old_owner=u.read(old/'monitor.json');frozen=dict(u.read(old/'plan.json')['source_sha256'])
    if (failure['error']!="RuntimeError('CPU tree resource/watchdog guard; complete group RAM preserved')" or
        not failure['actual_group_RAM_preserved'] or identity.same_owner(old_owner,identity.process(old_owner['pid'])) or
        not identity.same_owner(captured,current) or 'T' not in current['state'] or current['pgid']!=current['pid'] or
        str(u.ROOT/'scripts/upstream_own391_tree_source.py')+' declare' not in current['command'] or
        (u.ROOT/'experiments/upstream-own391-cpu-tree').exists() or (u.ROOT/'artifacts/upstream-own391-cpu-tree').exists()):
        raise ValueError('Only the exact stopped prefit declaration before artifacts may be adopted')
    for relative,digest in frozen.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Original frozen numerical source changed')
    files=[Path(__file__),Path(cache.__file__),old/'plan.json',old/'failure.json',old/'monitor.json',
        u.ROOT/'experiments/upstream-uncached-hash-fixture/verification.json',
        u.ROOT/'experiments/upstream-own391-cpu-tree-pairing-check.json']
    for path in files:
        key=str(path.relative_to(u.ROOT));digest=u.sha(path)
        if key in frozen and frozen[key]!=digest:raise ValueError('Original source expectation conflicts')
        frozen[key]=digest
    REPORT.mkdir();owner=identity.process(os.getpid());policy=cache.Policy(REPORT,owner);child=None;watch=None
    cache.write(REPORT/'monitor.json',owner);cache.write(REPORT/'child-ownership.json',captured)
    cache.write(REPORT/'plan.json',dict(source_sha256=frozen,original_child=captured,created=original.stamp(),
        rule='Adopt SAME stopped prefit declaration; no restart or numerical-source/budget changes. Temporarily setF_GLOBAL_NOCACHE only on explicit bound immutable large feature/design matrices, preservepriorstates/inodes, journalandrestore oncompletion/error/ownerloss. Readonly policy setup hasbounded descriptors/buffers andnoMLallocation; resumeonlyafteroriginal8GiB reservation,3stablecomp<3GiB,24research/5compress/128swap/18.75disk. Full CPUfit andindependentactualsavedmodel/canonical replay unchanged. All peerprocesses leftalone.'))
    def check():
        for relative,digest in frozen.items():
            if u.sha(u.ROOT/relative)!=digest:raise ValueError('Frozen recovery/numerical source changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('CPU tree recovery interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    baseline=u.read(old/'status.json')['memory']
    def wait_capacity(phase):
        good=0
        while True:
            check();workers=resources.footprints(os.getpid());current=memory.memory()
            good=good+1 if original.safe(workers,current,baseline,8*2**30) and current['compressor_bytes']<3*2**30 else 0
            if watch.poll() is not None:raise RuntimeError('Independent cache/RAM watchdog lost')
            if good>=3:return
            cache.write(REPORT/'status.json',dict(state='waiting_original_headroom_with_uncached_inputs',next_phase=phase,
                good_samples=good,workers=workers,memory=current,observed=original.stamp()));time.sleep(10)
    def monitor(phase):
        check();workers=resources.footprints(captured['pid']);current=memory.memory()
        groups={s.split()[0]:int(s.split()[1]) for s in subprocess.check_output(['ps','-axo','pid=,pgid='],text=True).splitlines() if len(s.split())==2}
        own=dict(bytes=sum(w['bytes'] for pid,w in workers.items() if groups.get(pid)==captured['pgid']),
            rss_bytes=sum(w['rss_bytes'] for pid,w in workers.items() if groups.get(pid)==captured['pgid']))
        if not original.safe(workers,current,baseline) or max(own['bytes'],own['rss_bytes'])>8*2**30 or watch.poll() is not None:
            raise RuntimeError('Original CPU tree guard; same complete group RAM preserved')
        cache.write(REPORT/'status.json',dict(state=phase,pid=captured['pid'],group_memory=own,
            workers=workers,memory=current,observed=original.stamp()))
    success=None
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],
                    cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            policy.apply();wait_capacity('continue_same_declaration')
            current=identity.process(captured['pid'])
            if not identity.same_owner(captured,current) or 'T' not in current['state']:
                raise ValueError('Exact suspended declaration required immediately before continuation')
            os.killpg(captured['pid'],signal.SIGCONT)
            cache.write(REPORT/'continuation.json',dict(same_actual_process=captured,restarted=False,continued=original.stamp()))
            while identity.same_owner(captured,identity.process(captured['pid'])):
                monitor('same_original_declaration_resumed_uncached');time.sleep(3)
            manifest=u.ROOT/'experiments/upstream-own391-cpu-tree/manifest.json'
            if not manifest.exists() or u.read(manifest)['state']!='declared_full_budget_matched391_cpu_tree':
                raise ValueError('Adopted declaration terminal without complete manifest; no restart')
            cache.write(REPORT/'adopted-declaration-terminal.json',dict(manifest_sha256=u.sha(manifest),
                exit_status_available=False,full_input_and_fit_verification_required=True,observed=original.stamp()))
            for phase,script in [('full_nested_fit','upstream_own391_tree_worker.py'),('independent_replay','verify_upstream_own391_cpu_tree.py')]:
                wait_capacity(phase);path=u.ROOT/'scripts'/script
                with (REPORT/(phase+'.log')).open('x') as log:
                    captured=None
                    child=subprocess.Popen([sys.executable,'-u',str(path)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    captured=identity.process(child.pid)
                    if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                        raise ValueError('Dedicated actual original CPU tree worker required')
                    cache.write(REPORT/'child-ownership.json',captured);cache.write(REPORT/(phase+'-ownership.json'),captured)
                    while child.poll() is None:monitor(phase);time.sleep(3)
                    if child.returncode:raise RuntimeError('CPU tree phase failed without retry: '+phase)
                    child=None
            proof=u.ROOT/'experiments/upstream-own391-cpu-tree/independent-verification.json'
            if u.read(proof)['state']!='complete_independent_matched391_CPU_tree_models_inputs_ridge_predictions_and_canonical_scores':
                raise ValueError('Full independent original CPU tree proof required')
            success=dict(state='complete_same_declaration_CPU_tree_full_fits_and_replay_with_restored_cache_policy',
                verification_sha256=u.sha(proof),qualifying_labels_read=False,observed=original.stamp())
    except BaseException as error:
        if child is not None and child.poll() is None:
            if captured is None or identity.same_owner(captured,identity.process(child.pid)):os.killpg(child.pid,signal.SIGSTOP)
        elif captured and identity.same_owner(captured,identity.process(captured['pid'])):os.killpg(captured['pid'],signal.SIGSTOP)
        cache.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_group_RAM_preserved=True,no_automatic_retry=True,observed=original.stamp()));raise
    finally:
        policy.restore()
        cache.write(REPORT/'owner-release.json',dict(state='owned_group_terminal_or_stopped_and_prior_cache_policy_restored'))
    if success:
        success['cache_restoration_sha256']=u.sha(REPORT/'cache-policy-restoration.json')
        cache.write(REPORT/'completion.json',success);cache.write(REPORT/'status.json',dict(state='complete',observed=original.stamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args();watchdog(args.watch) if args.watch else main()
