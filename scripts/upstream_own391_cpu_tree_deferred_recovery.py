#!/usr/bin/env python3
"""Resume the same readonly declaration after native/neural memory is released."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_own391_cpu_tree_queue as original

u,identity,resources,memory=original.u,original.identity,original.resources,original.memory
REPORT=u.ROOT/'experiments/upstream-own391-cpu-tree-deferred-recovery'


def watchdog(pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=pid:raise ValueError('Exact deferred CPU owner required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(pid)):
            if (REPORT/'owner-release.json').exists():return
            captured=u.read(REPORT/'child-ownership.json');stopped=False
            if identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            u.write(REPORT/'watchdog-recovery.json',dict(child=captured,actual_group_RAM_preserved=stopped,observed=original.stamp()));return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing deferred recovery retained')
    prior=u.ROOT/'experiments/upstream-own391-cpu-tree-recovery';failure=u.read(prior/'failure.json');captured=failure['child']
    old_owner=u.read(prior/'monitor.json');current=identity.process(captured['pid'])
    restored=u.read(prior/'cache-policy-restoration.json')
    if (failure['error']!="RuntimeError('Original CPU tree guard; same complete group RAM preserved')" or
        not failure['actual_group_RAM_preserved'] or identity.same_owner(old_owner,identity.process(old_owner['pid'])) or
        not identity.same_owner(captured,current) or 'T' not in current['state'] or current['pgid']!=current['pid'] or
        str(u.ROOT/'scripts/upstream_own391_tree_source.py')+' declare' not in current['command'] or
        restored['state']!='complete_prior_cache_policy_and_metadata_restoration' or
        u.read(u.ROOT/'experiments/upstream-read-cache-policy-owner.json')['state']!='restored' or
        (u.ROOT/'experiments/upstream-own391-cpu-tree').exists() or (u.ROOT/'artifacts/upstream-own391-cpu-tree').exists()):
        raise ValueError('Exact suspended prefit declaration and completed cache rollback required')
    frozen=dict(u.read(original.REPORT/'plan.json')['source_sha256'])
    files=[Path(__file__),prior/'failure.json',prior/'plan.json',prior/'monitor.json',prior/'cache-policy-restoration.json']
    for p in files:
        key=str(p.relative_to(u.ROOT));digest=u.sha(p)
        if key in frozen and frozen[key]!=digest:raise ValueError('Original source expectation conflicts')
        frozen[key]=digest
    REPORT.mkdir();child=None;watch=None;primary=None;u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    u.write(REPORT/'child-ownership.json',captured)
    u.write(REPORT/'plan.json',dict(source_sha256=frozen,original_child=captured,created=original.stamp(),
        rule='Real-input cache trial didnotavoidcompressorguard; all33priorpoliciesrestored. No morecachepolicychanges. Wait independently complete original1600 and primary391H128, thenoriginal8GiB/3stablecomp<3 admission. Continue SAME declarationinRAM, no restart. Fullunchanged CPUtreebudgets/actualmodel/operator/canonicalscores replay. Keep8group/24research/5compress/128swap/18.75disk. Scheduling supplement: only AFTER both priority prerequisites complete, acquireprimarylane nonblockingly aftermemory admission andretainthroughfullindependentreplay, preventing22GiBnative1200 fromstartingbetweenCPUfit/replayRAM phases. NoGPUlock/peer signals/automaticretries.'))
    def check():
        for rel,digest in frozen.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original deferred source changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('Deferred CPU monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    baseline=None
    def capacity(phase):
        good=0
        while True:
            check();workers=resources.footprints(os.getpid());current=memory.memory()
            good=good+1 if original.safe(workers,current,baseline,8*2**30) and current['compressor_bytes']<3*2**30 else 0
            if watch.poll() is not None:raise RuntimeError('Deferred watchdog unavailable')
            if good>=3:return
            u.write(REPORT/'status.json',dict(state='waiting_original_stable_CPU_headroom',next_phase=phase,
                good_samples=good,workers=workers,memory=current,observed=original.stamp()));time.sleep(10)
    def monitor(phase):
        check();workers=resources.footprints(captured['pid']);current=memory.memory()
        groups={s.split()[0]:int(s.split()[1]) for s in subprocess.check_output(['ps','-axo','pid=,pgid='],text=True).splitlines() if len(s.split())==2}
        own=dict(bytes=sum(w['bytes'] for pid,w in workers.items() if groups.get(pid)==captured['pgid']),rss_bytes=sum(w['rss_bytes'] for pid,w in workers.items() if groups.get(pid)==captured['pgid']))
        if not original.safe(workers,current,baseline) or max(own['bytes'],own['rss_bytes'])>8*2**30 or watch.poll() is not None:
            raise RuntimeError('Deferred CPU guard; same full group RAM preserved')
        u.write(REPORT/'status.json',dict(state=phase,pid=captured['pid'],group_memory=own,workers=workers,memory=current,observed=original.stamp()))
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            prerequisites=[(u.ROOT/'experiments/upstream-high-capacity-successor/completion.json','complete_original1600_and_independent_full_output_verification'),
                (u.ROOT/'experiments/upstream-own391-paired-h128-queue/completion.json','complete_matched_B391_neural_arm_actual_fits_replay_and_canonical_scores')]
            while not all(p.exists() for p,_ in prerequisites):
                check();current=identity.process(captured['pid'])
                if not identity.same_owner(captured,current) or 'T' not in current['state'] or watch.poll() is not None:
                    raise ValueError('Same stopped declaration/watchdog required while waiting')
                u.write(REPORT/'status.json',dict(state='waiting_original1600_and_primary391_memory_release',observed=original.stamp()));time.sleep(10)
            for p,state in prerequisites:
                if u.read(p)['state']!=state:raise ValueError('Both complete independent prerequisite proofs required')
                frozen[str(p.relative_to(u.ROOT))]=u.sha(p)
            baseline=memory.memory()
            primary=(u.ROOT/'experiments/full-upstream-reproduction/runner.lock').open('a')
            while True:
                capacity('resume_same_declaration_then_nonblocking_completion_lane')
                try:
                    fcntl.flock(primary,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if original.safe(resources.footprints(os.getpid()),memory.memory(),baseline,8*2**30) and memory.memory()['compressor_bytes']<3*2**30:break
                    fcntl.flock(primary,fcntl.LOCK_UN)
                except BlockingIOError:pass
                time.sleep(10)
            current=identity.process(captured['pid'])
            if not identity.same_owner(captured,current) or 'T' not in current['state']:raise ValueError('Exact stopped declaration required')
            os.killpg(captured['pid'],signal.SIGCONT);u.write(REPORT/'continuation.json',dict(same_actual_process=captured,restarted=False,observed=original.stamp()))
            while identity.same_owner(captured,identity.process(captured['pid'])):monitor('same_original_declaration_continued');time.sleep(3)
            manifest=u.ROOT/'experiments/upstream-own391-cpu-tree/manifest.json'
            if not manifest.exists() or u.read(manifest)['state']!='declared_full_budget_matched391_cpu_tree':raise ValueError('Full declaration required before fitting')
            u.write(REPORT/'adopted-declaration-terminal.json',dict(manifest_sha256=u.sha(manifest),exit_status_available=False,independent_full_verification_required=True))
            for phase,script in [('full_nested_fit','upstream_own391_tree_worker.py'),('independent_replay','verify_upstream_own391_cpu_tree.py')]:
                capacity(phase);path=u.ROOT/'scripts'/script
                with (REPORT/(phase+'.log')).open('x') as log:
                    captured=None;child=subprocess.Popen([sys.executable,'-u',str(path)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    captured=identity.process(child.pid)
                    if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                        raise ValueError('Dedicated actual CPU child required')
                    u.write(REPORT/'child-ownership.json',captured)
                    while child.poll() is None:monitor(phase);time.sleep(3)
                    if child.returncode:raise RuntimeError('Deferred phase failed without retry: '+phase)
                    child=None
            proof=u.ROOT/'experiments/upstream-own391-cpu-tree/independent-verification.json'
            if u.read(proof)['state']!='complete_independent_matched391_CPU_tree_models_inputs_ridge_predictions_and_canonical_scores':raise ValueError('Full independent CPU tree proof required')
            u.write(REPORT/'completion.json',dict(state='complete_deferred_same_declaration_CPU_tree_and_independent_replay',verification_sha256=u.sha(proof),qualifying_labels_read=False,observed=original.stamp()))
            u.write(REPORT/'status.json',dict(state='complete',observed=original.stamp()));u.write(REPORT/'owner-release.json',dict(state='all_owned_workers_terminal_after_full_independent_proof'))
    except BaseException as error:
        if child is not None and child.poll() is None:
            if captured is None or identity.same_owner(captured,identity.process(child.pid)):os.killpg(child.pid,signal.SIGSTOP)
        elif captured and identity.same_owner(captured,identity.process(captured['pid'])):os.killpg(captured['pid'],signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_group_RAM_preserved=True,no_automatic_retry=True,observed=original.stamp()));raise
    finally:
        if primary is not None:primary.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args();watchdog(args.watch) if args.watch else main()
