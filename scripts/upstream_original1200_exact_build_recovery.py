#!/usr/bin/env python3
"""Continue the exact paused compiler group, then original full23/23 producer."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_original1200_queue as original

u,experiment,resources,memory,identity=original.u,original.experiment,original.resources,original.memory,original.identity
REPORT=u.ROOT/'experiments/upstream-original1200-exact-build-recovery'
DECL=u.ROOT/'ORIGINAL1200_EXACT_BUILD_RECOVERY.md'


def members(pgid):
    result=[]
    for row in subprocess.check_output(['ps','-axo','pid=,pgid=,stat='],text=True).splitlines():
        pid,group,state=row.split()
        if int(group)==pgid and not state.startswith('Z'):result.append((int(pid),state))
    return result


def preserve(captured,known):
    if not captured or not members(captured['pgid']):return False
    if not any(identity.same_owner(p,identity.process(p['pid'])) for p in known):return False
    os.killpg(captured['pgid'],signal.SIGSTOP);return True


def build_admitted(workers,group,current,baseline):
    own={key:sum(w[key] for pid,w in workers.items() if pid in group) for key in ['bytes','rss_bytes']}
    if not group or max(own.values())>3*2**30:return False
    reserved={pid:dict(w) for pid,w in workers.items() if pid not in group}
    reserved['preserved_build_reservation']=dict(bytes=3*2**30,rss_bytes=3*2**30)
    return original.safe(reserved,current,baseline,disk_floor=20) and current['compressor_bytes']<4.5*2**30


def watchdog(pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=pid:raise ValueError('Exact original1200 recovery owner required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(pid)):
            if (REPORT/'owner-release.json').exists():return
            child=u.read(REPORT/'child-ownership.json');known=u.read(REPORT/'known-members.json')['members']
            stopped=preserve(child,known)
            u.write(REPORT/'watchdog-recovery.json',dict(child=child,actual_group_RAM_preserved=stopped,observed=original.stamp()));return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing exact compiler/native recovery retained')
    old=original.REPORT;failure=u.read(old/'failure.json');captured=failure['child'];actual=identity.process(captured['pid'])
    owner=u.read(old/'monitor.json')
    if (failure['error']!="RuntimeError('Original1200 resource/watchdog guard; actual group RAM preserved')" or
        not failure['actual_child_RAM_preserved'] or not failure['no_automatic_retry'] or
        identity.same_owner(owner,identity.process(owner['pid'])) or
        not identity.same_owner(captured,actual) or 'T' not in actual['state'] or captured['pgid']!=captured['pid'] or
        str(u.ROOT/'scripts/upstream_original1200.py')+' build' not in captured['command'] or
        (experiment.REPORT/'original-build-receipt.json').exists() or (old/'completion.json').exists()):
        raise ValueError('Only exact stopped original compiler group before build receipt may continue')
    initial=members(captured['pgid'])
    if not initial or any('T' not in state for _,state in initial):raise ValueError('Entire original compiler group must be stopped')
    known=[identity.process(pid) for pid,_ in initial]
    if not all(known):raise ValueError('Every actual compiler group identity required')
    bindings=dict(u.read(old/'plan.json')['source_sha256'])
    for path in [Path(__file__),DECL,old/'failure.json',old/'monitor.json',old/'plan.json',old/'status.json',experiment.REPORT/'execution-plan.json']:
        key=str(path.relative_to(u.ROOT));digest=u.sha(path)
        if key in bindings and bindings[key]!=digest:raise ValueError('Conflicting original source expectation')
        bindings[key]=digest
    for rel,digest in bindings.items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original source/preparation changed: '+rel)
    baseline=u.read(old/'status.json')['memory']
    REPORT.mkdir();child=None;watch=None;lanes=[]
    u.write(REPORT/'monitor.json',identity.process(os.getpid()));u.write(REPORT/'child-ownership.json',captured)
    u.write(REPORT/'known-members.json',dict(members=known))
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,original_build_group=captured,original_members=known,
        original_baseline=baseline,created=original.stamp(),
        rule='Wait complete H512 actual-model/canonical proof and original eligible H128 bag completion without labels. Serialize joint GPU+primary through SAME compiler/admission/full23+23native; no compiler restart. Credit existing group RAM toward original3GiB reservation; continuing already-launched build keeps original20GiB floor (originalfresh20.5 gate alreadypassed). Original3group/24research/5compress/128swap build/admission guards. Exact compiled binary/data admission before native. Native reserve22/24research,3stablecomp<4.5,20.125GiB initial; original4Rayon/1BLAS/full1200factors100neighbors23each/all58outputs. Originalnative22/24/swap128/disk18.75 checks unchanged; native swap baseline established at FIRSTnative start asoriginal, no midfit reset. Release lanes only after native terminal, full independent58output verification. Preserve actual owned group on guard/owner loss, no retries or numerical changes.'))
    def check():
        for rel,digest in bindings.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen exact compiler/native recovery changed: '+rel)
    def interrupted(signum,frame):raise KeyboardInterrupt('Exact original1200 recovery interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def capture(pid):
        nonlocal captured,known
        captured=identity.process(pid)
        if not captured or captured['pgid']!=pid:raise ValueError('Dedicated actual stage identity required')
        known=[captured];u.write(REPORT/'child-ownership.json',captured);u.write(REPORT/'known-members.json',dict(members=known))
    def snapshot():
        workers=resources.footprints(captured['pid']);group={str(pid) for pid,_ in members(captured['pgid'])}
        for pid in group-workers.keys():
            if identity.process(int(pid)):workers.update(resources.footprints(int(pid)))
        own={key:sum(w[key] for pid,w in workers.items() if pid in group) for key in ['bytes','rss_bytes']}
        return workers,own
    def capacity(phase,reserve,disk_floor,adopt=False,base=None):
        good=0
        while True:
            check();workers,own=snapshot();current=memory.memory();account={k:dict(v) for k,v in workers.items()}
            if adopt:
                ready=build_admitted(workers,{str(p) for p,_ in members(captured['pgid'])},current,base)
            else:ready=original.safe(account,current,base,reserve,disk_floor) and current['compressor_bytes']<4.5*2**30
            good=good+1 if ready else 0
            if watch.poll() is not None:raise RuntimeError('Exact original1200 watchdog unavailable')
            if good>=3:return
            u.write(REPORT/'status.json',dict(state='waiting_original_capacity',next_phase=phase,good_samples=good,workers=workers,
                group_memory=own,memory=current,observed=original.stamp()));time.sleep(10)
    def monitor(phase,base,disk_floor,cap):
        check();workers,own=snapshot();current=memory.memory()
        for pid,_ in members(captured['pgid']):
            if pid not in {p['pid'] for p in known}:
                p=identity.process(pid)
                if p:known.append(p);u.write(REPORT/'known-members.json',dict(members=known))
        if (not original.safe(workers,current,base,disk_floor=disk_floor) or max(own.values())>cap or
            (phase!='native' and current['compressor_bytes']>=5*2**30) or watch.poll() is not None):
            u.write(REPORT/'guard-sample.json',dict(workers=workers,group_memory=own,memory=current,phase=phase))
            raise RuntimeError('Exact original1200 guard; same actual owned group RAM preserved')
        u.write(REPORT/'status.json',dict(state=phase,pid=captured['pid'],workers=workers,group_memory=own,memory=current,observed=original.stamp()))
    def run_owned(command,cwd,env,phase,log_path,base,disk_floor,cap):
        nonlocal child,captured,known
        check()
        with log_path.open('x') as log:
            captured=None;known=[]
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            capture(child.pid)
            if str(command[0]) not in captured['command']:raise ValueError('Actual declared stage command required')
            while child.poll() is None:monitor(phase,base,disk_floor,cap);time.sleep(2)
            if child.returncode:raise RuntimeError('Exact original1200 phase failed without retry: '+phase)
            if members(captured['pgid']):raise ValueError('Stage cannot complete with live owned descendants')
            child=None
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],cwd=u.ROOT,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            h512=u.ROOT/'experiments/upstream-own391-paired-h512-queue/completion.json'
            bag=u.ROOT/'experiments/upstream-own391-fold-bag-queue/completion.json'
            while not h512.exists() or not bag.exists():
                check();actual=identity.process(captured['pid'])
                if not identity.same_owner(captured,actual) or any('T' not in state for _,state in members(captured['pgid'])) or watch.poll() is not None:
                    raise ValueError('Exact stopped compiler/watchdog required while prerequisites run')
                u.write(REPORT/'status.json',dict(state='waiting_H512_verification_and_original_H128_bag_completion',observed=original.stamp()));time.sleep(10)
            hp=u.read(h512);bp=u.read(bag)
            canonical=u.ROOT/'experiments/upstream-own391-paired-h512-mps/independent-canonical-verification.json'
            if (hp['state']!='complete_matched_B391_neural_arm_actual_fits_replay_and_canonical_scores' or hp['variant']!='h512' or
                hp['canonical_verification_sha256']!=u.sha(canonical) or u.read(canonical)['qualifying_labels_read'] is not False or
                bp['state'] not in ['complete_known_eligibility_check_no_qualifying_inference','complete_known_eligible_matched391_qualifying_bag_and_independent_replay'] or
                bp['qualifying_labels_read'] is not False):raise ValueError('Full actual neural proof and original outcome-free bag completion required')
            for path in [h512,canonical,bag]:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
            lanes=[(u.ROOT/rel).open('a') for rel in ['experiments/upstream-source-extension/run.lock','experiments/full-upstream-reproduction/runner.lock']]
            while True:
                capacity('same_build_then_joint_serialized_lanes',3*2**30,20,True,baseline);acquired=[]
                try:
                    for lane in lanes:fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired.append(lane)
                    workers,own=snapshot();current=memory.memory()
                    if build_admitted(workers,{str(p) for p,_ in members(captured['pgid'])},current,baseline):break
                except BlockingIOError:pass
                for lane in acquired:fcntl.flock(lane,fcntl.LOCK_UN)
                time.sleep(10)
            actual=identity.process(captured['pid'])
            if not identity.same_owner(captured,actual) or any('T' not in state for _,state in members(captured['pgid'])):raise ValueError('Same entire stopped compiler group required')
            os.killpg(captured['pgid'],signal.SIGCONT)
            u.write(REPORT/'build-continuation.json',dict(same_actual_group=captured,restarted=False,observed=original.stamp()))
            while members(captured['pgid']):monitor('same_original_build_continued',baseline,20,3*2**30);time.sleep(2)
            build=experiment.REPORT/'original-build-receipt.json'
            if not build.exists() or u.read(build)['preparation_sha256']!=u.sha(experiment.REPORT/'execution-plan.json'):
                raise ValueError('Adopted compiler terminal without original full build receipt; no restart')
            bindings[str(build.relative_to(u.ROOT))]=u.sha(build)
            u.write(REPORT/'adopted-build-terminal.json',dict(build_receipt_sha256=u.sha(build),all_original_group_members_terminal=True,
                exit_status_available=False,full_independent_admission_required=True))
            capacity('independent_admission',3*2**30,20,base=baseline)
            run_owned([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_original1200.py')],u.ROOT,None,'admission',REPORT/'admission.log',baseline,20,3*2**30)
            admission=experiment.REPORT/'independent-admission.json';bindings[str(admission.relative_to(u.ROOT))]=u.sha(admission)
            capacity('native',22*2**30,20.125,base=baseline)
            env=u.environment();threads={'RAYON_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'};env.update(threads)
            command=u.read(admission)['native_command'];started=original.stamp();native_baseline=memory.memory()
            u.write(REPORT/'first-native-baseline.json',dict(memory=native_baseline,observed=started,original_policy_first_native_start=True))
            run_owned(command,experiment.TREE,env,'native',experiment.REPORT/'native.log',native_baseline,18.75,22*2**30)
            u.write(experiment.REPORT/'native-run.json',dict(command=command,cwd=str(experiment.TREE),returncode=0,started=started,
                finished=original.stamp(),environment=threads,qualifying_labels_read=False))
            for lane in lanes:fcntl.flock(lane,fcntl.LOCK_UN)
            capacity('independent_complete',3*2**30,18.75,base=native_baseline)
            run_owned([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_original1200.py'),'--complete'],u.ROOT,None,
                'independent_complete',REPORT/'independent-complete.log',native_baseline,18.75,3*2**30)
            proof=experiment.REPORT/'independent-verification.json'
            if u.read(proof)['state']!='complete_independent_original1200_both23_phases_and_all58_numeric_outputs':raise ValueError('Full original23/23 and58-output proof required')
            completion=dict(state='complete_original1200_full23_23_and_independent58_output_verification',proof_sha256=u.sha(proof),
                observed=original.stamp(),qualifying_labels_read=False,no_score_promotion=True,exact_build_recovery_plan_sha256=u.sha(REPORT/'plan.json'))
            u.write(REPORT/'completion.json',completion)
            if (old/'completion.json').exists():raise ValueError('Original completion cannot be overwritten')
            u.write(old/'completion.json',completion)
            u.write(REPORT/'status.json',dict(state='complete',observed=original.stamp()));u.write(REPORT/'owner-release.json',dict(state='all_owned_original_stages_terminal_after_full_independent_proof'))
    except BaseException as error:
        stopped=preserve(captured,known)
        direct_pid=None
        if not stopped and child is not None and child.poll() is None:
            # An unreaped direct child cannot have its PID reused; setsid is
            # complete before Popen returns. Preserve even if capture failed.
            os.killpg(child.pid,signal.SIGSTOP);stopped=True;direct_pid=child.pid
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,direct_child_fallback_pid=direct_pid,
            actual_group_RAM_preserved=stopped,no_automatic_retry=True,observed=original.stamp()));raise
    finally:
        for lane in lanes:lane.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args()
    watchdog(args.watch) if args.watch else main()
