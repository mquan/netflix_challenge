#!/usr/bin/env python3
"""Continue the preserved own14 audit only after full original native completion."""
import argparse
import fcntl
import hashlib
import json
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

REPORT=u.ROOT/'experiments/upstream-own14-matched391-post-native-audit'
OLD=u.ROOT/'experiments/upstream-own14-matched391-bounded-sidecar'
LEGACY=u.ROOT/'experiments/upstream-own14-matched391-native-deferred'
STUDY=u.ROOT/'experiments/upstream-own14-matched391-calibration'
WORKER_BYTES=2**30


def stamp():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def journal(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def native_prerequisite():
    receipt=u.ROOT/'experiments/upstream-original1200-exact-build-recovery/completion.json'
    proof=u.ROOT/'experiments/upstream-original1200/independent-verification.json'
    if not receipt.exists() or not proof.exists():return False
    completion,verified=u.read(receipt),u.read(proof)
    schema=u.read(u.ROOT/'experiments/upstream-original1200/execution-plan.json')['exact_expected_outputs']
    prefix='artifacts/upstream-original1200/source/preds_new/'
    expected={prefix+row['name'] for row in schema}|{prefix+'tsvdx5-1200lm.out'}
    captured=u.read(OLD/'native-ownership.json')
    return (completion['state']=='complete_original1200_full23_23_and_independent58_output_verification' and
        completion['proof_sha256']==u.sha(proof) and
        verified['state']=='complete_independent_original1200_both23_phases_and_all58_numeric_outputs' and
        verified['epochs_each']==23 and verified['factors']==1200 and verified['neighbors']==100 and
        len(schema)==58 and len(expected)==59 and set(verified['outputs'])==expected and
        verified['qualifying_labels_read'] is False and
        not identity.same_owner(captured,identity.process(captured['pid'])))


def command(phase):
    action={'audit':'audit','separate_canonical':'canonical'}[phase]
    return [sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_own14_matched391_calibration.py'),action]


def same_child(captured,actual):
    return bool(actual and actual['pid']==captured['pid'] and actual['started']==captured['started'] and
        actual['pgid']==captured['pid'] and actual['command'] in [captured['command'],captured.get('worker_command')])


def stop_child(captured):
    actual=identity.process(captured['pid'])
    if not same_child(captured,actual):return False
    if not any(path in actual['command'].split() for path in [
        str(u.ROOT/'scripts/verify_upstream_own14_matched391_calibration.py'),str(Path(__file__).resolve())]):
        raise ValueError('Only the exact owned verification child may be signalled')
    os.killpg(captured['pid'],signal.SIGSTOP);return True


def bootstrap(phase,owner_pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Exact sidecar owner required')
    for name,digest in u.read(REPORT/'plan.json')['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen original sidecar source changed')
    captured=identity.process(os.getpid());captured['worker_command']=' '.join(command(phase))
    if captured['pgid']!=captured['pid']:raise ValueError('Dedicated audit bootstrap required')
    journal(REPORT/(phase+'-ownership.json'),captured);journal(REPORT/'child-ownership.json',captured)
    while True:
        if not identity.same_owner(owner,identity.process(owner_pid)):
            os.killpg(captured['pid'],signal.SIGSTOP);continue
        gate=REPORT/(phase+'-gate.json')
        if gate.exists():
            record=u.read(gate)
            if (record['pid']!=captured['pid'] or record['command']!=command(phase) or
                record['owner_sha256']!=u.sha(REPORT/'monitor.json') or
                not identity.same_owner(record['watchdog'],identity.process(record['watchdog']['pid'])) or
                not native_prerequisite()):
                raise ValueError('Exact live owner/watchdog/full-native gate required')
            u.write(REPORT/(phase+'-dispatch.json'),dict(pid=captured['pid'],command=command(phase),observed=stamp()))
            os.execvpe(sys.executable,command(phase),os.environ.copy())
        time.sleep(.1)


def safe(workers,current,baseline,reserve=0,disk_reserve=0):
    return (sum(w['bytes'] for w in workers.values())+reserve<20*2**30 and
        sum(w['rss_bytes'] for w in workers.values())+reserve<20*2**30 and current['compressor_bytes']<5*2**30 and
        current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and
        shutil.disk_usage(u.ROOT).free>=18.75*2**30+disk_reserve)


def child_safe(workers,current,baseline,pid):
    own=workers.get(str(pid))
    return bool(own and max(own['bytes'],own['rss_bytes'])<=WORKER_BYTES and safe(workers,current,baseline) and native_prerequisite())


def watchdog(pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=pid:raise ValueError('Exact own14 monitor required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(pid)):
            if (REPORT/'owner-release.json').exists():return
            p=REPORT/'child-ownership.json';captured=u.read(p) if p.exists() else None;stopped=False
            if captured and same_child(captured,identity.process(captured['pid'])):
                stopped=stop_child(captured)
            u.write(REPORT/'watchdog-recovery.json',dict(child=captured,actual_child_RAM_preserved=stopped,observed=stamp()));return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing own14 sidecar evidence preserved')
    failure=u.read(OLD/'failure.json');adopted=failure['child'];prior_owner=u.read(OLD/'monitor.json')
    prior_watch=u.read(OLD/'audit-gate.json')['watchdog'];recovery=u.read(OLD/'watchdog-recovery.json')
    actual=identity.process(adopted['pid'])
    pre=STUDY/'independent-preaudit-verification.json';verified=u.read(pre)
    if (failure['error']!="RuntimeError('Own14 original resource guard; actual child RAM preserved')" or
        not failure['actual_child_RAM_preserved'] or not failure['no_automatic_retry'] or
        identity.same_owner(prior_owner,identity.process(prior_owner['pid'])) or
        identity.same_owner(prior_watch,identity.process(prior_watch['pid'])) or
        recovery['child']!=adopted or recovery['actual_child_RAM_preserved'] is not True or
        not same_child(adopted,actual) or 'T' not in actual['state'] or
        not actual['command'].endswith('/scripts/verify_upstream_own14_matched391_calibration.py audit')):
        raise ValueError('Only exact stopped original audit after preserved sidecar guard may be adopted')
    if (verified['state']!='complete_independent_all16383_supports_ridge15_and_six_vectors_before_audit' or
        verified['freeze_sha256']!=u.sha(STUDY/'frozen.json') or verified['qualifying_labels_read'] is not False or
        verified['simplex_supports_examined']!=16383 or verified['ridge_equations']!=15 or
        not verified['all_six_candidate_vectors_bitwise'] or
        (STUDY/'independent-recovered-audit.json').exists() or (STUDY/'independent-canonical-verification.json').exists()):
        raise ValueError('Full original label-free preaudit and unfinished SAME audit required')
    artifacts=u.ROOT/'artifacts/upstream-own14-matched391-calibration'
    expected={'known-simplex.f32','known-ridge10.f32','known-fixed-half.f32','own14-simplex.f32','own14-ridge10.f32','matched391-fixed-half.f32'}
    if {p.name for p in artifacts.iterdir()}!=expected:raise ValueError('Only original six frozen vectors may exist')
    for name in expected:
        count=1128164 if name.startswith('known-') else 2817131
        if (artifacts/name).stat().st_size!=count*4:raise ValueError('All original frozen vectors must be complete')
    files=[Path(__file__),u.ROOT/'OWN14_POST_NATIVE_AUDIT.md',u.ROOT/'experiments/upstream-original1200/execution-plan.json',OLD/'plan.json',OLD/'monitor.json',OLD/'original-baseline.json',OLD/'failure.json',OLD/'guard-sample.json',OLD/'watchdog-recovery.json',OLD/'audit-gate.json',pre,
        STUDY/'plan.json',STUDY/'calibration.json',STUDY/'known-completion.json',STUDY/'frozen.json',
        *[artifacts/name for name in sorted(expected)],u.ROOT/'OWN14_MATCHED391_CALIBRATION_PLAN.md',
        u.ROOT/'scripts/upstream_own14_matched391_calibration.py',u.ROOT/'scripts/verify_upstream_own14_matched391_calibration.py',
        Path(resources.__file__),Path(memory.__file__),Path(identity.__file__),
        u.ROOT/'experiments/upstream-own391-paired-h128-mps/independent-canonical-verification.json']
    bindings=dict(u.read(OLD/'plan.json')['source_sha256'])
    for p in files:
        key=str(p.relative_to(u.ROOT));digest=u.sha(p)
        if key in bindings and bindings[key]!=digest:raise ValueError('Original frozen own14 source expectation conflicts')
        bindings[key]=digest
    for key,digest in bindings.items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Frozen original known study changed')
    REPORT.mkdir()
    parent_fd=os.open(REPORT.parent,os.O_RDONLY)
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
    child=None;captured=adopted;watch=None
    journal(REPORT/'monitor.json',identity.process(os.getpid()))
    journal(REPORT/'child-ownership.json',captured)
    journal(REPORT/'plan.json',dict(source_sha256=bindings,created=stamp(),
        adopted_child=adopted,prior_owner=prior_owner,
        rule='Scheduling-only restoration after bounded preaudit success and audit compressor guard. Preserve SAME actual audit26049 and original first swapbaseline; never restart/refit/regenerate vectors. Wait FULL original1200 both23phases/all58 numeric outputs independent proof AND nativehandle terminal. Then3stable original1GiBreserve/20GiBphysicalRSS/5comp(admit4.5)/128swap/18.75disk+64MiB gates. No lanes/native signals, no repeated overlap. Resume actual audit; afterward gated original separatecanonical. Full original scientific proof before transparent append-only oldestlegacy completion bridge naming successor and fullnative prerequisite. Stop/preserve owned actual child on any guard or ownerloss.'))
    baseline=u.read(OLD/'original-baseline.json')['memory']
    baseline=u.read(OLD/'original-baseline.json')['memory']
    journal(REPORT/'original-baseline.json',dict(memory=baseline,original_baseline_sha256=u.sha(OLD/'original-baseline.json')))
    bindings[str((REPORT/'original-baseline.json').relative_to(u.ROOT))]=u.sha(REPORT/'original-baseline.json')
    plan=u.read(REPORT/'plan.json');plan['source_sha256']=bindings;journal(REPORT/'plan.json',plan)
    def check():
        for rel,digest in bindings.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen own14 numerical/execution plan changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('Own14 monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def capacity(phase,adopt=False):
        good=0
        while True:
            if not native_prerequisite():raise RuntimeError('Full native completion proof unavailable; preserve same worker')
            check();workers=resources.footprints(os.getpid());current=memory.memory()
            account={k:dict(v) for k,v in workers.items()};reserve=WORKER_BYTES
            if adopt:
                own=account.get(str(adopted['pid']))
                if own is None or max(own['bytes'],own['rss_bytes'])>WORKER_BYTES:
                    raise ValueError('Actual preserved worker must fit original1GiB cap')
                own['bytes']=max(own['bytes'],WORKER_BYTES);own['rss_bytes']=max(own['rss_bytes'],WORKER_BYTES);reserve=0
            good=good+1 if safe(account,current,baseline,reserve,64*2**20) and current['compressor_bytes']<4.5*2**30 and native_prerequisite() else 0
            if watch.poll() is not None:raise RuntimeError('Own14 watchdog unavailable')
            if good>=3:return
            u.write(REPORT/'status.json',dict(state='waiting_one_GiB_and_original_compressor_headroom',next_phase=phase,good_samples=good,
                workers=workers,memory=current,observed=stamp()));time.sleep(10)
    def run(phase,script,arg):
        nonlocal child,captured
        capacity(phase);path=u.ROOT/'scripts'/script
        with (REPORT/(phase+'.log')).open('x') as log:
            captured=None;child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--bootstrap',str(os.getpid()),'--phase',phase],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ownership=REPORT/(phase+'-ownership.json')
            while not ownership.exists():
                if child.poll() is not None or watch.poll() is not None:raise RuntimeError('Live bounded audit bootstrap required')
                time.sleep(.1)
            captured=u.read(ownership)
            if captured['pid']!=child.pid or not same_child(captured,identity.process(child.pid)):raise ValueError('Exact audit bootstrap required')
            watcher=identity.process(watch.pid)
            if not watcher or watch.poll() is not None:raise RuntimeError('Live watchdog required before dispatch')
            journal(REPORT/(phase+'-gate.json'),dict(pid=child.pid,command=command(phase),owner_sha256=u.sha(REPORT/'monitor.json'),watchdog=watcher))
            while child.poll() is None:
                check();workers=resources.footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid))
                if child.poll() is not None:break
                if not child_safe(workers,current,baseline,child.pid) or watch.poll() is not None:
                    u.write(REPORT/'guard-sample.json',dict(workers=workers,memory=current,phase=phase))
                    raise RuntimeError('Own14 original resource guard; actual child RAM preserved')
                u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,observed=stamp()));time.sleep(3)
            if child.returncode:raise RuntimeError('Own14 phase failed without retry: '+phase)
            child=None;captured=None
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            h512=u.ROOT/'experiments/upstream-own391-paired-h512-queue/completion.json'
            while not h512.exists():
                check()
                if watch.poll() is not None:raise RuntimeError('Own14 watchdog lost')
                u.write(REPORT/'status.json',dict(state='waiting_full_H512_known_verification_before_fixed_architecture',observed=stamp()));time.sleep(10)
            hp=u.read(h512);cp=u.ROOT/'experiments/upstream-own391-paired-h512-mps/independent-canonical-verification.json'
            if hp['state']!='complete_matched_B391_neural_arm_actual_fits_replay_and_canonical_scores' or hp['canonical_verification_sha256']!=u.sha(cp):
                raise ValueError('Full original H512 canonical completion required')
            for path in [h512,cp]:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
            receipt=u.ROOT/'experiments/upstream-original1200-exact-build-recovery/completion.json'
            proof=u.ROOT/'experiments/upstream-original1200/independent-verification.json'
            while not receipt.exists():
                check();actual=identity.process(adopted['pid'])
                if not same_child(adopted,actual) or 'T' not in actual['state'] or watch.poll() is not None:
                    raise ValueError('Same stopped audit and live waiting watchdog required')
                u.write(REPORT/'status.json',dict(state='waiting_full_original1200_before_same_audit_continuation',same_actual_audit=adopted,observed=stamp()))
                time.sleep(10)
            if not native_prerequisite():raise ValueError('Full independently verified native23+23/all58 proof and terminal handle required')
            for path in [receipt,proof]:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
            plan=u.read(REPORT/'plan.json');plan['source_sha256']=bindings;journal(REPORT/'plan.json',plan)
            capacity('same_original_frozen_audit_continuation',True)
            current=identity.process(adopted['pid'])
            if not same_child(adopted,current) or 'T' not in current['state'] or watch.poll() is not None:
                raise ValueError('Exact stopped worker and watchdog required immediately before continuation')
            journal(REPORT/'continuation-intent.json',dict(same_actual_process=adopted,restarted=False,coefficients_refitted=False,observed=stamp()))
            os.killpg(adopted['pid'],signal.SIGCONT)
            journal(REPORT/'continuation.json',dict(same_actual_process=adopted,restarted=False,coefficients_refitted=False,observed=stamp()))
            while same_child(adopted,identity.process(adopted['pid'])):
                check();workers=resources.footprints(adopted['pid']);current=memory.memory()
                if not same_child(adopted,identity.process(adopted['pid'])):break
                if not child_safe(workers,current,baseline,adopted['pid']) or watch.poll() is not None:
                    u.write(REPORT/'guard-sample.json',dict(workers=workers,memory=current,phase='same_known_continued'))
                    raise RuntimeError('Exact known continuation resource guard; same actual RAM preserved')
                u.write(REPORT/'status.json',dict(state='same_original_frozen_audit_worker_continued',workers=workers,memory=current,observed=stamp()));time.sleep(3)
            done=u.read(STUDY/'independent-recovered-audit.json')
            if (done['state']!='complete_all_three_original_own14_frozen_candidate_audits' or
                done['preaudit_verification_sha256']!=u.sha(pre) or
                done['qualifying_labels_used_for_fitting_or_selection'] is not False):
                raise ValueError('Original frozen audit terminal without full original proof; no restart')
            journal(REPORT/'adopted-audit-terminal.json',dict(audit_sha256=u.sha(STUDY/'independent-recovered-audit.json'),
                exit_status_available=False,full_separate_canonical_verification_remaining=True))
            captured=None
            for phase,script,arg in [('separate_canonical','verify_upstream_own14_matched391_calibration.py','canonical')]:run(phase,script,arg)
            proof=STUDY/'independent-canonical-verification.json'
            if u.read(proof)['state']!='complete_independent_own14_full_source_models_coefficients_and_frozen_group_scores':
                raise ValueError('Full independent own14 canonical evidence required')
            journal(REPORT/'completion.json',dict(state='complete_own14_known_only_freeze_preaudit_and_separate_frozen_canonical_audit',
                verification_sha256=u.sha(proof),observed=stamp()))
            if (LEGACY/'completion.json').exists():raise ValueError('Existing legacy completion must not be overwritten')
            journal(LEGACY/'completion.json',dict(state='complete_own14_known_only_freeze_preaudit_and_separate_frozen_canonical_audit',
                verification_sha256=u.sha(proof),successor_completion=str((REPORT/'completion.json').relative_to(u.ROOT)),
                successor_completion_sha256=u.sha(REPORT/'completion.json'),
                actual_owner=u.read(REPORT/'monitor.json'),full_native_completion_verified_before_audit_continuation=True,
                original_native_process_signalled=False,observed=stamp()))
            u.write(REPORT/'status.json',dict(state='complete',observed=stamp()));u.write(REPORT/'owner-release.json',dict(state='all_owned_children_terminal_after_full_independent_proof'))
    except BaseException as error:
        stopped=False
        if child is not None and child.poll() is None:
            if captured is None or same_child(captured,identity.process(child.pid)):
                if captured is None:
                    os.killpg(child.pid,signal.SIGSTOP);stopped=True
                else:stopped=stop_child(captured)
        if child is None and captured and same_child(captured,identity.process(captured['pid'])):
            stopped=stop_child(captured)
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=stopped,no_automatic_retry=True,observed=stamp()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);parser.add_argument('--bootstrap',type=int)
    parser.add_argument('--phase',choices=['separate_canonical']);args=parser.parse_args()
    if args.watch:watchdog(args.watch)
    elif args.bootstrap:bootstrap(args.phase,args.bootstrap)
    else:main()
