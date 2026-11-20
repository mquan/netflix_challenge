#!/usr/bin/env python3
"""Guard the bounded own14 CPU comparison without taking compute lanes."""
import argparse
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

REPORT=u.ROOT/'experiments/upstream-own14-matched391-native-deferred'
OLD=u.ROOT/'experiments/upstream-own14-matched391-preaudit-continuation'
STUDY=u.ROOT/'experiments/upstream-own14-matched391-calibration'
WORKER_BYTES=2**30


def stamp():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers,current,baseline,reserve=0,disk_reserve=0):
    return (sum(w['bytes'] for w in workers.values())+reserve<24*2**30 and
        sum(w['rss_bytes'] for w in workers.values())+reserve<24*2**30 and current['compressor_bytes']<5*2**30 and
        current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and
        shutil.disk_usage(u.ROOT).free>=18.75*2**30+disk_reserve)


def child_safe(workers,current,baseline,pid):
    own=workers.get(str(pid))
    return bool(own and max(own['bytes'],own['rss_bytes'])<=WORKER_BYTES and safe(workers,current,baseline))


def watchdog(pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=pid:raise ValueError('Exact own14 monitor required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(pid)):
            if (REPORT/'owner-release.json').exists():return
            p=REPORT/'child-ownership.json';captured=u.read(p) if p.exists() else None;stopped=False
            if captured and identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            u.write(REPORT/'watchdog-recovery.json',dict(child=captured,actual_child_RAM_preserved=stopped,observed=stamp()));return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing own14 queue preserved')
    failure=u.read(OLD/'failure.json');adopted=failure['child'];prior_owner=u.read(OLD/'monitor.json')
    actual=identity.process(adopted['pid'])
    if (failure['error']!="RuntimeError('Exact known continuation resource guard; same actual RAM preserved')" or
        not failure['actual_child_RAM_preserved'] or not failure['no_automatic_retry'] or
        identity.same_owner(prior_owner,identity.process(prior_owner['pid'])) or
        not identity.same_owner(adopted,actual) or 'T' not in actual['state'] or actual['pgid']!=actual['pid'] or
        str(u.ROOT/'scripts/verify_upstream_own14_matched391_calibration.py')+' preaudit' not in actual['command'] or
        (STUDY/'independent-preaudit-verification.json').exists() or
        {p.name for p in STUDY.iterdir()}!={'plan.json','calibration.json','known-completion.json','frozen.json'}):
        raise ValueError('Exact paused original preaudit and complete frozen six vectors required')
    artifacts=u.ROOT/'artifacts/upstream-own14-matched391-calibration'
    expected={'known-simplex.f32','known-ridge10.f32','known-fixed-half.f32','own14-simplex.f32','own14-ridge10.f32','matched391-fixed-half.f32'}
    if {p.name for p in artifacts.iterdir()}!=expected:raise ValueError('Only original six frozen vectors may exist')
    for name in expected:
        count=1128164 if name.startswith('known-') else 2817131
        if (artifacts/name).stat().st_size!=count*4:raise ValueError('All original frozen vectors must be complete')
    files=[Path(__file__),u.ROOT/'OWN14_NATIVE_DEFERRED.md',OLD/'plan.json',OLD/'failure.json',OLD/'monitor.json',OLD/'status.json',
        STUDY/'plan.json',STUDY/'calibration.json',STUDY/'known-completion.json',STUDY/'frozen.json',OLD/'original-baseline.json',*[artifacts/name for name in sorted(expected)],u.ROOT/'OWN14_MATCHED391_CALIBRATION_PLAN.md',
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
    REPORT.mkdir();child=None;captured=adopted;watch=None
    u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    u.write(REPORT/'child-ownership.json',captured)
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,created=stamp(),
        adopted_child=adopted,rule='Wait original1200 full independently verified23+23/58 outputs, then resume SAME original independent preaudit worker, preserving exact hash traversal and original matrices/coefficients/source/guards. No refit/retry. Wait complete H512 canonical/actual proof. Reserve1GiB and64MiBdisk for bounded full14-family float64 math; no GPU or primary lane. Originalbag/largeproducer retain their lanes. Full14matrix/candidates/math unchanged. FullinheritedsourceSHA checked with descriptor-localNOCACHE; noglobalpolicy. All16383supports/ridge15/all6vectorsbeforeanylabels, frozen3audits/separatecanonical. Own1GiB/research24physicalRSS/comp5/swap128MiB/disk18.75 guards; preservedactualchildRAM onguard/ownerloss, no retry. ApparentknownfitNOTnestedCV, no source modelrefit.'))
    baseline=u.read(OLD/'original-baseline.json')['memory']
    u.write(REPORT/'original-baseline.json',dict(memory=baseline,original_baseline_sha256=u.sha(OLD/'original-baseline.json')))
    def check():
        for rel,digest in bindings.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen own14 numerical/execution plan changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('Own14 monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def capacity(phase,adopt=False):
        good=0
        while True:
            check();workers=resources.footprints(os.getpid());current=memory.memory()
            account={k:dict(v) for k,v in workers.items()};reserve=WORKER_BYTES
            if adopt:
                own=account.get(str(adopted['pid']))
                if own is None or max(own['bytes'],own['rss_bytes'])>WORKER_BYTES:
                    raise ValueError('Actual preserved worker must fit original1GiB cap')
                own['bytes']=max(own['bytes'],WORKER_BYTES);own['rss_bytes']=max(own['rss_bytes'],WORKER_BYTES);reserve=0
            good=good+1 if safe(account,current,baseline,reserve,64*2**20) and current['compressor_bytes']<4.5*2**30 else 0
            if watch.poll() is not None:raise RuntimeError('Own14 watchdog unavailable')
            if good>=3:return
            u.write(REPORT/'status.json',dict(state='waiting_one_GiB_and_original_compressor_headroom',next_phase=phase,good_samples=good,
                workers=workers,memory=current,observed=stamp()));time.sleep(10)
    def run(phase,script,arg):
        nonlocal child,captured
        capacity(phase);path=u.ROOT/'scripts'/script
        with (REPORT/(phase+'.log')).open('x') as log:
            captured=None;child=subprocess.Popen([sys.executable,'-u',str(path),arg],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            captured=identity.process(child.pid)
            if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:raise ValueError('Exact own14 scientific child required')
            u.write(REPORT/'child-ownership.json',captured)
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
            native=u.ROOT/'experiments/upstream-original1200-exact-build-recovery/completion.json'
            while not native.exists():
                check();current=identity.process(adopted['pid'])
                if not identity.same_owner(adopted,current) or 'T' not in current['state']:
                    raise ValueError('Same preserved preaudit must remain stopped while native runs')
                if watch.poll() is not None:raise RuntimeError('Deferred preaudit watchdog lost')
                u.write(REPORT/'status.json',dict(state='waiting_full_original1200_independent_completion_before_same_preaudit',
                    suspended_child=adopted,observed=stamp()));time.sleep(10)
            completion=u.read(native)
            # The producer completion must bind its original complete58-output proof.
            import upstream_original1200 as native_experiment
            proof=native_experiment.REPORT/'independent-verification.json'
            if (completion['state']!='complete_original1200_full23_23_and_independent58_output_verification' or
                completion['proof_sha256']!=u.sha(proof) or
                u.read(proof)['state']!='complete_independent_original1200_both23_phases_and_all58_numeric_outputs'):
                raise ValueError('Original full1200 completion and independent58-output proof required')
            for p in [native,proof]:bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
            capacity('same_original_preaudit_worker_continuation',True)
            current=identity.process(adopted['pid'])
            if not identity.same_owner(adopted,current) or 'T' not in current['state'] or watch.poll() is not None:
                raise ValueError('Exact stopped worker and watchdog required immediately before continuation')
            os.killpg(adopted['pid'],signal.SIGCONT)
            u.write(REPORT/'continuation.json',dict(same_actual_process=adopted,restarted=False,coefficients_refitted=False,observed=stamp()))
            while identity.same_owner(adopted,identity.process(adopted['pid'])):
                check();workers=resources.footprints(adopted['pid']);current=memory.memory()
                if not identity.same_owner(adopted,identity.process(adopted['pid'])):break
                if not child_safe(workers,current,baseline,adopted['pid']) or watch.poll() is not None:
                    u.write(REPORT/'guard-sample.json',dict(workers=workers,memory=current,phase='same_known_continued'))
                    raise RuntimeError('Exact known continuation resource guard; same actual RAM preserved')
                u.write(REPORT/'status.json',dict(state='same_original_independent_preaudit_worker_continued',workers=workers,memory=current,observed=stamp()));time.sleep(3)
            done=u.read(STUDY/'independent-preaudit-verification.json')
            if (done['state']!='complete_independent_all16383_supports_ridge15_and_six_vectors_before_audit' or
                done['freeze_sha256']!=u.sha(STUDY/'frozen.json') or done['qualifying_labels_read'] is not False or
                done['simplex_supports_examined']!=16383 or done['ridge_equations']!=15 or
                not done['all_six_candidate_vectors_bitwise']):
                raise ValueError('Adopted preaudit terminal without complete original proof; no retry')
            u.write(REPORT/'adopted-preaudit-terminal.json',dict(preaudit_sha256=u.sha(STUDY/'independent-preaudit-verification.json'),
                exit_status_available=False,full_separate_canonical_verification_remaining=True))
            captured=None
            for phase,script,arg in [('audit','verify_upstream_own14_matched391_calibration.py','audit'),
                ('separate_canonical','verify_upstream_own14_matched391_calibration.py','canonical')]:run(phase,script,arg)
            proof=STUDY/'independent-canonical-verification.json'
            if u.read(proof)['state']!='complete_independent_own14_full_source_models_coefficients_and_frozen_group_scores':
                raise ValueError('Full independent own14 canonical evidence required')
            u.write(REPORT/'completion.json',dict(state='complete_own14_known_only_freeze_preaudit_and_separate_frozen_canonical_audit',
                verification_sha256=u.sha(proof),observed=stamp()))
            u.write(REPORT/'status.json',dict(state='complete',observed=stamp()));u.write(REPORT/'owner-release.json',dict(state='all_owned_children_terminal_after_full_independent_proof'))
    except BaseException as error:
        stopped=False
        if child is not None and child.poll() is None:
            if captured is None or identity.same_owner(captured,identity.process(child.pid)):
                os.killpg(child.pid,signal.SIGSTOP);stopped=True
        if child is None and captured and identity.same_owner(captured,identity.process(captured['pid'])):
            os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=stopped,no_automatic_retry=True,observed=stamp()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args()
    watchdog(args.watch) if args.watch else main()
