#!/usr/bin/env python3
"""Wait known-only eligibility, then guard streamed391 inference and replay."""
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
import upstream_cf96_primary_lane as identity

REPORT=u.ROOT/'experiments/upstream-own391-fold-bag-queue'
KNOWN=u.ROOT/'experiments/upstream-own391-paired-h128-mps'
MASTER=u.ROOT/'experiments/upstream-own391-paired-h128-queue'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers,current,baseline,reserve=0):
    return (sum(w['bytes'] for w in workers.values())+reserve<24*2**30 and
        sum(w['rss_bytes'] for w in workers.values())+reserve<24*2**30 and
        current['compressor_bytes']<5*2**30 and
        current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and
        shutil.disk_usage(u.ROOT).free>=18.75*2**30)


def watchdog(owner_pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Exact original bag monitor required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(owner_pid)):
            if (REPORT/'owner-release.json').exists():return
            path=REPORT/'child-ownership.json';captured=u.read(path) if path.exists() else None
            stopped=False
            if captured and identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            u.write(REPORT/'watchdog-recovery.json',dict(child=captured,actual_child_RAM_preserved=stopped,observed=stamp()))
            return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing qualifying queue preserved')
    REPORT.mkdir();child=None;captured=None;watch=None;lane=None
    files=[Path(__file__),u.ROOT/'OWN391_FOLD_BAG_PLAN.md',MASTER/'plan.json',
        u.ROOT/'scripts/upstream_own391_fold_bag.py',u.ROOT/'scripts/verify_upstream_own391_fold_bag.py',
        Path(resources.__file__),Path(memory.__file__),Path(identity.__file__)]
    frozen=dict(u.read(MASTER/'plan.json')['source_sha256'])
    for p in files:
        key=str(p.relative_to(u.ROOT));digest=u.sha(p)
        if key in frozen and frozen[key]!=digest:raise ValueError('Original master source expectation conflicts: '+key)
        frozen[key]=digest
    u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    u.write(REPORT/'plan.json',dict(source_sha256=frozen,created=stamp(),
        rule='Wait complete391H128 actual-model/statistic/canonical verification. Only a fixed-half known score strictly below its declared original352 reference permits inference. Negative known result completes eligibility check without any Q prediction. Positive: original5fold/3seedslots, including originalzero budgets;618inputs, matchedBrawqueries, stream8192rows, one complete11MiBbag. No fit or qualifying labels. Reserve16GiB before nonblockingGPU;16worker/24research/5compress/128MiBswapgrowth/18.75disk. Sameactualchild RAM preserved onguard/monitorloss, noautomaticretry. Fullindependentoperator/model/seed/fold/bag replay before release.'))
    def check():
        for rel,digest in frozen.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen qualifying/known study source changed: '+rel)
    def interrupted(signum,frame):raise KeyboardInterrupt('Matched391 qualifying monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while not (MASTER/'completion.json').exists():
                check();u.write(REPORT/'status.json',dict(state='waiting_complete_matched391_H128_known_study',observed=stamp()));time.sleep(10)
            check();master=u.read(MASTER/'completion.json');cp=KNOWN/'independent-canonical-verification.json';c=u.read(cp)
            if (master['state']!='complete_matched_B391_neural_arm_actual_fits_replay_and_canonical_scores' or
                master['canonical_verification_sha256']!=u.sha(cp) or
                c['state']!='complete_separate_matched_B391_canonical_neural_and_fixed_half_scores' or c['variant']!='h128' or
                c['qualifying_labels_read'] is not False or
                c['improves_previous']!=(c['fixed_half_tree154_rmse']<c['declared_global_reference_rmse'])):
                raise ValueError('Original complete canonical known eligibility proof required')
            for path in [MASTER/'completion.json',cp]:frozen[str(path.relative_to(u.ROOT))]=u.sha(path)
            eligible=c['improves_previous']
            u.write(REPORT/'eligibility.json',dict(eligible=eligible,canonical_proof_sha256=u.sha(cp),
                known_fixed_half_rmse=c['fixed_half_tree154_rmse'],reference_rmse=c['declared_global_reference_rmse'],
                qualifying_labels_read=False,qualifying_predictions_generated=False))
            if not eligible:
                u.write(REPORT/'completion.json',dict(state='complete_known_eligibility_check_no_qualifying_inference',
                    eligibility_sha256=u.sha(REPORT/'eligibility.json'),qualifying_labels_read=False,observed=stamp()))
                u.write(REPORT/'status.json',dict(state='known_result_not_eligible_no_qualifying_predictions',observed=stamp()))
                return
            lane=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a');baseline=memory.memory();good=0
            while True:
                check();workers=resources.footprints(os.getpid());current=memory.memory()
                admitted=(safe(workers,current,baseline,16*2**30) and current['compressor_bytes']<4.5*2**30 and
                    shutil.disk_usage(u.ROOT).free>=18.75*2**30+16*2**20)
                good=good+1 if admitted else 0
                if good>=3:
                    try:
                        fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        if safe(resources.footprints(os.getpid()),memory.memory(),baseline,16*2**30):break
                        fcntl.flock(lane,fcntl.LOCK_UN)
                    except BlockingIOError:pass
                u.write(REPORT/'status.json',dict(state='waiting_sixteen_GiB_then_nonblocking_GPU',good_samples=good,workers=workers,memory=current,observed=stamp()));time.sleep(10)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],cwd=u.ROOT,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            for phase,script in [('stream_inference','upstream_own391_fold_bag.py'),('independent_replay','verify_upstream_own391_fold_bag.py')]:
                check();path=u.ROOT/'scripts'/script
                with (REPORT/(phase+'.log')).open('x') as log:
                    captured=None
                    child=subprocess.Popen([sys.executable,'-u',str(path)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    captured=identity.process(child.pid)
                    if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                        raise ValueError('Exact original qualifying child required')
                    u.write(REPORT/'child-ownership.json',captured)
                    while child.poll() is None:
                        check();workers=resources.footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                        if not safe(workers,current,baseline) or max(own['bytes'],own['rss_bytes'])>16*2**30 or watch.poll() is not None:
                            raise RuntimeError('Qualifying resource/watchdog guard; actual child RAM preserved')
                        u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,observed=stamp()));time.sleep(3)
                    if child.returncode:raise RuntimeError('Original qualifying inference/replay failed; no retry')
                    child=None;captured=None
            proof=u.ROOT/'experiments/upstream-own391-fold-bag/independent-verification.json'
            if u.read(proof)['state']!='complete_independent_matched391_actual_fold_models_all_query_streams_and_bag':
                raise ValueError('Complete original qualifying saved-model/input/bag replay required')
            u.write(REPORT/'completion.json',dict(state='complete_known_eligible_matched391_qualifying_bag_and_independent_replay',
                verification_sha256=u.sha(proof),qualifying_labels_read=False,observed=stamp()))
            u.write(REPORT/'status.json',dict(state='complete',observed=stamp()))
            u.write(REPORT/'owner-release.json',dict(state='all_original_bag_children_terminal_after_independent_replay'))
    except BaseException as error:
        if child is not None and child.poll() is None and (captured is None or identity.same_owner(captured,identity.process(child.pid))):
            os.killpg(child.pid,signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=True,no_automatic_retry=True,observed=stamp()));raise
    finally:
        if lane is not None:lane.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args()
    watchdog(args.watch) if args.watch else main()
