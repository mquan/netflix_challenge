#!/usr/bin/env python3
"""Adopt the exact original H512 process, then replay every saved pipeline."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import re
import shutil
import upstream_own391_paired_queue as original

u,identity,resources,memory=original.u,original.identities,original.resources,original.memory
REPORT=u.ROOT/'experiments/upstream-own391-h512-scan-continuation'
PRIOR=u.ROOT/'experiments/upstream-own391-h512-readonly-recovery'
OLD=original.directory('h512')
PEER_FAILURES=[u.ROOT/'experiments'/name/'failure.json' for name in [
    'upstream-original1200-queue','upstream-own391-cpu-tree-deferred-recovery','upstream-complete-cf-delta-serialized-recovery']]


def peers_stopped(peers):
    rows=[line.split(None,3) for line in subprocess.check_output(['ps','-axo','pid=,pgid=,stat=,command='],text=True).splitlines()]
    for peer in peers:
        if not identity.same_owner(peer,identity.process(peer['pid'])):return False
        group=[row for row in rows if len(row)==4 and int(row[1])==peer['pgid'] and not row[2].startswith('Z')]
        if not group or any('T' not in row[2] for row in group):return False
    return True


def watchdog(pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=pid:raise ValueError('Exact monitor required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(pid)):
            if (REPORT/'owner-release.json').exists():return
            captured=u.read(REPORT/'child-ownership.json');stopped=False
            if identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            diagnostic=REPORT/'diagnostic-ownership.json'
            if diagnostic.exists():
                helper=u.read(diagnostic)
                if identity.same_owner(helper,identity.process(helper['pid'])):os.killpg(helper['pid'],signal.SIGSTOP)
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
    return original.safe(reserved,current,baseline)


def diagnostic_safe(workers,current,baseline,pid,helper_pid=None,reserve_helper=False):
    """All scientific workers stay stopped; bounded ordinary diagnostic only."""
    own=workers.get(str(pid))
    if own is None or max(own['bytes'],own['rss_bytes'])>16*2**30:return False
    if helper_pid is not None:
        helper=workers.get(str(helper_pid))
        if helper is None or max(helper['bytes'],helper['rss_bytes'])>10*2**30:return False
        if reserve_helper:
            workers={k:dict(v) for k,v in workers.items()}
            workers[str(helper_pid)]['bytes']=max(helper['bytes'],10*2**30)
            workers[str(helper_pid)]['rss_bytes']=max(helper['rss_bytes'],10*2**30)
    return (sum(w['bytes'] for w in workers.values())<24*2**30 and
        sum(w['rss_bytes'] for w in workers.values())<24*2**30 and
        current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and
        shutil.disk_usage(u.ROOT).free>=18.75*2**30)


def main():
    if REPORT.exists():raise ValueError('Existing exact recovery retained')
    failure=u.read(OLD/'failure.json');captured=failure['child'];current=identity.process(captured['pid'])
    old_owner=u.read(OLD/'monitor.json')
    if (failure['error']!="RuntimeError('Original neural resource/watchdog guard; preserve exact child RAM')" or
        not failure['actual_child_RAM_preserved'] or not failure['no_automatic_retry'] or
        identity.same_owner(old_owner,identity.process(old_owner['pid'])) or
        not identity.same_owner(captured,current) or 'T' not in current['state'] or
        current['pgid']!=current['pid'] or
        str(u.ROOT/'scripts/upstream_own391_paired_mps.py')+' --variant h512 --worker' not in current['command'] or
        (OLD/'completion.json').exists()):raise ValueError('Only exact stopped original orphan H512 fit may continue')
    fixture=u.ROOT/'experiments/upstream-paused-readonly-scan-fixture/verification.json'
    fp=u.read(fixture)
    if (fp['state']!='private_fixture_readonly_scan_completed_bytes_equal_stop_preserved' or fp['scanner_returncode']!=0 or
        fp['fixture_bytes']!=8*2**20 or fp['sha256_before']!=fp['sha256_after'] or
        fp['stopped_state_before']!='Ts' or fp['stopped_state_after']!='Ts' or
        fp['scan_log_sha256']!=u.sha(fixture.parent/'scan.log')):raise ValueError('Actual successful readonly stopped private fixture required')
    frozen=dict(u.read(OLD/'plan.json')['source_sha256'])
    for path in [Path(__file__),u.ROOT/'OWN391_H512_SCAN_CONTINUATION.md',OLD/'plan.json',OLD/'failure.json',OLD/'monitor.json',
        u.ROOT/'experiments/upstream-high-capacity-successor/completion.json',fixture,fixture.parent/'scan.log',
        fixture.parent/'worker-source.txt',u.ROOT/'experiments/upstream-readonly-cache-trim-second-pass/verification.json',*PEER_FAILURES,
        PRIOR/'plan.json',PRIOR/'failure.json',PRIOR/'monitor.json',PRIOR/'diagnostic-ownership.json',
        u.ROOT/'scripts/upstream_own391_h512_readonly_recovery.py',u.ROOT/'OWN391_H512_READONLY_RECOVERY.md']:
        key=str(path.relative_to(u.ROOT));digest=u.sha(path)
        if key in frozen and frozen[key]!=digest:raise ValueError('Conflicting original frozen expectation')
        frozen[key]=digest
    prior_owner=u.read(PRIOR/'monitor.json');prior_failure=u.read(PRIOR/'failure.json')
    helper=u.read(PRIOR/'diagnostic-ownership.json');actual_helper=identity.process(helper['pid'])
    if (identity.same_owner(prior_owner,identity.process(prior_owner['pid'])) or
        prior_failure['error']!="RuntimeError('Readonly diagnostic guard; preserve helper and original target')" or
        not prior_failure['actual_child_RAM_preserved'] or not identity.same_owner(captured,prior_failure['child']) or
        not identity.same_owner(helper,actual_helper) or 'T' not in actual_helper['state'] or helper['pgid']!=helper['pid'] or
        helper['command']!='/usr/bin/leaks --noContent --nostacks --quiet '+str(captured['pid']) or
        (PRIOR/'diagnostic-verification.json').exists()):raise ValueError('Only the exact preserved partial scanner may continue')
    leaks_sha=u.sha('/usr/bin/leaks')
    peers=[u.read(p)['child'] for p in PEER_FAILURES]
    if not peers_stopped(peers):raise ValueError('All exact other scientific groups must remain stopped')
    baseline=u.read(OLD/'status.json')['memory']
    REPORT.mkdir();watch=None;child=None;lane=None;primary=None;scanner=None
    u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    u.write(REPORT/'child-ownership.json',captured)
    u.write(REPORT/'diagnostic-ownership.json',helper)
    u.write(REPORT/'plan.json',dict(source_sha256=frozen,leaks_sha256=leaks_sha,original_child=captured,adopted_diagnostic=helper,diagnostic_cap_GiB=10,stopped_peers=peers,baseline=baseline,created=original.stamp(),
        rule='Resume SAME stopped H512 process only, retaining tensors/optimizer/RNG and all original numerical sources. CF intake remains stopped. Serialize GPU+primary lanes, continue SAME standard readonly scanner after its1GiB helperguard; explicit10GiB helper reservation within unchanged24GiB total and16GiB target; no diagnostic restart; private fixture proof retained, then three stable ORIGINAL comp<5GiB observations, account already allocated RAM toward original16GiB reservation, nonblocking original GPU lane then recheck. Original own16/research24/comp5/swapgrowth128MiB/disk18.75 unchanged. Baseline is last recorded original preguard observation, not reset at admission. Full actual model/scaler/input replay then separate canonical verifier. Original failure/plan never overwritten. Supplemental original queue completion only after full proof for existing dependent waiters. No new fitting worker or automatic retry.'))
    def check():
        for rel,digest in frozen.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen source/evidence changed: '+rel)
        if u.sha('/usr/bin/leaks')!=leaks_sha:raise ValueError('Standard diagnostic binary changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('Exact neural monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def capacity(phase,adopt=False):
        good=0
        while True:
            check();workers=resources.footprints(captured['pid']);current=memory.memory()
            ready=admission(workers,current,baseline,captured['pid']) if adopt else original.safe(workers,current,baseline,reserve=16*2**30)
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
            primary=(u.ROOT/'experiments/full-upstream-reproduction/runner.lock').open('a')
            while True:
                check();workers=resources.footprints(helper['pid']);current=memory.memory()
                if diagnostic_safe(workers,current,baseline,captured['pid'],helper['pid'],reserve_helper=True):
                    try:
                        fcntl.flock(primary,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        try:fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:
                            fcntl.flock(primary,fcntl.LOCK_UN);raise
                        if diagnostic_safe(resources.footprints(helper['pid']),memory.memory(),baseline,captured['pid'],helper['pid'],reserve_helper=True):break
                        fcntl.flock(lane,fcntl.LOCK_UN);fcntl.flock(primary,fcntl.LOCK_UN)
                    except BlockingIOError:pass
                if watch.poll() is not None:raise RuntimeError('Readonly recovery watchdog unavailable')
                u.write(REPORT/'status.json',dict(state='waiting_serialized_readonly_diagnostic_capacity',workers=workers,memory=current,observed=original.stamp()));time.sleep(10)
            current=identity.process(captured['pid'])
            if not identity.same_owner(captured,current) or 'T' not in current['state'] or not peers_stopped(peers):raise ValueError('Exact target and other groups stopped before readonly scan required')
            diagnostic_before=memory.memory()
            actual_helper=identity.process(helper['pid'])
            if not identity.same_owner(helper,actual_helper) or 'T' not in actual_helper['state']:
                raise ValueError('Exact stopped scanner required before continuation')
            os.killpg(helper['pid'],signal.SIGCONT)
            u.write(REPORT/'diagnostic-continuation.json',dict(same_actual_diagnostic=helper,restarted=False,
                target_remains_stopped=captured,observed=original.stamp()))
            while identity.same_owner(helper,identity.process(helper['pid'])):
                check();workers=resources.footprints(helper['pid']);current_memory=memory.memory();target=identity.process(captured['pid'])
                if not identity.same_owner(helper,identity.process(helper['pid'])):break
                if (not identity.same_owner(captured,target) or 'T' not in target['state'] or not peers_stopped(peers) or
                    not diagnostic_safe(workers,current_memory,baseline,captured['pid'],helper['pid']) or
                    current_memory['compressor_bytes']>diagnostic_before['compressor_bytes']+128*2**20 or watch.poll() is not None):
                    u.write(REPORT/'diagnostic-guard-sample.json',dict(workers=workers,memory=current_memory,target=target))
                    raise RuntimeError('Continued readonly diagnostic guard; preserve same helper and target')
                u.write(REPORT/'status.json',dict(state='same_original_readonly_scan_continued_target_stopped',workers=workers,memory=current_memory,observed=original.stamp()));time.sleep(2)
            log_path=PRIOR/'diagnostic.log';text=log_path.read_text();target=identity.process(captured['pid'])
            if (not re.search(r'Process '+str(captured['pid'])+r': \d+ nodes malloced for',text) or
                not re.search(r'Process '+str(captured['pid'])+r': \d+ leaks for \d+ total leaked bytes',text) or
                not identity.same_owner(captured,target) or 'T' not in target['state']):
                raise ValueError('Complete successful scan summaries and exact stopped target required')
            u.write(REPORT/'diagnostic-verification.json',dict(state='complete_same_standard_readonly_scanner_full_summary_target_still_stopped',
                target=captured,same_actual_diagnostic=helper,exit_status_available=False,log_path=str(log_path.relative_to(u.ROOT)),
                log_sha256=u.sha(log_path),leaks_sha256=leaks_sha,fixture_sha256=u.sha(fixture),before=diagnostic_before,after=memory.memory(),
                target_still_stopped=True,real_numerical_bytes_compared=False,model_state_dumped=False,original_worker_continued=False))
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
            while identity.same_owner(captured,identity.process(captured['pid'])):monitor('same_original_H512_fit_continued');time.sleep(3)
            fit=u.ROOT/'experiments/upstream-own391-paired-h512-mps/fit-completion.json'
            if not fit.exists() or u.read(fit)['state']!='complete_all_five_original_budget_matched_B391_neural_fits':raise ValueError('Exact original fit terminal without complete receipt; no retry')
            u.write(REPORT/'adopted-fit-terminal.json',dict(fit_completion_sha256=u.sha(fit),exit_status_available=False,full_independent_replay_required=True))
            for phase,script in [('independent_actual_replay','verify_upstream_own391_paired_mps.py'),('separate_canonical_scores','verify_upstream_own391_paired_canonical.py')]:
                capacity(phase);path=u.ROOT/'scripts'/script
                with (REPORT/(phase+'.log')).open('x') as log:
                    captured=None
                    child=subprocess.Popen([sys.executable,'-u',str(path),'--variant','h512'],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    captured=identity.process(child.pid)
                    if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                        raise ValueError('Dedicated actual verification child required')
                    u.write(REPORT/'child-ownership.json',captured)
                    while child.poll() is None:monitor(phase);time.sleep(3)
                    if child.returncode:raise RuntimeError('Independent phase failed; no retry: '+phase)
                    child=None
            proof=u.ROOT/'experiments/upstream-own391-paired-h512-mps/independent-canonical-verification.json'
            if u.read(proof)['state']!='complete_separate_matched_B391_canonical_neural_and_fixed_half_scores':raise ValueError('Full independent canonical proof required')
            completion=dict(state='complete_matched_B391_neural_arm_actual_fits_replay_and_canonical_scores',variant='h512',
                canonical_verification_sha256=u.sha(proof),qualifying_predictions_generated=False,observed=original.stamp(),
                exact_recovery_plan_sha256=u.sha(REPORT/'plan.json'),original_guard_failure_preserved=True)
            u.write(REPORT/'completion.json',completion)
            if (OLD/'completion.json').exists():raise ValueError('Never overwrite original completion')
            u.write(OLD/'completion.json',completion)
            u.write(REPORT/'status.json',dict(state='complete',observed=original.stamp()))
            u.write(REPORT/'owner-release.json',dict(state='all_owned_workers_terminal_after_full_proof'))
    except BaseException as error:
        stopped=False
        if identity.same_owner(helper,identity.process(helper['pid'])):os.killpg(helper['pid'],signal.SIGSTOP)
        if captured and identity.same_owner(captured,identity.process(captured['pid'])):
            os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
        elif child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGSTOP);stopped=True
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=stopped,no_automatic_retry=True,observed=original.stamp()))
        raise
    finally:
        if lane is not None:lane.close()
        if primary is not None:primary.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args()
    watchdog(args.watch) if args.watch else main()
