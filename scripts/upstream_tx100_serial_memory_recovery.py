#!/usr/bin/env python3
"""Continue the same temporal process alone, preserving CF96 RAM and checkpoints."""
import datetime as dt
import os
from pathlib import Path
import shutil
import signal
import time
import upstream_repro as u
import upstream_cf96_primary_lane as owners
import upstream_cfnade_parallel as cf
import upstream_remote_cf96 as checker
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory

REPORT=u.ROOT/'experiments/upstream-tx100-serial-memory-recovery'
PIDS={'cf_owner':40876,'cf_x':40924,'tx_owner':61826,'tx':87902}
TOKENS={'cf_owner':'upstream_cf96_disk_resume.py','cf_x':'cfnade-stage x','tx_owner':'upstream_split3_next3.py','tx':'fit tx s3-tx100r'}
CAPTURED={}

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def safe(workers,current,baseline,free):
    return (sum(v['bytes'] for v in workers.values())<=24*2**30 and sum(v['rss_bytes'] for v in workers.values())<=24*2**30 and
        current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and free>=18.75*2**30)

def main():
    REPORT.mkdir(exist_ok=False);originals={}
    for key,pid in PIDS.items():
        p=owners.process(pid)
        if not p or TOKENS[key] not in p['command'] or 'T' not in p['state']:raise ValueError('Exact paused original owners/native states required')
        originals[key]=p
        CAPTURED[key]=p
    verifier=owners.process(61827)
    if not verifier or 'verify_upstream_split3_next3.py' not in verifier['command']:raise ValueError('Original independent temporal verifier must remain live')
    cleanup=u.ROOT/'experiments/upstream-retire-downloaded-public-qual-cache-v2/verification.json'
    if not cleanup.exists() or u.read(cleanup)['state']!='complete_authorized_downloaded_public_prediction_cache_retirement':raise ValueError('Actual completed authorized disk-reserve recovery required')
    old_plan=u.ROOT/'experiments/upstream-cfnade-parallel/plan.json';plan=u.read(old_plan);checkpoints={}
    for phase,train in cf.PHASES.items():
        query='probex' if phase=='x' else 'qual';rows=1128164 if phase=='x' else 2817131
        pointer,state,directory=checker.phase_checkpoint(cf.BASE/'parallel-checkpoints'/train,plan['source_id'],plan['data_id'],query,rows)
        checkpoints[phase]=dict(epoch=pointer['epoch'],arrays=len(state['files']),state_sha256=u.sha(directory/'state.json'),all_complete_saved_tensors_rehashed_and_finite=True)
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(resources.__file__),old_plan,cleanup,
        u.ROOT/'experiments/upstream-split3-next3/plan.json',u.ROOT/'experiments/upstream-split3-matched-queue/failure.json']}
    baseline=memory.memory();workers=resources.footprints();free=shutil.disk_usage(u.ROOT).free
    if not safe(workers,baseline,baseline,free):raise RuntimeError('Actual allocated original model and paused CF state exceed serial headroom')
    u.write(REPORT/'plan.json',dict(created=stamp(),original_processes=originals,original_verifier=verifier,source_sha256=bindings,checkpoints=checkpoints,
        rule='Same in-memory original TX100 process resumes by SIGCONT, no model/data/operator/epoch changes or restart. Original CF X native and CF supervisor remain SIGSTOP, Q complete81 checkpoint retained. TX runs alone; immediate measured24GiB total RSS/physical,128MiB swap-growth,18.75GiB disk guard suspends only original TX and its owner in place. Full22/22 native proof required before CF original owners resume. Matched5control+2expanded folds/partial preprocessing preserved, no promotion.'))
    for key in ['tx_owner','tx']:os.kill(PIDS[key],signal.SIGCONT)
    while True:
        for key in ['cf_owner','cf_x']:
            actual=owners.process(PIDS[key])
            if not owners.same_owner(originals[key],actual) or 'T' not in actual['state']:
                raise RuntimeError('Preserved CF original state changed unexpectedly')
        for key,h in bindings.items():
            if u.sha(u.ROOT/key)!=h:raise ValueError('Original serial-recovery bindings changed')
        native=owners.process(PIDS['tx']);controller=owners.process(PIDS['tx_owner'])
        if native and not owners.same_owner(originals['tx'],native):raise ValueError('Original native PID was reused')
        workers=resources.footprints();current=memory.memory();free=shutil.disk_usage(u.ROOT).free
        if not safe(workers,current,baseline,free):
            for key in ['tx','tx_owner']:
                actual=owners.process(PIDS[key])
                if owners.same_owner(originals[key],actual):os.kill(PIDS[key],signal.SIGSTOP)
            u.write(REPORT/'suspended.json',dict(state='original_same_native_and_owner_suspended_with_all_RAM_state_preserved',workers=workers,memory=current,disk_free_bytes=free,observed=stamp()))
            return
        proof=u.ROOT/'experiments/upstream-split3-next3/s3-tx100r-verification.json'
        u.write(REPORT/'status.json',dict(state='same_original_full_temporal_running_without_other_active_models',workers=workers,memory=current,
            native_live=native is not None,controller_live=controller is not None,disk_free_bytes=free,observed=stamp()))
        if proof.exists():
            result=u.read(proof)
            if result['state']!='complete_original_two_phase_split3_model' or result['epochs_per_phase']!=22 or not result['both_full_epoch_traces']:raise ValueError('Full original temporal proof required')
            # Native model has exited. Existing CF owners resume their exact
            # paused RAM trajectory/checkpoint-aware loop; no new command recipe.
            if native is not None:raise ValueError('Original full native must exit before CF resumes')
            os.kill(PIDS['cf_x'],signal.SIGCONT);os.kill(PIDS['cf_owner'],signal.SIGCONT)
            u.write(REPORT/'completion.json',dict(state='complete_original_TX22_22_then_resumed_same_CF_native_and_owner',observed=stamp(),
                temporal_verification_sha256=u.sha(proof),checkpoints=checkpoints,qualifying_labels_read=False))
            return
        if not native and not controller:
            actual_verifier=owners.process(61827)
            receipt=u.ROOT/'experiments/upstream-split3-next3/s3-tx100r-receipt.json'
            if not owners.same_owner(verifier,actual_verifier) or not receipt.exists():
                raise RuntimeError('Original execution/verifier exited without full proof; no restart attempted')
        time.sleep(5)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        for key in ['tx','tx_owner']:
            p=owners.process(PIDS[key])
            if key in CAPTURED and owners.same_owner(CAPTURED[key],p):os.kill(PIDS[key],signal.SIGSTOP)
        if REPORT.exists():u.write(REPORT/'failure.json',dict(error=repr(error),original_native_state_preserved=True,observed=stamp()))
        raise
