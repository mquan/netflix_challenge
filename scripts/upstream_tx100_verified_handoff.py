#!/usr/bin/env python3
"""Finish actual TX22/22 handoff after its post-fit disk guard, without retraining."""
import datetime as dt
import os
from pathlib import Path
import shutil
import signal
import upstream_repro as u
import upstream_cf96_primary_lane as owners
import upstream_cfnade_parallel as cf
import upstream_remote_cf96 as checkpoints

REPORT=u.ROOT/'experiments/upstream-tx100-verified-handoff'
SERIAL=u.ROOT/'experiments/upstream-tx100-serial-memory-recovery'
PREVIOUS=u.ROOT/'experiments/upstream-tx100-serial-memory-recovery-v2'


def main():
    REPORT.mkdir(exist_ok=False)
    if (SERIAL/'completion.json').exists():raise ValueError('Prior actual handoff already exists; do not repeat signals')
    plan=u.read(PREVIOUS/'plan.json');originals=plan['original_processes']
    if owners.same_owner(originals['tx'],owners.process(originals['tx']['pid'])):
        raise ValueError('Original temporal native must already be terminal')
    proof_path=u.ROOT/'experiments/upstream-split3-next3/s3-tx100r-verification.json';proof=u.read(proof_path)
    if (proof['state']!='complete_original_two_phase_split3_model' or proof['epochs_per_phase']!=22 or
        not proof['both_full_epoch_traces'] or proof['qualifying_labels_read'] is not False):
        raise ValueError('Complete original independent22/22 model proof required')
    for rel,digest in proof['bindings'].items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Complete original temporal source/input/output changed: '+rel)
    retired=u.ROOT/'experiments/upstream-retire-downloaded-public-latents/verification.json'
    if u.read(retired)['state']!='complete_authorized_index_verified_downloaded_latent_retirement':
        raise ValueError('Actual verified disk-reserve recovery required')
    cf_recovery=u.ROOT/'experiments/upstream-cf96-disk-resume/recovery-plan.json';recovery=u.read(cf_recovery)
    if shutil.disk_usage(u.ROOT).free<max(recovery['admission_disk_floor_bytes'],19*2**30)+.5*2**30:
        raise RuntimeError('Original CF and queued neural disk admission plus margin required')
    for rel,digest in recovery['source_sha256'].items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original CF source/controller changed')
    for key in ['cf_x','cf_owner']:
        actual=owners.process(originals[key]['pid'])
        if not owners.same_owner(originals[key],actual) or 'T' not in actual['state']:
            raise ValueError('Exact original paused CF native/controller required')
    recipe=u.read(u.ROOT/'experiments/upstream-cfnade-parallel/plan.json');states={}
    for phase,train in cf.PHASES.items():
        query='probex' if phase=='x' else 'qual';rows=1128164 if phase=='x' else 2817131
        pointer,state,directory=checkpoints.phase_checkpoint(cf.BASE/'parallel-checkpoints'/train,
                recipe['source_id'],recipe['data_id'],query,rows)
        states[phase]=dict(epoch=pointer['epoch'],tensors=len(state['files']),state_sha256=u.sha(directory/'state.json'),
                           all_original_tensors_rehashed_and_finite=True)
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
            original_processes=originals,temporal_verification_sha256=u.sha(proof_path),checkpoints=states,
            previous_post_fit_suspension_sha256=u.sha(PREVIOUS/'suspended.json'),
            disk_retirement_sha256=u.sha(retired),script_sha256=u.sha(Path(__file__)),
            rule='Original TX is terminal and full22/22 independently verified; no restart. Prior serial guard ended on its disk floor immediately before the verifier completed. Source/output closure and all CF checkpoint tensors rechecked. SIGCONT only the same original CF X native and owner; publish canonical completion for unchanged downstream queues. Original full95/95 budget, checkpoint state and source remain unchanged.'))
    for key in ['cf_x','cf_owner']:
        if not owners.same_owner(originals[key],owners.process(originals[key]['pid'])):
            raise ValueError('Original CF identity changed immediately before continuation')
        os.kill(originals[key]['pid'],signal.SIGCONT)
    result=dict(state='complete_original_TX22_22_then_resumed_same_CF_native_and_owner',
                observed=dt.datetime.now(dt.timezone.utc).isoformat(),temporal_verification_sha256=u.sha(proof_path),
                checkpoints=states,qualifying_labels_read=False,completed_native_restarted=False,
                supplemental_handoff_plan_sha256=u.sha(REPORT/'plan.json'))
    u.write(REPORT/'completion.json',result)
    u.write(SERIAL/'completion.json',result)
    print('Original TX22/22 verified and terminal; same CF X/controller resumed; canonical handoff published',flush=True)


if __name__=='__main__':main()
