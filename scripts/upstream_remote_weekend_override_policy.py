#!/usr/bin/env python3
"""Validate the pinned remote resource-policy variant, without admitting outputs."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
COMMIT='444fa334fede478f1724162e81d6080d4bb1bf21'
BASE_PLAN_SHA='2b8194eba4356db9a67af6d2b69fdb71816d2ba0b95df939b1cf122e86319514'
BASE_CODE_SHA='816a91bd2e6d4331dc4433339f6198d318ecca9109c8a56cd2085c87e900ca4c'
REMOTE_PLAN_SHA='9f75c1e35ae5cf833a5857d72c32c4e4ee45a4dc092c8d07a124e34bf83d066f'
REMOTE_CODE_SHA='84f92e4e8b8f6118a5e72a95dd9642a3dd3dbe9d417dba53d66a7d58a8d936f2'
JOBS={'s3-rbm400pf':'r8','s3-rbm800p':'r9','s3-tx300r':'r10'}
PLAN_PATH='experiments/upstream-remote-weekend/plan.json'
CODE_PATH='scripts/upstream_remote_weekend.py'
EXECUTION='experiments/upstream-remote-weekend/execution/'


def sha(data):return hashlib.sha256(data).hexdigest()


def validate(base_plan,base_code,remote_plan,remote_code,preparations):
    before,after=json.loads(base_plan),json.loads(remote_plan)
    expected=copy.deepcopy(before)
    expected['controller_sha256'][CODE_PATH]=REMOTE_CODE_SHA
    if after!=expected:
        raise ValueError('Only the recorded controller policy hash may change; all recipes/data/deadlines remain original')
    for data,digest in [(base_plan,BASE_PLAN_SHA),(base_code,BASE_CODE_SHA),
                        (remote_plan,REMOTE_PLAN_SHA),(remote_code,REMOTE_CODE_SHA)]:
        if sha(data)!=digest:raise ValueError('Exact pinned policy source bytes required')
    code=base_code.decode()
    marker='def capacity(workers, current, baseline, own=None, reserve=0):\n'
    compressor="            current['compressor_bytes'] < 12*2**30 and\n"
    admission="capacity(workers,current,baseline,reserve=64*2**30) and current['compressor_bytes']<10*2**30 and"
    if any(code.count(x)!=1 for x in (marker,compressor,admission)):
        raise ValueError('Exact original capacity/admission sites required')
    code=code.replace(marker,marker+'    return True  # owner override 2026-11-21: no memory/swap/compressor limits in this session\n')
    code=code.replace(compressor,'').replace(admission,'capacity(workers,current,baseline,reserve=64*2**30) and')
    if code.encode()!=remote_code:raise ValueError('Override contains an additional unreviewed code change')
    if set(preparations)!=set(JOBS):raise ValueError('All three fresh preparation records required')
    for name,payload in preparations.items():
        p=json.loads(payload)
        if (p['state']!='prepared_own_data_only' or p['plan_sha256']!=REMOTE_PLAN_SHA or
            p['source_pin']!=before['source_pin'] or p['row_map_sha256']!=before['row_map_sha256'] or
            p['known_rows_absent_from_training'] is not True or p['qualifying_targets_zero'] is not True or
            p['qualifying_labels_read'] is not False or p['public_prediction_values_read'] is not False or
            set(p['binary_sha256'])!={'target/release/fit','target/release/split_3'} or
            any(not re.fullmatch('[0-9a-f]{64}',x) for x in p['binary_sha256'].values())):
            raise ValueError('Original own-data-only preparation declarations required: '+name)
    jobs=after['jobs']
    return dict(state='verified_pinned_remote_resource_policy_variant_metadata_only',
        original_blueprint_sha256=BASE_PLAN_SHA,remote_plan_sha256=REMOTE_PLAN_SHA,
        original_controller_sha256=BASE_CODE_SHA,remote_controller_sha256=REMOTE_CODE_SHA,
        recipes_data_source_and_deadlines_unchanged=True,
        jobs=[dict(name=j['name'],epochs_per_phase=j['epochs_per_phase'],named_pairs=len(j['published_columns']),
                   tree='artifacts/upstream-remote-weekend-'+JOBS[j['name']]) for j in jobs],
        all67_named_pairs_and_both_full_phases_still_required=True,
        original_remote_memory_rss_compressor_swap_guarantees=False,
        original_remote_deadline_watchdog_and_disk_checks_retained=True,
        coordinator_local_guards_unchanged=True,
        preparation_flags_are_declarations_not_full_payload_admission=True,
        numerical_returns_admitted=False,qualifying_labels_read=False)


def git_blob(commit,path):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full pinned remote commit required')
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def bundle(commit=COMMIT):
    return ((ROOT/PLAN_PATH).read_bytes(),(ROOT/CODE_PATH).read_bytes(),
            git_blob(commit,PLAN_PATH),git_blob(commit,CODE_PATH),
            {job:git_blob(commit,EXECUTION+'bootstrap-'+tree+'/preparation.json') for job,tree in JOBS.items()})


def main(commit):
    values=bundle(commit);result=validate(*values)
    report=ROOT/'experiments/upstream-remote-weekend-owner-override-policy'
    if report.exists():raise ValueError('Existing policy observation preserved')
    report.mkdir()
    files={'remote-plan.json':values[2],'remote-controller.py':values[3],
           'remote-start.json':git_blob(commit,EXECUTION+'start.json')}
    files.update({job+'-preparation.json':payload for job,payload in values[4].items()})
    for name,payload in files.items():(report/name).write_bytes(payload)
    result.update(observed=dt.datetime.now(dt.timezone.utc).isoformat(),remote_commit=commit,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__).read_bytes()),
                       PLAN_PATH:BASE_PLAN_SHA,CODE_PATH:BASE_CODE_SHA},
        snapshots_sha256={name:sha(payload) for name,payload in files.items()},
        launch_receipts_observed={job:subprocess.run(['git','cat-file','-e',commit+':'+EXECUTION+job+'/'+job+'-launch.json'],
            cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 for job in JOBS})
    (report/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['state'],'no numerical output admitted',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-commit',default=COMMIT);main(p.parse_args().remote_commit)
