#!/usr/bin/env python3
"""Separate original1200 preparation/build; never train from this module."""
import argparse
from pathlib import Path
import shutil
import subprocess
import upstream_high_capacity_original as preparation

u=preparation.u
RUN=u.ROOT/'artifacts/upstream-original1200'
TREE=RUN/'source'
REPORT=u.ROOT/'experiments/upstream-original1200'
DECL=u.ROOT/'ORIGINAL1200_EXECUTION_PLAN.md'
NAME='tsvdx5-1200lm'


def frozen():
    return [Path(__file__),DECL,u.ROOT/'scripts/verify_upstream_original1200.py',
        u.ROOT/'scripts/upstream_original1200_queue.py',
        u.ROOT/'experiments/upstream-original1200-preflight/preflight.json']


def prepare():
    # Only process-local module variables change. The original source, target,
    # execution plan, owners and live native process are never modified.
    preparation.RUN,preparation.TREE,preparation.REPORT=RUN,TREE,REPORT
    preparation.JOBS={NAME:('tsvdx5-new',1200,23)}
    plan=preparation.prepare()
    if 'original1200_wrapper_sha256' in plan:
        if plan['original1200_wrapper_sha256']!=u.sha(Path(__file__)):
            raise ValueError('Original1200 preparation wrapper changed')
        return plan
    preflight=u.read(u.ROOT/'experiments/upstream-original1200-preflight/preflight.json')
    if (preflight['state']!='complete_original1200_exact_preset_and58_output_preflight_no_training' or
        preflight['source_pin']!=u.PIN or len(preflight['expected_outputs'])!=58 or
        preflight['original_registry_job']!=plan['jobs'][NAME]['original_registry_job']):
        raise ValueError('Exact full original1200 preflight required')
    for p in frozen():plan['bindings'][str(p.relative_to(u.ROOT))]=u.sha(p)
    plan.update(original1200_wrapper_sha256=u.sha(Path(__file__)),
        exact_expected_outputs=preflight['expected_outputs'],neighbors=100,
        output_reserve_bytes=int(1.25*2**30),new_namespace_only=True)
    u.write(REPORT/'execution-plan.json',plan)
    return plan


def build():
    output=REPORT/'original-build-receipt.json'
    if output.exists():raise ValueError('Original1200 build evidence retained')
    plan=u.read(REPORT/'execution-plan.json')
    for relative,digest in plan['bindings'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Full original1200 input changed before build')
    env=u.environment();env.update(CARGO_TARGET_DIR=str(RUN/'target'),CARGO_BUILD_JOBS='1')
    cargo=shutil.which('cargo',path=env['PATH'])
    if not cargo:raise ValueError('Pinned Rust compiler unavailable')
    command=[cargo,'build','--release','--locked','--features','blas','--bin','tsvdx5-new']
    subprocess.run(command,cwd=TREE,env=env,check=True)
    for relative,digest in plan['bindings'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Full original1200 input changed during build')
    binary=RUN/'target/release/tsvdx5-new'
    u.write(output,dict(source_commit=u.PIN,command=command,cwd=str(TREE),
        preparation_sha256=u.sha(REPORT/'execution-plan.json'),builder_sha256=u.sha(Path(__file__)),
        cargo=subprocess.check_output([cargo,'--version'],env=env,text=True).strip(),
        environment={k:env[k] for k in ['CARGO_TARGET_DIR','CARGO_BUILD_JOBS','RUSTFLAGS'] if k in env},
        binary_path=str(binary),binaries={binary.name:u.sha(binary)},
        original_source_and_complete_data_unchanged=True,training_launched=False))
    print('Original1200 isolated source build complete; no training launched',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','build']);args=parser.parse_args()
    print(prepare()['state'],flush=True) if args.action=='prepare' else build()
