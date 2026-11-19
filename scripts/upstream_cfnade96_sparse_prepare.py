#!/usr/bin/env python3
"""Prepare a private320/96 backend and compare it with the live reference math."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import upstream_cfnade48 as inherited

u=inherited.u
BASE=u.ROOT/'artifacts/upstream-cfnade96-sparse-resume'
TREE=BASE/'source'
REPORT=u.ROOT/'experiments/upstream-cfnade96-sparse-resume'
REFERENCE=u.ROOT/'artifacts/upstream-cfnade-checkpoint/source'


def prepare():
    if BASE.exists():raise ValueError('Existing private sparse96 preparation preserved')
    checked=inherited.verify_backend();origin=inherited.TREE
    if u.sha(origin/'src/cfnade_baseline.rs')!=u.sha(REFERENCE/'src/cfnade.rs'):
        raise ValueError('Sparse baseline is not the current original96 implementation')
    REPORT.mkdir(parents=True,exist_ok=True);TREE.mkdir(parents=True)
    for path in (origin/'src').rglob('*'):
        if path.is_symlink():raise ValueError('Source symlink needs explicit review before copying')
    shutil.copytree(origin/'src',TREE/'src')
    for name in ['Cargo.toml','Cargo.lock']:
        shutil.copyfile(origin/name,TREE/name)
    if (origin/'build.rs').exists():shutil.copyfile(origin/'build.rs',TREE/'build.rs')
    driver=(REFERENCE/'src/bin/cfnade-stage.rs').read_text()
    declaration='let cfg=CfNadeConfig{n_hidden:320,rank:96,n_epochs:95,..CfNadeConfig::default()};'
    if driver.count(declaration)!=1:raise ValueError('Original320/96/95 driver config anchor differs')
    driver=driver.replace('    '+declaration+'\n','')
    anchor='    let (train,query)=match phase.as_str()'
    if driver.count(anchor)!=1:raise ValueError('Original96 phase dispatch anchor differs')
    driver=driver.replace(anchor,'    '+declaration+'\n    if phase=="--describe"{println!("{:?}",cfg);return;}\n'+anchor)
    (TREE/'src/bin/cfnade96-sparse-stage.rs').write_text(driver)
    parity=(TREE/'src/bin/cfnade-fast-parity.rs').read_text()
    shapes='[(4,3),(40,12),(320,96)]'
    if parity.count(shapes)!=1:raise ValueError('Generic native parity coverage changed')
    # The generic executable checks24 original/candidate cases, including320/96.
    # This additional executable checks8 cases against the ACTUAL live backend,
    # so its timings isolate sparse-row traversal from previous optimizations.
    parity=parity.replace(shapes,'[(320,96)]').replace('cfnade_original','cfnade_baseline')
    (TREE/'src/bin/cfnade96-baseline-parity.rs').write_text(parity)
    source={str(p.relative_to(u.ROOT)):u.sha(p) for p in TREE.rglob('*') if p.is_file()}
    for p in [Path(__file__),inherited.candidate.REPORT/'verification.json',REFERENCE/'src/cfnade.rs',
              u.ROOT/'experiments/upstream-cfnade-parallel/plan.json']:
        source[str(p.relative_to(u.ROOT))]=u.sha(p)
    preparation=dict(state='prepared_private_kernel_only',source_pin=u.PIN,source_sha256=source,
        source_id=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        inherited_preparation=checked,hidden=320,rank=96,epochs_per_phase=95,
        live_reference_source_unchanged=True,checkpoint_migration_performed=False,production_eligible=False)
    u.write(REPORT/'preparation.json',preparation)
    env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_BUILD_JOBS='2',CARGO_NET_OFFLINE='true')
    for key in ['CF_CHECKPOINT_DIR','CF_CHECKPOINT_SOURCE','CF_CHECKPOINT_DATA','CF_EXIT_AFTER_CHECKPOINT','CF_PROOF_OUTPUT','CF_PROOF_INTERRUPT']:
        env.pop(key,None)
    names=['cfnade-fast-parity','cfnade96-baseline-parity','cfnade96-sparse-stage']
    args=[str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline']
    for name in names:args+=['--bin',name]
    with (REPORT/'build.log').open('w') as log:
        subprocess.run(args,cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    records=[]
    for name in names[:2]:
        with (REPORT/(name+'.log')).open('w') as log:
            subprocess.run([str(BASE/'target/release'/name)],cwd=TREE,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        records.append(dict(binary=name,sha256=u.sha(BASE/'target/release'/name),log_sha256=u.sha(REPORT/(name+'.log'))))
    description=subprocess.check_output([str(BASE/'target/release/cfnade96-sparse-stage'),'--describe'],cwd=TREE,env=env,text=True).strip()
    for value in ['n_hidden: 320','rank: 96','n_epochs: 95','swa_start: 18']:
        if value not in description:raise ValueError('Actual original96 runtime configuration differs')
    for relative,digest in source.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Prepared sparse96 source changed')
    result=dict(state='verified_kernel_only',preparation_sha256=u.sha(REPORT/'preparation.json'),
        source_id=preparation['source_id'],description=description,checks=records,
        stage_binary_sha256=u.sha(BASE/'target/release/cfnade96-sparse-stage'),
        checkpoint_migration_performed=False,full_data_resume_parity_established=False,production_eligible=False,
        scope='Original-source24-case parity plus8 actual-live-backend320/96 cases, initialization and7 epochs; all84 state arrays/RNG and direct/grouped predictions. Full-data warm-resume/next-epoch equality is still required.')
    u.write(REPORT/'kernel-verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':prepare()
