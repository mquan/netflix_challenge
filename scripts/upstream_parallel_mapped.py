#!/usr/bin/env python3
"""Verify the published parallel factor model with isolated mapped user arrays."""
import argparse
from pathlib import Path
import subprocess
import upstream_repro as u
import upstream_mapped_prepare as serial

BASE=u.ROOT/'artifacts/upstream-parallel-mapped'
TREE=BASE/'source'
REPORT=u.ROOT/'experiments/upstream-parallel-mapped'
SUPPORT=u.ROOT/'research/parallel_mapped'
ORIGINAL_SERIAL_BASE=serial.BASE


def prepare():
    serial.BASE,serial.TREE=BASE,TREE
    serial.prepare();REPORT.mkdir(parents=True,exist_ok=True)
    inherited={name:u.sha(TREE/name) for name in ['src/tx.rs','src/tx_original.rs','src/tx_storage.rs']}
    if any(digest!=u.sha(ORIGINAL_SERIAL_BASE/'source'/name) for name,digest in inherited.items()):
        raise ValueError('Inherited serial storage implementation differs from its existing proof')
    source=serial.pinned('src/tsvdx4p.rs')
    audit='\nimpl Tsvdx4pModel { pub fn audit_parameters(&self)->Vec<u32>{self.inner.audit_parameters()} }\n'
    original=serial.replace_once(source,'use crate::tx::{TxConfig, TxModel, freq_bin, day_ranges};',
        'use crate::tx_original::{TxConfig, TxModel, freq_bin, day_ranges};')
    (TREE/'src/tsvdx4p_original.rs').write_text(original+audit)
    (TREE/'src/tsvdx4p.rs').write_text(source+audit)
    lib=TREE/'src/lib.rs';lib.write_text(lib.read_text()+'\npub mod tsvdx4p_original;\n')
    dispatcher=serial.pinned('src/bin/tsvdx4p-new.rs')
    config=dispatcher.split('fn run_lm(',1)[1].split('    let cfg = ',1)[1].split('    if save_train {',1)[0].strip().removesuffix(';')
    proof=(SUPPORT/'parity.rs').read_text().replace('PUBLISHED_CONFIG',config)
    (TREE/'src/bin/tsvdx4p-storage-parity.rs').write_text(proof)
    patch=subprocess.check_output(['git','diff','--','src/tx.rs','src/tsvdx4p.rs','src/lib.rs','Cargo.toml'],cwd=TREE)
    (BASE/'parallel-storage.patch').write_bytes(patch)
    generated={name:u.sha(TREE/name) for name in ['src/tx.rs','src/tx_original.rs','src/tx_storage.rs',
        'src/tsvdx4p.rs','src/tsvdx4p_original.rs','src/lib.rs','src/bin/tsvdx4p-storage-parity.rs',
        'Cargo.toml','data/qual_ratings/qual_ratings.csv.gz']}
    record=dict(source_pin=u.PIN,generated=generated,patch_sha256=u.sha(BASE/'parallel-storage.patch'),
        inherited_serial_source=inherited,serial_prepare_sha256=u.sha(Path(serial.__file__)),
        serial_proof_sha256=u.sha(u.ROOT/'experiments/full-upstream-reproduction/mapped-parity/verification.json'),
        support_sha256=u.sha(SUPPORT/'parity.rs'),script_sha256=u.sha(Path(__file__)),
        scope='Storage candidate only. Published low_memory=true, factors/epochs and logical chunk configurations preserved in parity cases. No full-data training or resource-feasibility claim.')
    u.write(BASE/'parallel-preparation.json',record);return record


def verify():
    prepared=prepare()
    env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_BUILD_JOBS='2',CARGO_NET_OFFLINE='true')
    with (REPORT/'build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
            '--bin','tsvdx4p-storage-parity'],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    executable=BASE/'target/release/tsvdx4p-storage-parity';records=[];reference=None
    for workers in [2,4]:
        for mapped in [False,True]:
            name=('mapped' if mapped else 'heap')+'-workers'+str(workers)
            arm={k:v for k,v in env.items() if k not in ['TX_MMAP_DIR','TX_LAZY_DAY']}
            arm['RAYON_NUM_THREADS']=str(workers)
            scratch=BASE/('scratch-'+name)
            if mapped:arm['TX_MMAP_DIR']=str(scratch)
            child=subprocess.run([str(executable)],cwd=TREE,env=arm,capture_output=True,text=True,timeout=600)
            output=child.stdout+child.stderr;path=REPORT/(name+'.log');path.write_text(output)
            signatures=[line for line in child.stdout.splitlines() if line.startswith('CASE ')]
            if reference is None and child.returncode==0:reference=signatures
            same=signatures==reference and len(signatures)==24
            if list(scratch.glob('tx-*.f32')):raise ValueError('Mapping files survived successful proof exit')
            records.append(dict(arm=name,exit_code=child.returncode,log_sha256=u.sha(path),
                signatures_match_across_storage_and_workers=same,tail=output[-1400:]))
            print(name,child.returncode,'same signatures:',same,flush=True)
    if any(u.sha(TREE/name)!=digest for name,digest in prepared['generated'].items()):
        raise ValueError('Prepared parallel candidate changed during verification')
    result=dict(source_pin=u.PIN,preparation=prepared,executable_sha256=u.sha(executable),
        cargo_lock_sha256=u.sha(TREE/'Cargo.lock'),arms=records,
        pass_all=all(r['exit_code']==0 and r['signatures_match_across_storage_and_workers'] for r in records),
        scope='Synthetic arithmetic parity through16000 factors and20 epochs, including original sequential epochs1/10/20 and16 logical chunks, both heap/mapped backends and2/4 physical worker pools. Not full-scale resource validation or a target score.')
    u.write(REPORT/'verification.json',result)
    if not result['pass_all']:raise ValueError('Parallel storage parity failed; failed artifacts preserved')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare()
    else:verify()
