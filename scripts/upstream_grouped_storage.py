#!/usr/bin/env python3
"""Separate mapped candidate with once-per-user prediction eviction and reuse."""
from pathlib import Path
import os
import subprocess
import upstream_repro as u
import upstream_parallel_mapped as inherited

serial=inherited.serial
BASE=u.ROOT/'artifacts/upstream-grouped-storage'
TREE=BASE/'source'
REPORT=u.ROOT/'experiments/upstream-grouped-storage'


def prepare():
    inherited.BASE,inherited.TREE,inherited.REPORT=BASE,TREE,REPORT
    previous=inherited.prepare()
    tx=TREE/'src/tx.rs';source=tx.read_text()
    source=serial.replace_once(source,
        'fn predict_day(&self, u: usize, i: usize, day: i32, day_su: &[f32]) -> f32 {\n        self.storage_release_user(u);',
        'fn predict_day(&self, u: usize, i: usize, day: i32, day_su: &[f32]) -> f32 {')
    source=serial.replace_once(source,'fn predict(&self, u: usize, i: usize, day: i32) -> f32 {',
        'fn predict(&self, u: usize, i: usize, day: i32) -> f32 {\n        self.storage_release_user(u);')
    source=serial.replace_once(source,'fn groups_by_user(&self) -> bool { self.cfg.lazy_day }',
        'fn groups_by_user(&self) -> bool { true }')
    end='''            *o = self.predict_day(u, ds.item_idxs[idx] as usize, day as i32, day_su);
        }
    }
'''
    source=serial.replace_once(source,end,end[:-6]+'        self.storage_release_user(u);\n    }\n')
    tx.write_text(source)
    parallel=TREE/'src/tsvdx4p.rs';source=parallel.read_text()
    anchor='''    fn predict(&self, u: usize, i: usize, day: i32) -> f32 {
        self.inner.predict(u, i, day)
    }
'''
    source=serial.replace_once(source,anchor,anchor+'''
    fn groups_by_user(&self) -> bool { true }
    fn predict_user_rows(&self, ds: &Dataset, u: usize, rows: &[usize], out: &mut [f32]) {
        self.inner.predict_user_rows(ds,u,rows,out);
    }
''')
    parallel.write_text(source)
    proof=TREE/'src/bin/tsvdx4p-storage-parity.rs';source=proof.read_text()
    source=serial.replace_once(source,'    for ds in [tr,pr]{','''    for ds in [tr,pr]{
        assert_eq!(netflix_prize::gen_preds_parallel(a,ds).mapv(f32::to_bits),
                   netflix_prize::gen_preds_parallel(b,ds).mapv(f32::to_bits),
                   "actual library row/grouped generation {case} epoch {epoch}");''')
    proof.write_text(source)
    patch=subprocess.check_output(['git','diff','--','src/tx.rs','src/tsvdx4p.rs','src/lib.rs','Cargo.toml'],cwd=TREE)
    (BASE/'parallel-storage.patch').write_bytes(patch)
    prepared=dict(previous,generated={name:u.sha(TREE/name) for name in previous['generated']},
        patch_sha256=u.sha(BASE/'parallel-storage.patch'),script_sha256=u.sha(Path(__file__)),
        inherited_preparation=previous,
        strategy='Prediction traversal and eviction placement only. Both models use the existing Tx per-user predictor; each grouped user releases mapped arrays once. Direct prediction and component APIs retain eviction. All training update/configuration source is unchanged.',
        scope='Separate candidate. Requires new arithmetic and actual prediction-library parity; no current training tree changed.')
    u.write(BASE/'parallel-preparation.json',prepared);return prepared


def verify():
    prepared=prepare();env=u.environment()
    env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_BUILD_JOBS='2',CARGO_NET_OFFLINE='true')
    with (REPORT/'build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
            '--bin','tsvdx4p-storage-parity','--bin','tx-storage-parity'],cwd=TREE,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    records=[];reference=None;executable=BASE/'target/release/tsvdx4p-storage-parity'
    for workers in [2,4]:
        for mapped in [False,True]:
            name=('mapped' if mapped else 'heap')+'-workers'+str(workers)
            arm={k:v for k,v in env.items() if k not in ['TX_MMAP_DIR','TX_LAZY_DAY']}
            arm['RAYON_NUM_THREADS']=str(workers);scratch=BASE/('scratch-'+name)
            if mapped:arm['TX_MMAP_DIR']=str(scratch)
            child=subprocess.run([str(executable)],cwd=TREE,env=arm,capture_output=True,text=True,timeout=600)
            path=REPORT/(name+'.log');path.write_text(child.stdout+child.stderr)
            signatures=[line for line in child.stdout.splitlines() if line.startswith('CASE ')]
            if reference is None and child.returncode==0:reference=signatures
            same=signatures==reference and len(signatures)==24
            records.append(dict(arm=name,exit_code=child.returncode,log_sha256=u.sha(path),signatures_match_across_storage_and_workers=same))
            if list(scratch.glob('tx-*.f32')):raise ValueError('Candidate mapping files survived exit')
            print(name,child.returncode,same,flush=True)
    serial_records=[];serial_executable=BASE/'target/release/tx-storage-parity'
    for mapped in [False,True]:
        for lazy in [False,True]:
            name=('mapped' if mapped else 'heap')+('-lazy' if lazy else '-cached')
            arm={k:v for k,v in env.items() if k not in ['TX_MMAP_DIR','TX_LAZY_DAY']}
            scratch=BASE/('serial-'+name)
            if mapped:arm['TX_MMAP_DIR']=str(scratch)
            if lazy:arm['TX_LAZY_DAY']='1'
            child=subprocess.run([str(serial_executable)],cwd=TREE,env=arm,capture_output=True,text=True,timeout=600)
            path=REPORT/('serial-'+name+'.log');path.write_text(child.stdout+child.stderr)
            if list(scratch.glob('tx-*.f32')):raise ValueError('Serial mapping files survived exit')
            serial_records.append(dict(arm=name,exit_code=child.returncode,log_sha256=u.sha(path)))
            print('serial',name,child.returncode,flush=True)
    if any(u.sha(TREE/name)!=digest for name,digest in prepared['generated'].items()):raise ValueError('Candidate changed during proof')
    passed=all(r['exit_code']==0 and r['signatures_match_across_storage_and_workers'] for r in records) and all(r['exit_code']==0 for r in serial_records)
    u.write(REPORT/'verification.json',dict(source_pin=u.PIN,preparation=prepared,executable_sha256=u.sha(executable),
        serial_executable_sha256=u.sha(serial_executable),cargo_lock_sha256=u.sha(TREE/'Cargo.lock'),
        arms=records,serial_arms=serial_records,pass_all=passed,
        scope='Parallel24 cases x4 modes through16000 factors/20 epochs, all parameter/cache bits, direct/grouped/component predictions and actual library row-versus-grouped generation. Serial288 cases x4 modes through4800 factors/5 epochs plus72MB mapped persistence. Synthetic only; no speedup, full-data resource or goal-quality claim.'))
    if not passed:raise ValueError('Grouped candidate failed; preserve artifacts')


if __name__=='__main__':verify()
