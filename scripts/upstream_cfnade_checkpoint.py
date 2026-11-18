#!/usr/bin/env python3
"""Add exact, identity-bound optimizer/RNG checkpoints to the proven CF candidate."""
import os
import re
import shutil
import subprocess
import uuid
import tomli
from pathlib import Path
import upstream_repro as u
import upstream_cfnade_fast_v2 as second

BASE=u.ROOT/'artifacts/upstream-cfnade-checkpoint'
TREE=BASE/'source'
REPORT=u.ROOT/'experiments/upstream-cfnade-checkpoint'


def prepare():
    second.BASE,second.TREE,second.REPORT=BASE,TREE,REPORT
    second.prepare();once=second.first.once
    text=(TREE/'src/cfnade.rs').read_text()
    text=once(text,'use rand::{Rng, SeedableRng, prelude::SliceRandom, rngs::StdRng};',
        'use rand::{Rng, SeedableRng, prelude::SliceRandom};\nuse rand_chacha::ChaCha12Rng as StdRng;')
    text=once(text,'pub struct CfNadeModel {','''pub struct CfNadeModel {
    checkpoint_directory: Option<std::path::PathBuf>,
    checkpoint_lock: Option<std::fs::File>,
    checkpoint_identity: serde_json::Value,
    checkpoint_loaded_epoch: usize,''')
    start=text.index('    fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {')
    end=text.index('    /// Number of training epochs',start)
    constructor=text[start:end]
    constructor=once(constructor,'        Self {','''        let mut model=Self {
            checkpoint_directory:None,checkpoint_lock:None,
            checkpoint_identity:serde_json::Value::Null,checkpoint_loaded_epoch:0,''')
    constructor=once(constructor,'            swapped_in: false,\n        }\n    }',
        '            swapped_in: false,\n        };\n        model.checkpoint_open(tr,pr);\n        model\n    }')
    text=text[:start]+constructor+text[end:]
    hook='''    fn resumed_epoch(&self)->usize{self.checkpoint_loaded_epoch}
    fn save_epoch_checkpoint(&mut self,epoch:usize){self.checkpoint_save(epoch);}
    fn restore_epoch_prediction(&self,ds:&Dataset,model:&str,preds:&str,save_each:bool){
        let epoch=self.checkpoint_loaded_epoch;
        if save_each && epoch>0{
            let values=crate::gen_preds_parallel(self,ds);
            ndarray_npy::write_npy(format!("{preds}/{model}_ep{epoch:02}.{}.npy",ds.name),&values).unwrap();
        }
    }

'''
    text=once(text,'    /// Runs one epoch: batched SGD over all users, then updates (and finally swaps in) the SWA model.',
        hook+'    /// Runs one epoch: batched SGD over all users, then updates (and finally swaps in) the SWA model.')
    original=(TREE/'src/cfnade_original.rs').read_text()
    fields=re.findall(r'^    (\w+): (Array[123]<f32>)',original.split('pub struct CfNadeModel {',1)[1].split('\n}',1)[0],re.M)
    io='\nimpl CfNadeModel {\nfn checkpoint_write_arrays(&self,root:&std::path::Path)->serde_json::Map<String,serde_json::Value>{\nlet mut files=serde_json::Map::new();\n'
    for name,_ in fields:
        io+=f'let path=root.join("{name}.npy");ndarray_npy::write_npy(&path,&self.{name}).unwrap();checkpoint_sync(&path);files.insert("{name}.npy".into(),serde_json::json!(checkpoint_md5(&path)));\n'
    io+='files\n}\nfn checkpoint_read_arrays(&mut self,root:&std::path::Path,files:&serde_json::Map<String,serde_json::Value>){\n'
    io+=f'assert_eq!(files.len(),{len(fields)});\n'
    for name,ty in fields:
        io+=f'let path=root.join("{name}.npy");assert_eq!(checkpoint_md5(&path),files["{name}.npy"].as_str().unwrap(),"checkpoint tensor checksum: {name}");let value:{ty}=ndarray_npy::read_npy(&path).unwrap();assert_eq!(value.raw_dim(),self.{name}.raw_dim(),"checkpoint tensor shape: {name}");self.{name}=value;\n'
    io+='}\n}\n'
    text+='\ninclude!("cfnade_checkpoint.rs");\n'+io
    (TREE/'src/cfnade.rs').write_text(text)
    shutil.copyfile(u.ROOT/'research/cfnade_fast/checkpoint.rs',TREE/'src/cfnade_checkpoint.rs')
    lib=(TREE/'src/lib.rs').read_text()
    lib=once(lib,'    fn n_epochs(&self) -> usize { 0 }','''    fn n_epochs(&self) -> usize { 0 }
    fn resumed_epoch(&self)->usize{0}
    fn save_epoch_checkpoint(&mut self,_epoch:usize){}
    fn restore_epoch_prediction(&self,_ds:&Dataset,_model:&str,_preds:&str,_save_each:bool){}''')
    # Both pinned drivers initialize M, then run one complete epoch and its outputs.
    needle='let n_epochs = model.n_epochs();'
    if lib.count(needle)!=2:raise ValueError('Unexpected native training drivers')
    lib=lib.replace(needle,needle+'\n    model.restore_epoch_prediction(&pr,model_name,preds_dir,save_probe_each_epoch);')
    starts=[m.start() for m in re.finditer(re.escape('for epoch in 1..=n_epochs {'),lib)]
    if len(starts)!=2:raise ValueError('Unexpected native epoch loops')
    for begin in reversed(starts):
        brace=lib.index('{',begin);depth=1;cursor=brace+1
        while depth:
            depth+=(lib[cursor]=='{')-(lib[cursor]=='}');cursor+=1
        lib=lib[:cursor-1]+'    model.save_epoch_checkpoint(epoch);\n'+lib[cursor-1:]
        lib=lib[:begin]+lib[begin:].replace('for epoch in 1..=n_epochs {','for epoch in (model.resumed_epoch()+1)..=n_epochs {',1)
    (TREE/'src/lib.rs').write_text(lib)
    cargo=subprocess.check_output(['git','show',u.PIN+':Cargo.toml'],cwd=u.SOURCE,text=True)
    cargo=cargo.replace('[dependencies]','[dependencies]\nrand_chacha = "0.9.0"\nlibc = "0.2"')
    (TREE/'Cargo.toml').write_text(cargo)
    shutil.copyfile(u.ROOT/'research/cfnade_fast/checkpoint_proof.rs',TREE/'src/bin/cfnade-checkpoint-proof.rs')
    patch=subprocess.check_output(['git','diff','--','src/cfnade.rs','src/lib.rs','Cargo.toml'],cwd=TREE)
    (REPORT/'candidate.patch').write_bytes(patch)
    record=dict(source_pin=u.PIN,script_sha256=u.sha(Path(__file__)),base_script_sha256=u.sha(Path(second.__file__)),
        patch_sha256=u.sha(REPORT/'candidate.patch'),files={p:u.sha(TREE/p) for p in
            ['src/cfnade.rs','src/cfnade_original.rs','src/cfnade_kernels.rs','src/cfnade_checkpoint.rs',
             'src/lib.rs','Cargo.toml','src/bin/cfnade-fast-parity.rs','src/bin/cfnade-checkpoint-proof.rs']},
        scope='Exact arithmetic, explicit same-ChaCha12 RNG stream, per-phase writer locks, identity/hash/shape checks, atomic bounded state checkpoints.')
    u.write(REPORT/'preparation.json',record);return record


def verify():
    record=prepare();env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_BUILD_JOBS='2')
    for key in ['CF_CHECKPOINT_DIR','CF_CHECKPOINT_SOURCE','CF_CHECKPOINT_DATA','CF_EXIT_AFTER_CHECKPOINT']:env.pop(key,None)
    with (REPORT/'build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
            '--bin','cfnade-fast-parity','--bin','cfnade-checkpoint-proof'],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    original_lock=tomli.loads(subprocess.check_output(['git','show',u.PIN+':Cargo.lock'],cwd=u.SOURCE,text=True))
    actual_lock=tomli.loads((TREE/'Cargo.lock').read_text())
    package=next(p for p in actual_lock['package'] if p['name']=='netflix_prize')
    for name in ['libc','rand_chacha']:package['dependencies'].remove(name)
    if actual_lock!=original_lock:raise ValueError('Unexpected dependency version changes')
    record['cargo_lock_sha256']=u.sha(TREE/'Cargo.lock');u.write(REPORT/'preparation.json',record)
    results=[]
    def execute(binary,args,name,extra=None,expect=0,cwd=TREE,contains=None):
        with (REPORT/(name+'.log')).open('w') as log:
            p=subprocess.run([str(BASE/'target/release'/binary)]+args,cwd=cwd,env=dict(env,**(extra or {})),stdout=log,stderr=subprocess.STDOUT)
        passed=p.returncode==expect if expect in [0,73] else p.returncode!=0
        if contains is not None:passed=passed and contains in (REPORT/(name+'.log')).read_text()
        results.append(dict(name=name,exit_code=p.returncode,expected=expect,pass_check=passed,log_sha256=u.sha(REPORT/(name+'.log'))))
        if not passed:raise ValueError('Checkpoint stage failed: '+name)
    execute('cfnade-fast-parity',[],'parity')
    directories=[]
    for interrupted in [3,5]:
        scratch=BASE/('checkpoint-proof-'+uuid.uuid4().hex);scratch.mkdir();directories.append(str(scratch.relative_to(u.ROOT)))
        extra=dict(CF_CHECKPOINT_DIR=str(scratch/'state'),CF_CHECKPOINT_SOURCE=u.sha(REPORT/'preparation.json'),
                   CF_CHECKPOINT_DATA=u.sha(TREE/'src/bin/cfnade-checkpoint-proof.rs'),CF_PROOF_OUTPUT=str(scratch),CF_PROOF_INTERRUPT=str(interrupted))
        execute('cfnade-checkpoint-proof',['create'],f'create-{interrupted}',dict(extra,CF_EXIT_AFTER_CHECKPOINT=str(interrupted)),expect=73,cwd=scratch)
        execute('cfnade-checkpoint-proof',['resume'],f'resume-{interrupted}',extra,cwd=scratch)
        execute('cfnade-checkpoint-proof',['load'],f'identity-rejected-{interrupted}',dict(extra,CF_CHECKPOINT_DATA='0'*64),expect=1,cwd=scratch,contains='checkpoint identity differs')
        latest=u.read(scratch/'state/trainx/latest.json')['epoch']
        damaged=scratch/'state/trainx'/f'epoch-{latest:05}'/'enc_proj.npy'
        with damaged.open('r+b') as stream:
            stream.seek(-1,2);value=stream.read(1);stream.seek(-1,2);stream.write(bytes([value[0]^1]))
        execute('cfnade-checkpoint-proof',['load'],f'tamper-rejected-{interrupted}',extra,expect=1,cwd=scratch,contains='checkpoint tensor checksum: enc_proj')
    result=dict(pass_all=all(r['pass_check'] for r in results),preparation=record,checks=results,
        binaries={name:u.sha(BASE/'target/release'/name) for name in ['cfnade-fast-parity','cfnade-checkpoint-proof']},
        checkpoint_proof_directories=directories)
    u.write(REPORT/'verification.json',result)
    if not result['pass_all']:raise ValueError('Checkpoint/parity proof failed')


if __name__=='__main__':verify()
