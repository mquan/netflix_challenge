#!/usr/bin/env python3
"""Train the additional published CFNADE48 preset with guarded independent phases."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import numpy as np
import tomli
import upstream_repro as u
import upstream_cfnade_sparse_rows as candidate
from upstream_full_reproduce import group_rss,stop

BASE,TREE=candidate.BASE,candidate.TREE
REPORT=u.ROOT/'experiments/upstream-cfnade48'
CHECKPOINTS=BASE/'preset48-checkpoints'
OUT=TREE/'preds_preset48'
PHASES={'x':('trainx','probex'),'qual':('fulltrain','qual')}


def verify_backend():
    u.validate_fresh();proof=u.read(candidate.REPORT/'verification.json');prepared=u.read(candidate.REPORT/'preparation.json')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()!=u.PIN:
        raise ValueError('CF48 backend source pin changed')
    if not proof['pass_all'] or proof['preparation']!=prepared or prepared['script_sha256']!=u.sha(Path(candidate.__file__)):
        raise ValueError('CF48 requires the unchanged passing backend proof')
    for name,digest in prepared['files'].items():
        if u.sha(TREE/name)!=digest:raise ValueError('CF48 backend source changed: '+name)
    if u.sha(BASE/'target/release/cfnade-fast-parity')!=proof['executable_sha256']:
        raise ValueError('CF48 backend proof binary changed')
    original=tomli.loads(subprocess.check_output(['git','show',u.PIN+':Cargo.lock'],cwd=u.SOURCE,text=True))
    actual=tomli.loads((TREE/'Cargo.lock').read_text());package=next(p for p in actual['package'] if p['name']=='netflix_prize')
    for name in ['libc','rand_chacha']:package['dependencies'].remove(name)
    if actual!=original:raise ValueError('CF48 dependencies changed beyond existing checkpoint dependencies')
    inherited=u.ROOT/'experiments/upstream-cfnade-checkpoint'
    checkpoint_proof=u.read(inherited/'verification.json');checkpoint_prepared=u.read(inherited/'preparation.json')
    required={'parity'}|{f'{kind}-{e}' for e in [3,5] for kind in ['create','resume','identity-rejected','tamper-rejected']}
    if (not checkpoint_proof['pass_all'] or checkpoint_proof['preparation']!=checkpoint_prepared or
        {r['name'] for r in checkpoint_proof['checks']}!=required or not all(r['pass_check'] for r in checkpoint_proof['checks'])):
        raise ValueError('Original checkpoint proof incomplete')
    old=u.ROOT/'artifacts/upstream-cfnade-checkpoint/source'
    if u.sha(TREE/'src/cfnade_checkpoint.rs')!=u.sha(old/'src/cfnade_checkpoint.rs'):
        raise ValueError('CF48 checkpoint protocol differs')
    delimiter='include!("cfnade_checkpoint.rs");'
    if (TREE/'src/cfnade.rs').read_text().split(delimiter,1)[1]!=(old/'src/cfnade.rs').read_text().split(delimiter,1)[1]:
        raise ValueError('CF48 array serialization differs')
    if u.sha(old/'src/cfnade_checkpoint.rs')!=checkpoint_prepared['files']['src/cfnade_checkpoint.rs']:
        raise ValueError('Checkpoint proof support changed')
    if u.sha(old/'src/cfnade.rs')!=checkpoint_prepared['files']['src/cfnade.rs']:
        raise ValueError('Original checkpoint serialization source changed')
    for name in ['src/bin/cfnade-new.rs','pipeline-new.toml']:
        if (TREE/name).read_bytes()!=subprocess.check_output(['git','show',u.PIN+':'+name],cwd=u.SOURCE):
            raise ValueError('CF48 public dispatcher/pipeline differs')
    changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=TREE,text=True).splitlines())
    if changed-{'src/cfnade.rs','src/lib.rs','Cargo.toml','Cargo.lock','data/qual_ratings/qual_ratings.csv.gz'}:
        raise ValueError('Unexpected CF48 tracked source changes')
    if u.sha(TREE/'data/qual_ratings/qual_ratings.csv.gz')!=u.sha(u.FRESH/'data/qual_ratings/qual_ratings.csv.gz'):
        raise ValueError('CF48 qualifying sentinel changed')
    return prepared


def layout():
    for split in ['raw','train','probe','trainx','probex','fulltrain','qual']:
        path=TREE/'data'/split;target=u.FRESH/'data'/split
        if not path.is_symlink():
            if path.exists():raise ValueError('Unowned CF48 native data directory')
            path.symlink_to(target,target_is_directory=True)
        if path.resolve()!=target.resolve():raise ValueError('Incorrect CF48 data alias')
    OUT.mkdir(exist_ok=True)


def prepare():
    if (REPORT/'plan.json').exists():return verify_plan()
    verify_backend();layout();REPORT.mkdir(parents=True,exist_ok=True)
    dispatcher=(TREE/'src/bin/cfnade-new.rs').read_text()
    if not re.search(r'"cfnade-48"\s*=>\s*run_preset\(job_name,\s*160,\s*48,\s*95\)',dispatcher):
        raise ValueError('Pinned CF48 preset not found')
    driver=TREE/'src/bin/cfnade48-stage.rs'
    driver.write_text('''use netflix_prize::{fit,LOG_FILE};
use netflix_prize::cfnade::{CfNadeConfig,CfNadeModel};
use std::io::BufWriter;
fn main(){
    netflix_prize::suppress_progress();
    let cfg=CfNadeConfig{n_hidden:160,rank:48,n_epochs:95,..CfNadeConfig::default()};
    let reference={let n_hidden=160;let rank=48;let n_epochs=95;let swa_start=CfNadeConfig::default().swa_start;
        CfNadeConfig{n_hidden,rank,n_epochs,swa_start,..CfNadeConfig::default()}};
    assert_eq!(format!("{:?}",cfg),format!("{:?}",reference));
    let phase=std::env::args().nth(1).expect("x, qual or --describe");
    if phase=="--describe"{println!("{:?}",cfg);return;}
    let (train,query)=match phase.as_str(){"x"=>("trainx","probex"),"qual"=>("fulltrain","qual"),_=>panic!("bad phase")};
    std::fs::create_dir_all("preds_preset48").unwrap();
    *LOG_FILE.lock().unwrap()=Some(BufWriter::new(std::fs::File::create(format!("preds_preset48/cfnade-48.phase-{query}.out")).unwrap()));
    netflix_prize::teeln!("[published cfnade-48 independent phase] {:?}",cfg);
    fit::<CfNadeModel>(cfg,"rtg",train,query,"cfnade-48",false,false,true,"preds_preset48",false,false);
    LOG_FILE.lock().unwrap().take();
}
''')
    specific=TREE/'src/bin/cfnade48-parity.rs'
    text=(TREE/'src/bin/cfnade-fast-parity.rs').read_text();anchor='[(4,3),(40,12),(320,96)]'
    if text.count(anchor)!=1:raise ValueError('Unexpected CF48 parity source')
    specific.write_text(text.replace(anchor,'[(160,48)]'))
    env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_NET_OFFLINE='true',CARGO_BUILD_JOBS='2',RAYON_NUM_THREADS='2')
    for key in ['CF_CHECKPOINT_DIR','CF_CHECKPOINT_SOURCE','CF_CHECKPOINT_DATA','CF_EXIT_AFTER_CHECKPOINT','CF_PROOF_OUTPUT','CF_PROOF_INTERRUPT']:env.pop(key,None)
    with (REPORT/'build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
            '--bin','cfnade48-stage','--bin','cfnade48-parity'],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (REPORT/'preset-parity.log').open('w') as log:
        subprocess.run([str(BASE/'target/release/cfnade48-parity')],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    description=subprocess.check_output([str(BASE/'target/release/cfnade48-stage'),'--describe'],cwd=TREE,env=env,text=True).strip()
    if not all(value in description for value in ['n_hidden: 160','rank: 48','n_epochs: 95','swa_start: 18']):raise ValueError('CF48 runtime configuration differs')
    verify_backend()
    inputs={str(p.relative_to(u.ROOT)):u.sha(p) for split in ['trainx','probex','fulltrain','qual']
        for p in (u.FRESH/'data'/split).glob('*.npy')}
    data_id=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    source_files={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(candidate.__file__),
        candidate.REPORT/'preparation.json',candidate.REPORT/'verification.json',TREE/'Cargo.lock',driver,specific,
        TREE/'src/bin/cfnade-new.rs',TREE/'pipeline-new.toml',u.BASE/'registry.json',
        u.ROOT/'experiments/upstream-cfnade-checkpoint/verification.json',
        u.ROOT/'experiments/upstream-cfnade-checkpoint/preparation.json']}
    source_files.update({str((TREE/name).relative_to(u.ROOT)):digest for name,digest in verify_backend()['files'].items()})
    binaries={str((BASE/'target/release'/name).relative_to(u.ROOT)):u.sha(BASE/'target/release'/name) for name in ['cfnade48-stage','cfnade48-parity']}
    if list(OUT.glob('cfnade-48*')) or (CHECKPOINTS.exists() and list(CHECKPOINTS.iterdir())):
        raise ValueError('CF48 outputs/checkpoints without an owned plan')
    source_id=hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()
    plan=dict(source_pin=u.PIN,job='cfnade-48',hidden=160,rank=48,epochs_per_phase=95,description=description,
        canonical_reference_command=u.read(u.BASE/'registry.json')['jobs']['cfnade-48']['command'],
        execution_commands={p:['nice','-n','10',str(BASE/'target/release/cfnade48-stage'),p] for p in PHASES},
        source_id=source_id,data_id=data_id,source_sha256=source_files,input_sha256=inputs,binaries=binaries,
        preset_parity_log_sha256=u.sha(REPORT/'preset-parity.log'),qualifying_labels_read=False,
        semantics='Additional original160/48/95 public preset, independently initialized source fit phases. Existing320/96/95 jobs continue. No pretrained features or qualifying outcomes in learning.')
    u.write(REPORT/'plan.json',plan);return plan


def verify_plan():
    plan=u.read(REPORT/'plan.json')
    for section in ['source_sha256','input_sha256','binaries']:
        for name,digest in plan[section].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen CF48 plan input changed: '+name)
    return plan


def epoch(phase):
    p=CHECKPOINTS/PHASES[phase][0]/'latest.json'
    return u.read(p)['epoch'] if p.exists() else 0


def other_rss(exclude):
    rows=[line.split(None,3) for line in subprocess.check_output(['ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines()]
    groups={int(r[1]) for r in rows if len(r)==4 and ('/target/release/' in r[3].split()[0] or str(u.ROOT/'.venv/bin/python') in r[3])}
    groups.difference_update(exclude);groups.discard(os.getpgrp())
    return sum(int(r[2])*1024 for r in rows if len(r)==4 and int(r[1]) in groups)


def admission(other,reservations,free):
    return other+sum(reservations)+3*2**30<24*2**30 and free>28*2**30


def phase_complete(phase):
    p=REPORT/(phase+'-complete.json')
    if not p.exists():return False
    record=u.read(p);query=PHASES[phase][1];terminal=OUT/f'cfnade-48.{query}.npy'
    if (record['plan_sha256']!=u.sha(REPORT/'plan.json') or record['epochs']!=95 or epoch(phase)!=95 or
        len(record['outputs'])!=96 or record['outputs'].get(str(terminal.relative_to(TREE)))!=u.sha(terminal)):
        raise ValueError('CF48 completed phase marker changed or incomplete')
    return True


def finalize(phase,plan):
    if epoch(phase)!=95:raise ValueError('CF48 child exited before original95 epochs')
    train,query=PHASES[phase];pointer=u.read(CHECKPOINTS/train/'latest.json')
    state_path=CHECKPOINTS/train/'epoch-00095/state.json'
    if u.sha(state_path,'md5')!=pointer['manifest_md5']:raise ValueError('Final CF48 checkpoint manifest changed')
    state=u.read(state_path)
    if (state['epoch']!=95 or state['identity']['train']!=train or state['identity']['query']!=query or
        state['identity']['source']!=plan['source_id'] or state['identity']['data']!=plan['data_id'] or
        state['identity']['config']!=plan['description'] or len(state['files'])!=84):
        raise ValueError('Final CF48 checkpoint identity differs')
    for name,digest in state['files'].items():
        if u.sha(state_path.parent/name,'md5')!=digest:raise ValueError('Final CF48 tensor checksum differs')
    rows=1128164 if phase=='x' else 2817131;outputs={}
    for suffix in ['']+[f'_ep{e:02}' for e in range(1,96)]:
        p=OUT/f'cfnade-48{suffix}.{query}.npy';a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.shape!=(rows,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Invalid CF48 output')
        outputs[str(p.relative_to(TREE))]=u.sha(p)
    u.write(REPORT/(phase+'-complete.json'),dict(plan_sha256=u.sha(REPORT/'plan.json'),epochs=95,outputs=outputs,
        checkpoint_manifest_sha256=u.sha(state_path),all84_checkpoint_tensors_rehashed=True))


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=prepare()
        env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),RAYON_NUM_THREADS='2',
            CF_CHECKPOINT_DIR=str(CHECKPOINTS),CF_CHECKPOINT_SOURCE=plan['source_id'],CF_CHECKPOINT_DATA=plan['data_id'])
        for key in ['CF_EXIT_AFTER_CHECKPOINT','CF_PROOF_OUTPUT','CF_PROOF_INTERRUPT']:env.pop(key,None)
        active={};logs={};high=0
        try:
            while not all(phase_complete(p) for p in PHASES):
                for phase,child in list(active.items()):
                    if child.poll() is not None:
                        logs.pop(phase).close();active.pop(phase)
                        if child.returncode:raise RuntimeError('CF48 native phase failed; checkpoint preserved')
                        finalize(phase,plan)
                other=other_rss(set(active_child.pid for active_child in active.values()))
                rss={p:group_rss(child.pid) for p,child in active.items()}
                fs=os.statvfs(BASE);free=fs.f_bavail*fs.f_frsize
                for phase in PHASES:
                    if phase in active or phase_complete(phase):continue
                    if admission(other,list(rss.values()),free):
                        log=(REPORT/f'{phase}-attempt-{time.time_ns()}.log').open('w');logs[phase]=log
                        active[phase]=subprocess.Popen(plan['execution_commands'][phase],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        rss[phase]=3*2**30
                high=high+1 if other+sum(rss.values())>25*2**30 or free<16*2**30 else 0
                if high>=3 and active:
                    phase='qual' if 'qual' in active else next(iter(active));child=active.pop(phase)
                    stop(child);logs.pop(phase).close();high=0
                    if child.returncode==0:finalize(phase,plan)
                u.write(REPORT/'status.json',dict(state='training' if active else 'waiting_for_memory',epochs={p:epoch(p) for p in PHASES},
                    active={p:child.pid for p,child in active.items()},rss_bytes=rss,other_rss_bytes=other,disk_free_bytes=free,
                    heartbeat=dt.datetime.now(dt.timezone.utc).isoformat()))
                time.sleep(10)
            verify_plan();outputs={}
            for phase in PHASES:
                record=u.read(REPORT/(phase+'-complete.json'))
                if record['plan_sha256']!=u.sha(REPORT/'plan.json'):raise ValueError('CF48 completion recipe changed')
                for name,digest in record['outputs'].items():
                    if u.sha(TREE/name)!=digest:raise ValueError('CF48 completed output changed')
                    outputs[name]=digest
            u.write(REPORT/'native-receipt.json',dict(source_commit=u.PIN,job='cfnade-48',plan_sha256=u.sha(REPORT/'plan.json'),
                canonical_reference_command=plan['canonical_reference_command'],execution_commands=plan['execution_commands'],
                epochs_per_phase=95,outputs=outputs,qualifying_outcomes_zero=True,
                provenance='Fresh original public160/48/95 preset; independent source-driver phases with exact sparse-row backend and optimizer/RNG checkpoints. No second-hand trained predictions or qualifying outcome fitting.'))
            u.write(REPORT/'status.json',dict(state='complete'))
        except BaseException as error:
            for child in active.values():stop(child)
            for log in logs.values():log.close()
            u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:
        REPORT.mkdir(parents=True,exist_ok=True)
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
    else:run()
