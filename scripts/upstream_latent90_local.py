#!/usr/bin/env python3
"""Isolated, memory-admitted reproduction of the original 90 latent supports."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
import upstream_repro as u
import upstream_cf96_memory_resume as memory

RUN=u.ROOT/'artifacts/upstream-latent90-local'
TREE=RUN/'source'
REPORT=u.ROOT/'experiments/upstream-latent90-local'
REGISTRY=u.ROOT/'experiments/upstream-all-pipeline-coverage/registry.json'
JOBS={
    'rbm50lat': ('rbm',dict(EPOCHS='20',H='50',UFEAT='1')),
    'tx20lat': ('tx',dict(EPOCHS='20',F='20',IFEAT='1',LR_W='0.002',NBR='25',UFEAT='1',WF='0.8')),
}

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def prepare():
    path=REPORT/'plan.json'
    if path.exists():
        plan=u.read(path)
        for name,digest in plan['bound_files'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
        return plan
    if RUN.exists() or REPORT.exists():raise ValueError('Partial setup preserved')
    u.validate_fresh()
    coverage=u.read(REGISTRY.parent/'coverage.json')
    if coverage['registry_sha256']!=u.sha(REGISTRY):raise ValueError('Registry changed')
    registry=u.read(REGISTRY)['new']['jobs']
    for name,(kind,params) in JOBS.items():
        expected='cargo build --release --bin fit && RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 '
        expected+=' '.join(k+'='+v for k,v in params.items())
        expected+=f' ./target/release/fit {kind} {name} -p pipeline-new.toml --final --preds preds_new '
        if registry[name]['command']!=expected:raise ValueError('Original recipe mismatch')
    RUN.mkdir();REPORT.mkdir();TREE.mkdir()
    bound={}
    def bind(p):bound[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in coverage['source_tracked_files_sha256'].items():
        if name.startswith(('data/','preds_')):continue
        source=u.SOURCE/name
        if u.sha(source)!=digest:raise ValueError('Source changed: '+name)
        target=TREE/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);bind(target)
    env=u.environment()
    command=[shutil.which('cargo',path=env['PATH']),'build','--release','--locked','--features','blas','--bin','predfeat']
    with (REPORT/'build.log').open('w') as log:
        subprocess.run(command,cwd=u.SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=u.BASE/'build-receipt.json';build=u.read(receipt)
    if build['source_commit']!=u.PIN or build['binaries']['fit']!=u.sha(u.BASE/'target/release/fit'):
        raise ValueError('Fit binary not bound to original source')
    (TREE/'target/release').mkdir(parents=True)
    for name in ['fit','predfeat']:
        target=TREE/'target/release'/name
        shutil.copy2(u.BASE/'target/release'/name,target);bind(target)
    (TREE/'data').mkdir()
    for split in ['trainx','probex','fulltrain','qual']:
        original=u.FRESH/'data'/split
        (TREE/'data'/split).symlink_to(original,target_is_directory=True)
        for p in original.glob('*.npy'):bind(p)
    if np.any(np.load(TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    for p in [Path(__file__),Path(u.__file__),Path(memory.__file__),REGISTRY,REGISTRY.parent/'coverage.json',receipt]:bind(p)
    plan=dict(created=stamp(),source_pin=u.PIN,bound_files=bound,
        original_registry_jobs={name:registry[name] for name in JOBS},
        predfeat_build_command=command,build_rustflags=env.get('RUSTFLAGS'),
        admission='Two stable samples, all native and meta-fitting physical footprints below 16 GiB; compressor below 5 GiB; swap growth below 32 MiB per 30 seconds. Models serialized. No active job interrupted.',
        support_semantics='Unchanged predfeat uses trainx user/item factors for BOTH query sets, as in original source. Fulltrain factors preserved separately.',
        public_prediction_values_read=False,qualifying_labels_read=False)
    u.write(path,plan);return plan

def footprints():
    result=memory.footprints()
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        pid,cmd=row.strip().split(None,1)
        if pid in result or 'upstream_' not in cmd or '--worker' not in cmd.split():continue
        output=subprocess.run(['vmmap','-summary',pid],capture_output=True,text=True).stdout
        match=re.search(r'^Physical footprint:\s+([\d.]+)([KMG])',output,re.M)
        if not match:
            if subprocess.run(['ps','-p',pid,'-o','pid='],capture_output=True,text=True).stdout.strip():
                raise RuntimeError('Live worker footprint unavailable: '+pid)
            continue
        result[pid]=dict(bytes=float(match[1])*{'K':2**10,'M':2**20,'G':2**30}[match[2]],command=cmd)
    return result

def admitted(previous,current,total,free):
    return (total<16*2**30 and current['compressor_bytes']<5*2**30 and
        current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>24*2**30)

def wait_memory(name):
    previous=memory.memory();good=0
    while good<2:
        current=memory.memory();workers=footprints();disk=os.statvfs(u.ROOT)
        good=good+1 if admitted(previous,current,sum(r['bytes'] for r in workers.values()),disk.f_bavail*disk.f_frsize) else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_memory',job=name,observed=stamp(),
            good_samples=good,memory=current,workers=workers))
        if good<2:previous=current;time.sleep(30)

def model_environment(params):
    env=u.environment()
    # Whitelist process/toolchain controls: source model knobs cannot leak in.
    keep={'PATH','HOME','TMPDIR','USER','LOGNAME','SHELL','LANG','LC_ALL','CARGO_HOME','RUSTUP_HOME','CARGO_TARGET_DIR','RUSTFLAGS','CARGO_BUILD_JOBS','DYLD_LIBRARY_PATH','SDKROOT','MACOSX_DEPLOYMENT_TARGET'}
    env={k:v for k,v in env.items() if k in keep}
    env.update(RAYON_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',VECLIB_MAXIMUM_THREADS='1',**params)
    return env

def run():
    prepare()
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,(kind,params) in JOBS.items():
            receipt=REPORT/(name+'-receipt.json')
            if receipt.exists():
                for rel,digest in u.read(receipt)['outputs'].items():
                    if u.sha(TREE/rel)!=digest:raise ValueError('Completed output changed')
                continue
            out=TREE/'preds_new'
            if out.exists() and list(out.glob(name+'.*')):raise ValueError('Partial native outputs preserved; no automatic restart')
            wait_memory(name);prepare()
            command=['./target/release/fit',kind,name,'-p','pipeline-new.toml','--final','--preds','preds_new']
            started=stamp();logpath=REPORT/(name+'.log');env=model_environment(params)
            with logpath.open('x') as log:
                child=subprocess.Popen(command,cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT)
                u.write(REPORT/'status.json',dict(state='native_running',job=name,pid=child.pid,command=command,started=started))
                code=child.wait()
            if code:raise RuntimeError('Native failure preserved: '+str(code))
            shutil.copyfile(logpath,out/(name+'.out'))
            for split,count in [('probex',1128164),('qual',2817131)]:
                a=np.load(out/f'{name}.{split}.npy',mmap_mode='r')
                if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid prediction')
            for split in ['trainx','fulltrain']:
                for factor,rows,width in ([('ufeat',480189,50)] if kind=='rbm' else [('ufeat',480189,20),('ifeat',17770,20)]):
                    a=np.load(out/f'{name}.{factor}.{split}.npy',mmap_mode='r')
                    if a.shape!=(rows,width) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid factor matrix')
            command2=['./target/release/predfeat','-p','pipeline-new.toml','--latent-users',name]
            if kind=='tx':command2+=['--latent-items',name]
            with (REPORT/(name+'-predfeat.log')).open('x') as log:
                subprocess.run(command2,cwd=TREE,env=model_environment({}),stdout=log,stderr=subprocess.STDOUT,check=True)
            outputs={str(p.relative_to(TREE)):u.sha(p) for p in out.glob(name+'.*') if p.is_file()}
            supports={str(p.relative_to(TREE)):u.sha(p) for p in (out/'lat').glob(name+'.*.npy')}
            if len(supports)!=(100 if kind=='rbm' else 80):raise ValueError('Support count mismatch')
            for rel in supports:
                a=np.load(TREE/rel,mmap_mode='r');count=1128164 if '.probex.' in rel else 2817131
                if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid support')
            outputs.update(supports)
            u.write(receipt,dict(state='complete_pending_independent_verification',started=started,finished=stamp(),
                plan_sha256=u.sha(REPORT/'plan.json'),command=command,model_environment=params,
                outputs=outputs,qualifying_labels_read=False,public_prediction_values_read=False))
        u.write(REPORT/'status.json',dict(state='complete_pending_independent_verification',observed=stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    else:run()
