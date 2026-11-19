#!/usr/bin/env python3
"""Original split3 transposed DNN, no-MF DNN and small NADE, serialized."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
import upstream_split3_pilot as pilot
import upstream_latent90_local as latent

u=pilot.u
RUN=u.ROOT/'artifacts/upstream-split3-diversity'
TREE=RUN/'source'
REPORT=u.ROOT/'experiments/upstream-split3-diversity'
JOBS={
 's3-dnn16t':('dnn',dict(D='16',EPOCHS='7',EPSAVE='1',NMF='192',TRANSPOSE='1')),
 's3-dnn32nm':('dnn',dict(D='32',EPOCHS='7',EPSAVE='1',NMF='0')),
 's3-nade12s':('nade',dict(EPOCHS='8',EPSAVE='1',H='40',RANK='12',SAVETR='1')),
}

def stamp():return pilot.stamp()

def prepare():
    path=REPORT/'plan.json'
    if path.exists():
        plan=u.read(path)
        for name,digest in plan['bindings'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
        return plan
    if RUN.exists() or REPORT.exists():raise ValueError('Partial setup preserved')
    pm=pilot.prepare();dp=pilot.REPORT/'data-verification.json';data=u.read(dp)
    cp=pilot.REPORT/'completion-verification.json';complete=u.read(cp)
    if (complete['state']!='complete_original_two_phase_split3_pilot' or
        not data['known_rows_absent_from_training'] or not complete['full_known_projection_recomputed']):
        raise ValueError('Complete verified original split3 data/projection required')
    registry=u.read(pilot.REGISTRY)['3']['jobs'];recipes={}
    for name,(kind,params) in JOBS.items():
        expected='cargo build --release --bin fit && RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 '
        expected+=' '.join(k+'='+v for k,v in params.items())
        expected+=f' ./target/release/fit {kind} {name} -p pipeline-3.toml --final --preds preds_3 '
        if registry[name]['command']!=expected:raise ValueError('Original recipe mismatch: '+name)
        recipes[name]=registry[name]
    RUN.mkdir();REPORT.mkdir();TREE.mkdir();bindings={}
    def bind(p,digest=None):
        actual=u.sha(p)
        if digest is not None and digest!=actual:raise ValueError('Source changed: '+str(p))
        bindings[str(p.relative_to(u.ROOT))]=actual
    for name,digest in pm['source_files_sha256'].items():
        src=pilot.TREE/name;bind(src,digest);dst=TREE/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst);bind(dst,digest)
    (TREE/'target/release').mkdir(parents=True)
    shutil.copy2(pilot.TREE/'target/release/fit',TREE/'target/release/fit')
    bind(TREE/'target/release/fit',pm['binary_sha256']['fit'])
    (TREE/'data').mkdir()
    for split in ['train_3','probe_3','fulltrain','qual']:
        directory=pilot.TREE/'data'/split
        (TREE/'data'/split).symlink_to(directory.resolve(),target_is_directory=True)
        for p in directory.glob('*.npy'):bind(p)
    for name,digest in data['data_sha256'].items():bind(pilot.RUN/name,digest)
    bind(pilot.RUN/'probex-row-map.npy',data['row_map_sha256'])
    for p in [Path(__file__),Path(latent.__file__),Path(pilot.__file__),Path(u.__file__),Path(latent.memory.__file__),
              pilot.REGISTRY,pilot.REPORT/'plan.json',dp,cp]:bind(p)
    if np.any(np.load(TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets nonzero')
    plan=dict(created=stamp(),source_pin=u.PIN,bindings=bindings,original_registry_jobs=recipes,
        original_model_environments={k:v[1] for k,v in JOBS.items()},
        admission='Serialize after the existing local RBM50/Tx20lat controller exits naturally. Two stable samples; all native/meta physical footprints below20GiB; compressor below5GiB; swap growth below32MiB/30s; disk above20GiB. Reserve4GiB for original small native model. No active training interrupted.',
        fixed_published_outputs=['s3-dnn16t','s3-dnn32nm','s3-nade12s','s3-nade12s_ep01'],
        isolation='Separate pinned-source/output tree; reuse verified own split3 data and exact known-row map. Full original two-phase recipes. No numerical public features or qualifying labels.')
    u.write(path,plan);return plan

def owner_live(pid):
    text=subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],capture_output=True,text=True).stdout.strip()
    return bool(text and not text.startswith('Z') and 'upstream_latent90_local.py' in text)

def wait_memory():
    previous=latent.memory.memory();good=0
    while good<2:
        current=latent.memory.memory();workers=latent.footprints();free=shutil.disk_usage(u.ROOT).free
        waiting=owner_live(70300)
        admitted=(not waiting and sum(r['bytes'] for r in workers.values())<20*2**30 and
            current['compressor_bytes']<5*2**30 and current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>20*2**30)
        good=good+1 if admitted else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_latent_lane_and_memory',observed=stamp(),
            latent_owner_live=waiting,workers=workers,memory=current,good_samples=good,disk_free_bytes=free))
        if good<2:previous=current;time.sleep(30)

def run():
    plan=prepare()
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        mapping=np.load(pilot.RUN/'probex-row-map.npy')
        for name,(kind,params) in JOBS.items():
            receipt=REPORT/(name+'-receipt.json')
            if receipt.exists():
                for rel,digest in u.read(receipt)['outputs'].items():
                    if u.sha(TREE/rel)!=digest:raise ValueError('Completed output changed')
                continue
            output=TREE/'preds_3'
            if output.exists() and list(output.glob(name+'.*')):raise ValueError('Partial native outputs preserved')
            wait_memory();prepare()
            command=['./target/release/fit',kind,name,'-p','pipeline-3.toml','--final','--preds','preds_3']
            started=stamp();logpath=REPORT/(name+'.log')
            with logpath.open('x') as log:
                child=subprocess.Popen(command,cwd=TREE,env=latent.model_environment(params),stdout=log,stderr=subprocess.STDOUT)
                u.write(REPORT/(name+'-launch.json'),dict(pid=child.pid,started=started,command=command,model_environment=params))
                u.write(REPORT/'status.json',dict(state='native_running',job=name,pid=child.pid,started=started))
                code=child.wait()
            if code:raise RuntimeError('Native failure preserved: '+str(code))
            shutil.copyfile(logpath,output/(name+'.out'))
            epochs=int(params['EPOCHS']);trace=[int(v) for v in re.findall(r'^Epoch\s+(\d+)\s*\|',logpath.read_text(),re.M)]
            if trace!=list(range(1,epochs+1))*2:raise ValueError('Both full epoch traces required')
            outputs={str(p.relative_to(TREE)):u.sha(p) for p in output.glob(name+'*') if p.is_file()}
            if not set(plan['original_registry_jobs'][name]['outputs']).issubset(outputs):raise ValueError('Missing declared output')
            for split,count in [('probe_3',1760494),('qual',2817131)]:
                for p in output.glob(name+'*.'+split+'.npy'):
                    a=np.load(p,mmap_mode='r')
                    if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Malformed query vector')
            pairs=[]
            for column in [n for n in plan['fixed_published_outputs'] if n==name or n.startswith(name+'_ep')]:
                path=RUN/(column+'.probex.npy');a=np.load(output/(column+'.probe_3.npy'),mmap_mode='r');np.save(path,a[mapping])
                qual=output/(column+'.qual.npy')
                pairs.append(dict(name=column,probex=str(path.relative_to(u.ROOT)),qual=str(qual.relative_to(u.ROOT)),
                    probex_sha256=u.sha(path),qual_sha256=u.sha(qual)))
            u.write(receipt,dict(state='complete_pending_independent_verification',job=name,source_pin=u.PIN,
                plan_sha256=u.sha(REPORT/'plan.json'),started=started,finished=stamp(),command=command,
                model_environment=params,epochs_per_phase=epochs,both_full_epoch_traces=True,outputs=outputs,
                projected_pairs=pairs,qualifying_labels_read=False,public_prediction_values_read=False))
        u.write(REPORT/'status.json',dict(state='complete_pending_independent_verification',observed=stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    else:run()
