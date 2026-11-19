#!/usr/bin/env python3
"""Original split3 attention pilot in an isolated, own-data source tree."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np
import upstream_repro as u
import upstream_cf96_memory_resume as memory

RUN=u.ROOT/'artifacts/upstream-split3-pilot'
TREE=RUN/'source'
REPORT=u.ROOT/'experiments/upstream-split3-pilot'
JOB='s3-attn16'
REGISTRY=u.ROOT/'experiments/upstream-all-pipeline-coverage/registry.json'
COMMAND=['./target/release/fit','attn',JOB,'-p','pipeline-3.toml','--final','--preds','preds_3']
MODEL_ENV=dict(RAYON_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',D='16',EPSAVE='1',SEED='11')


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def keys(users,items,dates):
    if (users.ndim!=1 or items.shape!=users.shape or dates.shape!=users.shape or
            np.any(users<0) or np.any(users>=480189) or np.any(items<0) or np.any(items>=17770) or
            np.any(dates<-32768) or np.any(dates>32767)):
        raise ValueError('Invalid row identities')
    return ((users.astype(np.int64)*17770+items.astype(np.int64))<<16)+(dates.astype(np.int64)+32768)


def projection(source,known):
    """Unique, exact row map; both datasets must have the same known labels."""
    a=keys(*source[:3]);b=keys(*known[:3]);order=np.argsort(a);ordered=a[order]
    if np.any(ordered[1:]==ordered[:-1]) or len(np.unique(b))!=len(b):
        raise ValueError('Duplicate row identity')
    indices=np.searchsorted(ordered,b)
    if np.any(indices==len(ordered)) or not np.array_equal(ordered[indices],b):
        raise ValueError('Known evaluation row missing from source holdout')
    result=order[indices]
    if not np.array_equal(source[3][result],known[3]):raise ValueError('Known labels disagree')
    return result


def assert_disjoint(sorted_holdout,training):
    for lo in range(0,len(training[0]),262144):
        hi=min(lo+262144,len(training[0]));k=keys(*[a[lo:hi] for a in training])
        ix=np.searchsorted(sorted_holdout,k);inside=ix<len(sorted_holdout)
        if np.any(sorted_holdout[ix[inside]]==k[inside]):
            raise ValueError('Evaluation identity appears in training data')


def arrays(directory,labels=False):
    names=['user_idxs','item_idxs','dates']+(['ratings'] if labels else [])
    return [np.load(directory/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in names]


def implementations():
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(u.__file__),Path(memory.__file__)]}


def prepare():
    if (REPORT/'plan.json').exists():
        plan=u.read(REPORT/'plan.json')
        if plan['implementation_sha256']!=implementations():raise ValueError('Frozen pilot implementation changed')
        for name,digest in plan['source_files_sha256'].items():
            if u.sha(TREE/name)!=digest:raise ValueError('Pilot source changed: '+name)
        for name,digest in plan['input_sha256'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Pilot input changed: '+name)
        for name,digest in plan['binary_sha256'].items():
            if u.sha(TREE/'target/release'/name)!=digest:raise ValueError('Pilot binary changed')
        return plan
    if RUN.exists() or REPORT.exists():raise ValueError('Partial pilot setup preserved')
    u.validate_fresh();coverage=u.read(REGISTRY.parent/'coverage.json')
    if coverage['registry_sha256']!=u.sha(REGISTRY):raise ValueError('Source registry changed')
    native=u.read(REGISTRY)['3']['jobs'][JOB]
    expected='cargo build --release --bin fit && RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 D=16 EPSAVE=1 SEED=11 ./target/release/fit attn s3-attn16 -p pipeline-3.toml --final --preds preds_3 '
    if native['command']!=expected:raise ValueError('Original model recipe differs')
    if os.statvfs(u.ROOT).f_bavail*os.statvfs(u.ROOT).f_frsize<14*2**30:raise ValueError('Pilot needs 14 GiB disk headroom')
    RUN.mkdir();REPORT.mkdir();TREE.mkdir()
    code={}
    for name,digest in coverage['source_tracked_files_sha256'].items():
        # Copy code/configuration only: no public prediction files or label payload.
        if name.startswith(('data/','preds_')):continue
        original=u.SOURCE/name
        if u.sha(original)!=digest:raise ValueError('Pinned source file changed: '+name)
        target=TREE/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
        if u.sha(target)!=digest:raise ValueError('Source snapshot copy differs')
        code[name]=digest
    env=u.environment();cargo=shutil.which('cargo',path=env['PATH'])
    with (REPORT/'build.log').open('w') as log:
        subprocess.run([cargo,'build','--release','--locked','--features','blas','--bin','split_3'],
            cwd=u.SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    build=u.read(u.BASE/'build-receipt.json')
    if build['source_commit']!=u.PIN or build['binaries']['fit']!=u.sha(u.BASE/'target/release/fit'):
        raise ValueError('Original fit binary lacks matching build receipt')
    binaries={};(TREE/'target/release').mkdir(parents=True)
    for name in ['fit','split_3']:
        shutil.copy2(u.BASE/'target/release'/name,TREE/'target/release'/name)
        binaries[name]=u.sha(TREE/'target/release'/name)
    inputs={str(p.relative_to(u.ROOT)):u.sha(p) for p in [REGISTRY,REGISTRY.parent/'coverage.json',u.BASE/'build-receipt.json']}
    (TREE/'data').mkdir()
    for split in ['train','probe','fulltrain','qual']:
        original=u.FRESH/'data'/split
        (TREE/'data'/split).symlink_to(original,target_is_directory=True)
        for path in original.glob('*.npy'):inputs[str(path.relative_to(u.ROOT))]=u.sha(path)
    if np.any(np.load(TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets are not zero')
    plan=dict(created=stamp(),source_pin=u.PIN,job=JOB,original_registry_job=native,
        actual_model_command=COMMAND,model_environment=MODEL_ENV,implementation_sha256=implementations(),
        source_files_sha256=code,input_sha256=inputs,binary_sha256=binaries,
        original_capacity=dict(attention_dimensions=16,mf_dimensions=192,pool=32,epochs=12,seed=11,model_threads=1),
        compilation='Original fit executable reused with matching pinned build receipt; split_3 compiled unchanged from pinned source.',
        isolation='Private source/output tree; only verified own ingested data symlinked. No ingest or downloaded prediction values. Qualifying targets zero.',
        admission='Two stable memory samples; primary scheduler absent during coordinated CF96 slot; owned native footprint <12 GiB; compressor <6 GiB; swap growth <32 MiB/30 seconds.',
        qualifying_labels_read=False,public_prediction_values_read=False)
    u.write(REPORT/'plan.json',plan);return plan


def primary_processes():
    names={'upstream_primary_guarded.py','upstream_primary_guarded_aex.py','upstream_full_reproduce.py'}
    result={}
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        pid,command=row.strip().split(None,1)
        if any(Path(arg).name in names for arg in command.split()[1:]):result[pid]=command
    return result


def stable_temporal_identity():
    row=subprocess.run(['ps','-p','16305','-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
    return row if row.endswith('./target/release/tsvdx5-new tsvdx5-60') else None


def memory_admitted(previous,current,owned_bytes,primary_present,stable_temporal):
    # The small attention/data-prep pilot reserves four GiB, whereas the CF96
    # two-phase admission reserves eight. Permit the already allocated temporal
    # job alongside this pilot; arbitrary newly loading primary jobs still wait.
    limit=20*2**30 if primary_present and stable_temporal else 12*2**30
    return ((not primary_present or stable_temporal) and owned_bytes<limit and
        current['compressor_bytes']<6*2**30 and
        current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20)


def wait_memory(stage):
    previous=memory.memory();good=0;expected=u.read(REPORT/'plan.json').get('admission_baseline_native')
    while True:
        current=memory.memory();primary=primary_processes()
        actual=stable_temporal_identity();stable=bool(expected and actual==expected)
        owned=memory.footprints()
        admitted=memory_admitted(previous,current,sum(r['bytes'] for r in owned.values()),bool(primary),stable)
        good=good+1 if admitted else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_memory_and_primary_idle',stage=stage,observed=stamp(),
            primary_processes=primary,memory=current,owned_workers=owned,good_samples=good,
            original_temporal_identity_unchanged=stable))
        if good>=2:return
        previous=current;time.sleep(30)


def native(command,env,log):
    u.write(REPORT/'status.json',dict(state='native_running',command=command,observed=stamp()))
    with log.open('w') as out:
        child=subprocess.Popen(command,cwd=TREE,env=env,stdout=out,stderr=subprocess.STDOUT)
        u.write(REPORT/(log.stem+'-launch.json'),dict(pid=child.pid,command=command,observed=stamp()))
        # Observation timeouts never terminate or restart this uncheckpointed job.
        code=child.wait()
    if code:raise RuntimeError('Native job failed; outputs preserved: '+str(code))


def run():
    plan=prepare()
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'paired-receipt.json').exists():raise ValueError('Completed pilot preserved')
        wait_memory('split3_data_preparation')
        if (TREE/'data/train_3').exists() or (TREE/'data/probe_3').exists():
            raise ValueError('Existing split3 preparation preserved; verify before resuming')
        native(['./target/release/split_3'],u.environment(),REPORT/'split3-data.log')
        source=arrays(TREE/'data/probe_3',labels=True);known=arrays(u.FRESH/'data/probex',labels=True)
        mapping=projection(source,known)
        heldout=np.sort(keys(*known[:3]));assert_disjoint(heldout,arrays(TREE/'data/train_3'))
        np.save(RUN/'probex-row-map.npy',mapping)
        data_files={str(p.relative_to(RUN)):u.sha(p) for split in ['train_3','probe_3'] for p in (TREE/'data'/split).glob('*.npy')}
        u.write(REPORT/'data-verification.json',dict(state='complete',rows=len(mapping),
            source_rows=len(source[0]),source_pin=u.PIN,plan_sha256=u.sha(REPORT/'plan.json'),
            row_map_sha256=u.sha(RUN/'probex-row-map.npy'),data_sha256=data_files,
            known_rows_absent_from_training=True,known_labels_match=True,qualifying_labels_read=False))
        wait_memory('original_attention_two_phase_fit')
        # Recheck all actual immutable inputs after the potentially long wait.
        prepare()
        for name,digest in data_files.items():
            if u.sha(RUN/name)!=digest:raise ValueError('Prepared split3 data changed')
        env=u.environment()
        for key in list(env):
            if key.startswith(('CF_','LAB_')) or key in {'QUAL_ONLY','EPOCHS','THREADS','NMF','POOL','TRANSPOSE','D','SEED','EPSAVE'}:
                env.pop(key)
        env.update(MODEL_ENV)
        started=stamp();native(COMMAND,env,REPORT/'model.log')
        shutil.copyfile(REPORT/'model.log',TREE/f'preds_3/{JOB}.out')
        outputs={str(p.relative_to(TREE)):u.sha(p) for p in (TREE/'preds_3').iterdir() if p.is_file()}
        if not set(plan['original_registry_job']['outputs']).issubset(outputs):raise ValueError('Original declared outputs incomplete')
        for split,count in [('probe_3',len(source[0])),('qual',2817131)]:
            for p in (TREE/'preds_3').glob('*.'+split+'.npy'):
                a=np.load(p,mmap_mode='r',allow_pickle=False)
                if a.shape!=(count,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Malformed native prediction')
            if not np.array_equal(np.load(TREE/f'preds_3/{JOB}.{split}.npy'),np.load(TREE/f'preds_3/{JOB}_ep12.{split}.npy')):
                raise ValueError('Terminal and final epoch disagree')
        pairs=[]
        for name in [JOB,JOB+'_ep08']:
            path=RUN/(name+'.probex.npy');np.save(path,np.load(TREE/f'preds_3/{name}.probe_3.npy')[mapping])
            pairs.append(dict(name=name,probex=str(path.relative_to(u.ROOT)),probex_sha256=u.sha(path),
                qual=str((TREE/f'preds_3/{name}.qual.npy').relative_to(u.ROOT)),qual_sha256=outputs[f'preds_3/{name}.qual.npy']))
        u.write(REPORT/'paired-receipt.json',dict(state='complete',job=JOB,source_pin=u.PIN,
            plan_sha256=u.sha(REPORT/'plan.json'),data_verification_sha256=u.sha(REPORT/'data-verification.json'),
            started=started,finished=stamp(),command=COMMAND,model_environment=MODEL_ENV,
            binary_sha256=plan['binary_sha256']['fit'],outputs=outputs,projected_pairs=pairs,
            qualifying_labels_read=False,public_prediction_values_read=False,
            scope='Original complete two-phase native split3 pilot with known-ProbeX projection. No score or target-success claim.'))
        u.write(REPORT/'status.json',dict(state='complete_pending_independent_verification',observed=stamp()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    else:run()
