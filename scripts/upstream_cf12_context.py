#!/usr/bin/env python3
"""Known-ProbeX context NN with the completed fresh CFNADE12 predictor."""
import argparse
import copy
import datetime as dt
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import upstream_context_blend as context
import upstream_remote_batch as algebra
import upstream_cfnade48 as resources
from upstream_full_reproduce import receipt_valid,group_rss,stop

n,u,x=context.n,context.u,context.x
parent=context.parent
verifier=context.verifier
RUN=u.ROOT/'artifacts/upstream-cf12-context'
REPORT=u.ROOT/'experiments/upstream-cf12-context'
CONTEXT_REPORT=context.REPORT
CONFIG=dict(context.CONFIG,architecture='context-weighted-residual-cf12-v1')
SIGNALS=list(range(63))+list(range(68,91))+[221,222]
CONTEXT=list(context.CONTEXT)


def model_factory(width,cfg):
    import torch
    if width!=223 or cfg['architecture']!=CONFIG['architecture']:
        raise ValueError('CF12 context model requires the frozen 86-predictor layout')
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(SIGNALS))
            self.register_buffer('context',torch.tensor(CONTEXT))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),
                torch.nn.Dropout(d),torch.nn.Linear(h,h),torch.nn.GELU(),
                torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(SIGNALS),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),
                torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:
                torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,values):
            factors=self.project(values.index_select(1,self.signals))
            weights=self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()


def neural_model(checkpoint):
    import torch
    torch.set_num_threads(CONFIG['threads'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['config']!=CONFIG:raise ValueError('Saved CF12 context recipe changed')
    model=model_factory(saved['feature_width'],CONFIG)
    model.load_state_dict(saved['state']);return model


def code_hashes():
    return {**context.code_hashes(),
        str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),
        'scripts/upstream_cfnade48.py':u.sha(Path(resources.__file__))}


def configure():
    # Only this interpreter's factory is rebound. The unchanged context training
    # loop still implements the original optimizer, seeding and stopping rules.
    context.model_factory=model_factory
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,CONFIG,code_hashes
    n.train_nn=context.train_nn
    verifier.neural_model=neural_model


def prepare():
    configure()
    if (REPORT/'manifest.json').exists():return n.verify_prepared()
    if RUN.exists():raise ValueError('Partial CF12 preparation preserved')
    n.CONFIG=copy.deepcopy(context.BASE_CONFIG);parent.configure();pm=n.verify_prepared()
    sources={}
    def bind(path):sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for root in [parent.REPORT,CONTEXT_REPORT]:
        evaluation=root/'evaluation.json';proof=root/'independent-verification.json'
        if u.read(proof)['evaluation_sha256']!=u.sha(evaluation):
            raise ValueError('Preceding development result is not independently verified')
        for p in [root/'manifest.json',evaluation,proof]:bind(p)
    previous_row=next(r for r in u.read(CONTEXT_REPORT/'evaluation.json')['records'] if r['arm']=='fixed-half-tree-nn')
    previous_vector=context.RUN/'fixed-half-tree-nn.f32'
    if u.sha(previous_vector)!=previous_row['prediction_sha256']:raise ValueError('Previous context prediction changed')
    bind(previous_vector)
    replay=parent.REPORT/'tree-replay.json'
    if len(u.read(replay)['records'])!=5 or not all(r['bitwise_equal'] for r in u.read(replay)['records']):
        raise ValueError('Reused tree replay incomplete')
    bind(replay)
    tree=next(r for r in u.read(parent.REPORT/'evaluation.json')['records'] if r['arm']=='tree')
    if u.sha(parent.RUN/'tree.f32')!=tree['prediction_sha256']:raise ValueError('Reused tree changed')
    for fold in range(5):
        path=parent.REPORT/'tree'/(str(fold)+'.json');r=u.read(path)
        if r['manifest_sha256']!=u.sha(parent.REPORT/'manifest.json'):raise ValueError('Reused tree receipt changed')
        for name,digest in r['outputs'].items():
            if u.sha(parent.RUN/'tree'/str(fold)/name)!=digest:raise ValueError('Reused tree model changed')
        bind(path)
    u.validate_fresh();jobs=u.read(u.BASE/'registry.json')['jobs']
    if not receipt_valid('cfnade-12',jobs):raise ValueError('Complete paired CF12 native receipt required')
    bind(u.BASE/'fresh-receipts/cfnade-12.json')
    added=None
    for split,count in [('probex',pm['rows']),('qual',2817131)]:
        p=u.FRESH/f'preds_new/cfnade-12.{split}.npy';a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.shape!=(count,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Malformed paired CF12 predictor')
        if not np.array_equal(a,np.load(u.FRESH/f'preds_new/cfnade-12_ep40.{split}.npy',mmap_mode='r')):
            raise ValueError('CF12 terminal and original40-epoch snapshot differ')
        bind(p)
        if split=='probex':added=np.clip(a[:,None],0,6)
    old=np.load(parent.RUN/'design.npy',mmap_mode='r');features=np.load(parent.RUN/'features.npy',mmap_mode='r')
    y,users,groups=[np.load(parent.RUN/(key+'.npy')) for key in ['labels','users','groups']]
    if not np.array_equal(y,np.load(u.FRESH/'data/probex/ratings.npy')) or not np.array_equal(users,np.load(u.FRESH/'data/probex/user_idxs.npy')):
        raise ValueError('Development row identities changed')
    if old.shape!=(len(y),161) or features.shape!=(len(y),221):raise ValueError('Parent feature layout differs')
    scored=algebra.screen(old,added,y,groups,dict(control=[],main=[0]))
    if abs(x.metric(scored['control']['prediction'],y)-pm['baseline_rmse'])>1e-9:raise ValueError('Parent ridge control differs')
    configure();REPORT.mkdir(parents=True,exist_ok=True);RUN.mkdir()
    z=np.lib.format.open_memmap(RUN/'design.npy',mode='w+',dtype=np.float64,shape=(len(y),162))
    f=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(len(y),222))
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));z[lo:hi]=algebra.augmented_block(old,added,lo,hi)
        f[lo:hi]=np.column_stack([features[lo:hi],added[lo:hi]-3.6])
    z.flush();f.flush()
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    for name,value in [('labels',y),('users',users),('groups',groups),('baseline',scored['main']['prediction'])]:np.save(RUN/(name+'.npy'),value)
    shutil.copyfile(parent.RUN/'tree.f32',RUN/'tree.f32')
    previous=previous_row['rmse']
    manifest=dict(config=CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        rows=len(y),baseline_rmse=x.metric(scored['main']['prediction'],y),prediction_columns=86,
        design_columns=162,nonlinear_features=222,signal_indices=SIGNALS,context_indices=CONTEXT,
        previous_best_rmse=previous,reused_tree_prediction_sha256=tree['prediction_sha256'],
        arms=['control','tree','nn','fixed-half-tree-nn'],qualifying_labels_read=False,
        isolation=pm['isolation'],scope='Adaptive known-ProbeX development. Complete own CFNADE12 added to context NN; original85-predictor tree reused unchanged. New NN ridge/scalers/budgets exclude scoring users. No production fitter or qualifying audit launched.')
    u.write(REPORT/'manifest.json',manifest);return manifest


def summarize():
    manifest=n.verify_prepared();y,groups,baseline=[np.load(RUN/(k+'.npy')) for k in ['labels','groups','baseline']]
    folds=groups//2;nn=np.empty(len(y),dtype=np.float64)
    for fold in range(5):
        root=RUN/'nn'/str(fold);receipt=u.read(REPORT/'nn'/(str(fold)+'.json'))
        if receipt['manifest_sha256']!=u.sha(REPORT/'manifest.json'):raise ValueError('NN fold recipe changed')
        for name,digest in receipt['outputs'].items():
            if u.sha(root/name)!=digest:raise ValueError('NN fold output changed')
        part=np.fromfile(root/'prediction.f32',dtype='<f4');held=folds==fold
        if part.shape!=(int(held.sum()),) or not np.isfinite(part).all():raise ValueError('Malformed NN fold')
        nn[held]=part
    tree=np.fromfile(RUN/'tree.f32',dtype='<f4').astype(np.float64)
    predictions=dict(control=baseline,tree=tree,nn=nn)
    predictions['fixed-half-tree-nn']=(tree+nn)/2;records=[]
    for arm,value in predictions.items():
        path=RUN/(arm+'.f32')
        if arm!='tree':value.astype('<f4').tofile(path)
        stored=np.fromfile(path,dtype='<f4');score=x.metric(stored,y)
        records.append(dict(arm=arm,rmse=score,delta_vs_previous_best=score-manifest['previous_best_rmse'],
            fold_rmse=[x.metric(stored[folds==fold],y[folds==fold]) for fold in range(5)],prediction_sha256=u.sha(path)))
    result=dict(state='complete',manifest_sha256=u.sha(REPORT/'manifest.json'),records=records,
        previous_best_rmse=manifest['previous_best_rmse'],scope=manifest['scope'],isolation=manifest['isolation'])
    u.write(REPORT/'evaluation.json',result);verifier.score();return result


def guarded(command,log_path):
    with log_path.open('a') as log:
        child=subprocess.Popen(['nice','-n','10']+command,cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own=group_rss(child.pid);other=resources.other_rss({child.pid});fs=os.statvfs(u.ROOT);free=fs.f_bavail*fs.f_frsize
                high=high+1 if own+other>24*2**30 else 0
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or free<22*2**30:raise RuntimeError('CF12 context resource guard; partial outputs preserved')
                time.sleep(5)
        except BaseException:stop(child);raise
        if child.returncode:raise RuntimeError('CF12 context child failed; partial outputs preserved')


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while resources.other_rss(set())>17*2**30:
            u.write(REPORT/'status.json',dict(state='waiting_for_memory'));time.sleep(30)
        prepare();u.write(REPORT/'status.json',dict(state='nested_nn_fitting'))
        guarded([sys.executable,str(Path(__file__)),'--worker'],REPORT/'nn.log')
        u.write(REPORT/'status.json',dict(state='saved_nn_replay'))
        guarded([sys.executable,str(Path(__file__)),'--replay'],REPORT/'nn-replay.log')
        result=summarize();u.write(REPORT/'status.json',dict(state='development_complete',records=result['records'],goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();configure()
    if args.worker:n.worker('nn')
    elif args.replay:verifier.replay('nn')
    else:
        try:run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise
