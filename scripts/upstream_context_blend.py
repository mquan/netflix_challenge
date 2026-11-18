#!/usr/bin/env python3
"""Fixed context-weighted residual NN on the verified fresh 85-predictor bank."""
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
import upstream_voting_blend as parent
import upstream_nonlinear as n
import verify_upstream_nonlinear as verifier
from upstream_full_reproduce import group_rss, stop
from upstream_extension_run import native_rss

u, x = n.u, n.x
RUN = u.ROOT/'artifacts/upstream-context-blend'
REPORT = u.ROOT/'experiments/upstream-context-blend'
BASE_CONFIG = copy.deepcopy(n.CONFIG)
CONFIG = dict(BASE_CONFIG, architecture='context-weighted-residual-v1', context_rank=16)
SIGNALS = list(range(63))+list(range(68,91))+[221]
CONTEXT = list(range(63,68))+list(range(91,221))


def model_factory(width, cfg):
    import torch
    if width != 222 or cfg['architecture'] != CONFIG['architecture']:
        raise ValueError('Context model requires the frozen 85-predictor/130-voting layout')
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            h, r, d = cfg['nn_hidden'], cfg['context_rank'], cfg['nn_dropout']
            self.register_buffer('signals', torch.tensor(SIGNALS))
            self.register_buffer('context', torch.tensor(CONTEXT))
            self.base = torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),
                torch.nn.Dropout(d),torch.nn.Linear(h,h),torch.nn.GELU(),
                torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project = torch.nn.Linear(len(SIGNALS),r,bias=False)
            self.gate = torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),
                torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:
                torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self, values):
            factors = self.project(values.index_select(1,self.signals))
            weights = self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()


def train_nn(features, residual, cfg, seed, epochs, valid=None):
    import torch
    torch.set_num_threads(cfg['threads']);torch.manual_seed(seed)
    model = model_factory(features.shape[1],cfg)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['nn_lr'],weight_decay=cfg['nn_weight_decay'])
    tx = torch.from_numpy(np.ascontiguousarray(features))
    ty = torch.from_numpy(np.asarray(residual,dtype=np.float32))
    best, chosen = (float(np.mean(valid[1]**2)),0) if valid is not None else (None,epochs)
    for epoch in range(1,epochs+1):
        model.train()
        for ix in torch.randperm(len(tx)).split(cfg['nn_batch']):
            optimizer.zero_grad(set_to_none=True)
            (model(tx[ix]).squeeze(1)-ty[ix]).square().mean().backward()
            optimizer.step()
        if valid is not None:
            loss = float(np.mean((n.nn_predict(model,valid[0])-valid[1])**2))
            if loss < best:best,chosen = loss,epoch
            n.event(stage='nn_epoch',seed=seed,epoch=epoch,inner_mse=loss,chosen=chosen)
            if epoch-chosen >= cfg['nn_patience']:break
    return model,chosen


def neural_model(checkpoint):
    import torch
    torch.set_num_threads(CONFIG['threads'])
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['config'] != CONFIG:raise ValueError('Saved context recipe changed')
    model = model_factory(saved['feature_width'],CONFIG)
    model.load_state_dict(saved['state']);return model


def code_hashes():
    return {**parent.code_hashes(),str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),
        'scripts/verify_upstream_nonlinear.py':u.sha(u.ROOT/'scripts/verify_upstream_nonlinear.py'),
        'scripts/upstream_extension_run.py':u.sha(u.ROOT/'scripts/upstream_extension_run.py')}


def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes,n.train_nn = RUN,REPORT,CONFIG,code_hashes,train_nn
    verifier.neural_model = neural_model


def prepare():
    configure()
    if (REPORT/'manifest.json').exists():return n.verify_prepared()
    if RUN.exists():raise ValueError('Partial preparation preserved; inspect before resuming')
    n.CONFIG=BASE_CONFIG;parent.configure();pm=n.verify_prepared()
    evaluation=parent.REPORT/'evaluation.json'
    proof=u.read(parent.REPORT/'independent-verification.json')
    if proof['evaluation_sha256']!=u.sha(evaluation):raise ValueError('Parent score binding changed')
    tree_row=next(r for r in u.read(evaluation)['records'] if r['arm']=='tree')
    if u.sha(parent.RUN/'tree.f32')!=tree_row['prediction_sha256']:raise ValueError('Parent tree changed')
    replay=u.read(parent.REPORT/'tree-replay.json')
    if len(replay['records'])!=5 or not all(r['bitwise_equal'] for r in replay['records']):
        raise ValueError('Parent tree replay incomplete')
    configure();REPORT.mkdir(parents=True,exist_ok=True);RUN.mkdir()
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in [parent.REPORT/'manifest.json',
        evaluation,parent.REPORT/'independent-verification.json',parent.REPORT/'tree-replay.json']}
    for fold in range(5):
        p=parent.REPORT/'tree'/(str(fold)+'.json');receipt=u.read(p)
        if receipt['manifest_sha256']!=u.sha(parent.REPORT/'manifest.json'):
            raise ValueError('Parent tree receipt recipe changed')
        for filename,digest in receipt['outputs'].items():
            if u.sha(parent.RUN/'tree'/str(fold)/filename)!=digest:raise ValueError('Parent tree fold changed')
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name in pm['prepared_sha256']:
        shutil.copyfile(parent.RUN/name,RUN/name)
    shutil.copyfile(parent.RUN/'tree.f32',RUN/'tree.f32')
    features=np.load(RUN/'features.npy',mmap_mode='r')
    if features.shape!=(pm['rows'],221):raise ValueError('Parent feature layout changed')
    manifest=dict(config=CONFIG,versions=n.versions(),code_sha256=code_hashes(),
        source_commit=u.PIN,source_manifests_sha256=sources,
        prepared_sha256={name:u.sha(RUN/name) for name in list(pm['prepared_sha256'])+['tree.f32']},
        rows=pm['rows'],baseline_rmse=pm['baseline_rmse'],signal_indices=SIGNALS,context_indices=CONTEXT,
        reused_tree_prediction_sha256=tree_row['prediction_sha256'],
        isolation=pm['isolation'],qualifying_labels_read=False,
        scope='Adaptive known-ProbeX development only. Own fresh parent features, original user folds and ridge; one predeclared rank-16 context interaction architecture. Reuses the verified parent tree without retraining or new tree receipts.',
        arms=['control','tree','nn','fixed-half-tree-nn'])
    u.write(REPORT/'manifest.json',manifest);return manifest


def summarize():
    manifest=n.verify_prepared()
    y,groups,baseline=[np.load(RUN/(key+'.npy')) for key in ['labels','groups','baseline']]
    folds=groups//2;nn=np.empty(len(y),dtype=np.float64)
    for fold in range(5):
        root=RUN/'nn'/str(fold);receipt=u.read(REPORT/'nn'/(str(fold)+'.json'))
        if receipt['manifest_sha256']!=u.sha(REPORT/'manifest.json'):raise ValueError('NN recipe changed')
        for filename,digest in receipt['outputs'].items():
            if u.sha(root/filename)!=digest:raise ValueError('NN fold artifact changed')
        part=np.fromfile(root/'prediction.f32',dtype='<f4');held=folds==fold
        if part.shape!=(int(held.sum()),) or not np.isfinite(part).all():raise ValueError('Invalid NN fold')
        nn[held]=part
    tree=np.fromfile(RUN/'tree.f32',dtype='<f4').astype(np.float64)
    if tree.shape!=y.shape:raise ValueError('Invalid reused tree')
    predictions=dict(control=baseline,tree=tree,nn=nn)
    predictions['fixed-half-tree-nn']=(tree+nn)/2
    previous=next(r['rmse'] for r in u.read(parent.REPORT/'evaluation.json')['records'] if r['arm']=='fixed-half-tree-nn')
    records=[]
    for arm,value in predictions.items():
        path=RUN/(arm+'.f32')
        if arm!='tree':value.astype('<f4').tofile(path)
        stored=np.fromfile(path,dtype='<f4');score=x.metric(stored,y)
        records.append(dict(arm=arm,rmse=score,delta_vs_control=score-manifest['baseline_rmse'],
            delta_vs_previous_best=score-previous,fold_rmse=[x.metric(stored[folds==f],y[folds==f]) for f in range(5)],
            prediction_sha256=u.sha(path)))
    result=dict(state='complete',manifest_sha256=u.sha(REPORT/'manifest.json'),records=records,
        parent_best_rmse=previous,scope=manifest['scope'],isolation=manifest['isolation'])
    u.write(REPORT/'evaluation.json',result);verifier.score();return result


def guarded(command, log_path):
    with log_path.open('a') as log:
        child=subprocess.Popen(['nice','-n','10']+command,cwd=u.ROOT,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                own,other=group_rss(child.pid),native_rss()
                fs=os.statvfs(u.ROOT);free=fs.f_bavail*fs.f_frsize
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,native_rss_bytes=other,
                    disk_free_bytes=free,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat()))
                if own+other>25*2**30 or free<20*2**30:
                    raise RuntimeError('Context experiment resource guard; existing native fits retain priority')
                time.sleep(5)
        except BaseException:stop(child);raise
        if child.returncode:raise RuntimeError('Context child failed; partial outputs preserved')


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while native_rss()>18*2**30:
            u.write(REPORT/'status.json',dict(state='waiting_for_memory'));time.sleep(30)
        prepare()
        u.write(REPORT/'status.json',dict(state='nested_nn_training'))
        guarded([sys.executable,str(Path(__file__)),'--worker'],REPORT/'nn.log')
        u.write(REPORT/'status.json',dict(state='saved_nn_replay'))
        guarded([sys.executable,str(Path(__file__)),'--replay'],REPORT/'nn-replay.log')
        result=summarize()
        u.write(REPORT/'status.json',dict(state='development_complete',records=result['records'],goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();configure()
    if args.worker:n.worker('nn')
    elif args.replay:verifier.replay('nn')
    else:
        try:run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise
