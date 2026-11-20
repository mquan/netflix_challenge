#!/usr/bin/env python3
"""LightGBM-only full tree fitting from a pipe; never import Torch here."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import lightgbm as lgb
import upstream_repro as u
import upstream_nonlinear as recipe
import upstream_tree_pipe as wire

RUN=u.ROOT/'artifacts/upstream-own391-cpu-tree'
REPORT=u.ROOT/'experiments/upstream-own391-cpu-tree'
SOURCE=u.ROOT/'scripts/upstream_own391_tree_source.py'


def checked():
    if 'torch' in sys.modules:raise ValueError('Torch must remain isolated from LightGBM')
    m=u.read(REPORT/'manifest.json')
    if (m['state']!='declared_full_budget_matched391_cpu_tree' or m['config']!=recipe.CONFIG or
        m['versions']!=recipe.versions() or m['qualifying_labels_read'] is not False or
        m['nonlinear_OOF_predictions_used_as_features'] is not False or m['rows']!=1128164 or m['features']!=617):
        raise ValueError('Original complete matched391 tree declaration required')
    for rel,digest in m['source_sha256'].items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original tree source changed: '+rel)
    for name,digest in m['prepared_sha256'].items():
        if u.sha(RUN/name)!=digest:raise ValueError('Original labels/groups/statistics/NN partner changed')
    return m


def receive(independent=False):
    m=checked();action='stream-independent' if independent else 'stream'
    log=(REPORT/('replay-source.log' if independent else 'fit-source.log')).open('x')
    child=subprocess.Popen([sys.executable,'-u',str(SOURCE),action],cwd=u.ROOT,stdout=subprocess.PIPE,stderr=log)
    try:
        header=wire.read_json(child.stdout)
        if (header['state']!='matched391_tree_pipe_v1' or header['rows']!=1128164 or header['features']!=617 or
            header['baselines']!=10 or header['chunk']!=8192 or header['manifest_sha256']!=u.sha(REPORT/'manifest.json')):
            raise ValueError('Exact complete original feature pipe schema required')
        with np.load(RUN/'stats.npz') as saved:gram,rhs=saved['gram'],saved['rhs']
        weights=np.stack([recipe.ridge_weights(gram,rhs,(np.arange(10)//2!=fold)&selector,1000)
            for fold in range(5) for selector in [np.arange(10)%2==0,np.ones(10,dtype=bool)]])
        if hashlib.sha256(weights.tobytes()).hexdigest()!=header['weights_sha256']:
            raise ValueError('Every eligible original ridge coefficient must match the source')
        features=np.empty((1128164,618),dtype=np.float32);features[:,-1]=0
        bases=np.empty((1128164,10));hashes=[hashlib.sha256(),hashlib.sha256()]
        for lo in range(0,1128164,8192):
            hi=min(lo+8192,1128164);f=wire.read_array(child.stdout,(hi-lo,617),'<f4');b=wire.read_array(child.stdout,(hi-lo,10),'<f8')
            if not np.isfinite(f).all() or not np.isfinite(b).all():raise ValueError('Finite complete source arrays required')
            features[lo:hi,:617]=f;bases[lo:hi]=b;hashes[0].update(f.tobytes());hashes[1].update(b.tobytes())
        footer=wire.read_json(child.stdout)
        if child.stdout.read(1):raise ValueError('Unexpected bytes after complete feature stream')
        code=child.wait()
        if (code or hashes[0].hexdigest()!=footer['operators_sha256'][1] or
            hashes[1].hexdigest()!=footer['baseline_sha256']):raise ValueError('Complete original source process/stream verification required')
    except BaseException:
        if child.poll() is None:child.terminate()
        child.wait();raise
    finally:child.stdout.close();log.close()
    footer['weights_sha256']=header['weights_sha256']
    return m,features,bases,weights,footer


def parameters(cfg,seed):
    return dict(objective='regression',metric='l2',learning_rate=cfg['tree_lr'],num_leaves=cfg['tree_leaves'],
        min_data_in_leaf=cfg['tree_min_leaf'],feature_fraction=.9,bagging_fraction=.8,bagging_freq=5,
        seed=seed,num_threads=cfg['threads'],verbosity=-1,deterministic=True,force_col_wise=True)


def main():
    if (REPORT/'fit-completion.json').exists() or (RUN/'tree').exists():raise ValueError('Existing tree fits retained')
    m,x,bases,weights,streams=receive();cfg=m['config'];y=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy')
    if y.shape!=(1128164,) or set(np.unique(groups))!=set(range(10)):raise ValueError('All canonical known rows/groups required')
    tree=RUN/'tree';tree.mkdir();records=[]
    for fold in range(5):
        root=tree/str(fold);root.mkdir();training=groups//2!=fold;held=~training
        fit=training&(groups%2==0);valid=training&(groups%2==1)
        inner,full=bases[:,fold*2],bases[:,fold*2+1]
        np.savez(root/'ridge.npz',inner=weights[fold*2],full=weights[fold*2+1])
        correction=np.zeros(int(held.sum()));counts=[];selections=[]
        for seed in cfg['seeds']:
            params=parameters(cfg,seed+fold);x[:,-1]=inner
            baseline_mse=float(np.mean((y[valid]-inner[valid])**2))
            select=lgb.train(params,lgb.Dataset(x[fit],label=y[fit]-inner[fit]),num_boost_round=cfg['tree_rounds'],
                valid_sets=[lgb.Dataset(x[valid],label=y[valid]-inner[valid])],
                callbacks=[lgb.early_stopping(cfg['tree_patience'],verbose=False)])
            best=float(select.best_score['valid_0']['l2']);count=int(select.best_iteration) if best<baseline_mse else 0
            selections.append(dict(seed=seed,actual_seed=seed+fold,baseline_mse=baseline_mse,best_mse=best,best_iteration=int(select.best_iteration),chosen=count))
            select.free_dataset();del select;gc.collect();x[:,-1]=full
            if count:
                model=lgb.train(params,lgb.Dataset(x[training],label=y[training]-full[training]),num_boost_round=count)
                correction+=model.predict(x[held],num_threads=cfg['threads'])/3
                model.save_model(str(root/('tree-%s.txt'%seed)));model.free_dataset();del model;gc.collect()
            counts.append(count);recipe.event(stage='matched391_tree_seed_complete',fold=fold,seed=seed,iterations=count)
        prediction=np.clip(full[held]+correction,1.05,4.95).astype('<f4');prediction.tofile(root/'prediction.f32')
        receipt=dict(fold=fold,chosen_iterations=counts,selections=selections,rows=int(held.sum()),
            manifest_sha256=u.sha(REPORT/'manifest.json'),outputs={p.name:u.sha(p) for p in root.iterdir()})
        u.write(REPORT/('fold-%s.json'%fold),receipt);records.append(receipt)
    checked()
    u.write(REPORT/'fit-completion.json',dict(state='complete_original_budget_matched391_cpu_tree_fits_pending_replay',
        manifest_sha256=u.sha(REPORT/'manifest.json'),streams=streams,
        receipt_sha256={str(f):u.sha(REPORT/('fold-%s.json'%f)) for f in range(5)},
        qualifying_labels_read=False,actual_full_models=sum(c>0 for r in records for c in r['chosen_iterations'])))
    recipe.event(stage='all_matched391_tree_fits_complete')


if __name__=='__main__':main()
