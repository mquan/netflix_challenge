#!/usr/bin/env python3
"""Isolated LightGBM audit, avoiding the Torch/OpenMP runtime collision."""
import os
for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='2'
import datetime as dt
import fcntl
import gc
import math
from pathlib import Path
import numpy as np
import lightgbm as lgb
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-item154-tree-realization-sensitivity'
RUN=u.ROOT/'artifacts/upstream-item154-tree-realization-sensitivity'

def main():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing evidence preserved')
    REPORT.mkdir();RUN.mkdir()
    lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=u.ROOT/'artifacts/upstream-item154-tree';origin=u.ROOT/'experiments/upstream-item154-tree'
    groups=np.load(cache/'groups.npy');labels=np.load(cache/'labels.npy')
    mp=origin/'manifest.json';manifest=u.read(mp)
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),mp,cache/'groups.npy',cache/'labels.npy']}
    paths={'A':u.ROOT/'artifacts/upstream-item154-features','B':u.ROOT/'artifacts/upstream-item154-stream-bank'}
    vectors={k:np.empty(len(labels),dtype='<f4') for k in paths};records=[]
    for fold in range(5):
        held=groups//2==fold;root=cache/'tree'/str(fold)
        rp=origin/'tree'/f'{fold}.json';receipt=u.read(rp)
        if receipt['manifest_sha256']!=u.sha(mp):raise ValueError('Original model manifest differs')
        bindings[str(rp.relative_to(u.ROOT))]=u.sha(rp)
        for name,digest in receipt['outputs'].items():
            p=root/name
            if u.sha(p)!=digest:raise ValueError('Actual saved tree changed')
            bindings[str(p.relative_to(u.ROOT))]=digest
        for realization,directory in paths.items():
            prefix='' if realization=='A' else 'probex-'
            zp=directory/(prefix+'design.npy');fp=directory/(prefix+'features.npy')
            for p in [zp,fp]:
                key=str(p.relative_to(u.ROOT))
                if key not in bindings:bindings[key]=u.sha(p)
            z=np.load(zp,mmap_mode='r');features=np.load(fp,mmap_mode='r')
            with np.load(root/'ridge.npz') as saved:base=np.clip(z@saved['full'],1.05,4.95)
            x=np.empty((int(held.sum()),398),dtype=np.float32)
            x[:,:-1]=features[held];x[:,-1]=base[held]
            correction=np.zeros(len(x),dtype=np.float64)
            for seed,count in zip(manifest['config']['seeds'],receipt['chosen_iterations']):
                if count:
                    model=lgb.Booster(model_file=str(root/f'tree-{seed}.txt'))
                    correction+=model.predict(x,num_threads=manifest['config']['threads'])/3
                    del model
            actual=np.clip(base[held]+correction,1.05,4.95).astype('<f4')
            if realization=='A':
                expected=np.fromfile(root/'prediction.f32',dtype='<f4')
                if not np.array_equal(actual.view('u4'),expected.view('u4')):raise ValueError('Original tree replay differs')
            vectors[realization][held]=actual
            records.append(dict(fold=fold,realization=realization,rows=len(actual),original_bitwise_replay=realization=='A'))
            u.write(REPORT/'status.json',dict(state='running',records=records,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
            print(fold,realization,'complete',flush=True)
            del x,z,features,base;gc.collect()
    scores=[]
    for realization,p in vectors.items():
        path=RUN/(realization+'.f32');p.tofile(path)
        clipped=np.clip(p.astype(np.float64),1.05,4.95)
        rmse=math.sqrt(math.fsum(float(a-b)**2 for a,b in zip(clipped,labels))/len(labels))
        scores.append(dict(realization=realization,rmse=rmse,prediction_sha256=u.sha(path)))
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound source changed')
    delta=vectors['B'].astype(np.float64)-vectors['A']
    u.write(REPORT/'verification.json',dict(state='complete_isolated_tree_input_realization_sensitivity',records=records,scores=scores,
        source_sha256=bindings,prediction_delta_rms=float(np.sqrt(np.mean(delta**2))),all_five_original_folds_bitwise=True,
        models_fitted=0,qualifying_labels_read=False,scope='Frozen-model input sensitivity, not a newly fitted CV result. Original combined Torch/LightGBM attempt preserved with its OpenMP failure.'))

if __name__=='__main__':main()
