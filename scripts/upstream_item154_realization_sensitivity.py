#!/usr/bin/env python3
"""Frozen-model known-row sensitivity; original realization must replay exactly."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='2'
import datetime as dt
import fcntl
import gc
import math
from pathlib import Path
import numpy as np
import upstream_item154_mps_context as recipe
import upstream_exact_neural_scaling as exact
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-item154-realization-sensitivity'
RUN=u.ROOT/'artifacts/upstream-item154-realization-sensitivity'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def score(p,y):
    a=np.clip(p.astype(np.float64),1.05,4.95)
    return math.sqrt(math.fsum(float(v-t)**2 for v,t in zip(a,y))/len(y))

def main():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing diagnostic preserved')
    REPORT.mkdir();RUN.mkdir()
    lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    recipe.configure()
    bindings={str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__))}
    paths={'A':u.ROOT/'artifacts/upstream-item154-features',
           'B':u.ROOT/'artifacts/upstream-item154-stream-bank'}
    u.write(REPORT/'declaration.json',dict(created=stamp(),
        rule='Both complete known-row input realizations through unchanged actual saved outer-fold models, ridge coefficients and training-only scalers. A must replay each original prediction bitwise before B is scored. No new fit or qualifying outcomes. B is sensitivity, not a separately trained CV result.',
        families=['upstream-item154-mps-context','upstream-item154-tree'],
        qualifying_labels_read=False,models_fitted=0))
    records=[]
    for family,name in [('nn','upstream-item154-mps-context'),('tree','upstream-item154-tree')]:
        cache=u.ROOT/'artifacts'/name;report=u.ROOT/'experiments'/name
        manifest=u.read(report/'manifest.json');cfg=manifest['config']
        groups=np.load(cache/'groups.npy');y=np.load(cache/'labels.npy')
        for p in [report/'manifest.json',cache/'groups.npy',cache/'labels.npy']:
            bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
        predictions={k:np.empty(len(y),dtype='<f4') for k in paths}
        for fold in range(5):
            held=groups//2==fold;root=cache/family/str(fold)
            receipt_path=report/family/(str(fold)+'.json');receipt=u.read(receipt_path)
            if receipt['manifest_sha256']!=u.sha(report/'manifest.json'):raise ValueError('Receipt differs')
            bindings[str(receipt_path.relative_to(u.ROOT))]=u.sha(receipt_path)
            for filename,digest in receipt['outputs'].items():
                p=root/filename
                if u.sha(p)!=digest:raise ValueError('Frozen model changed')
                bindings[str(p.relative_to(u.ROOT))]=digest
            expected=np.fromfile(root/'prediction.f32',dtype='<f4')
            for realization,directory in paths.items():
                prefix='' if realization=='A' else 'probex-'
                zp=directory/(prefix+'design.npy');fp=directory/(prefix+'features.npy')
                for p in [zp,fp]:
                    key=str(p.relative_to(u.ROOT))
                    if key not in bindings:bindings[key]=u.sha(p)
                z=np.load(zp,mmap_mode='r');features=np.load(fp,mmap_mode='r')
                with np.load(root/'ridge.npz') as saved:
                    baseline=np.clip(z@saved['full'],1.05,4.95)
                inputs=exact.append_baseline(features[held],baseline[held])
                if family=='nn':
                    with np.load(root/'scalers.npz') as saved:
                        scaler={k:saved['full_'+k] for k in ['lo','hi','mean','std']}
                    inputs=exact.apply_scaler(inputs,scaler)
                correction=np.zeros(int(held.sum()),dtype=np.float64)
                for seed,iterations in zip(cfg['seeds'],receipt['chosen_iterations']):
                    if not iterations:continue
                    if family=='nn':
                        model=recipe.neural_model(root/f'nn-{seed}.pt')
                        correction+=recipe.mps_backend.predict(model,inputs)/len(cfg['seeds'])
                    else:
                        import lightgbm as lgb
                        model=lgb.Booster(model_file=str(root/f'tree-{seed}.txt'))
                        correction+=model.predict(inputs,num_threads=cfg['threads'])/len(cfg['seeds'])
                    del model
                actual=np.clip(baseline[held]+correction,1.05,4.95).astype('<f4')
                if realization=='A' and not np.array_equal(actual.view('u4'),expected.view('u4')):
                    raise ValueError(f'Original replay differs: {family}/{fold}, max={np.max(abs(actual-expected))}')
                predictions[realization][held]=actual
                del inputs,features,z,baseline;gc.collect()
                u.write(REPORT/'status.json',dict(state='running',family=family,fold=fold,
                    realization=realization,original_bitwise_replay=realization=='A',observed=stamp()))
                print(family,fold,realization,'complete',flush=True)
        for realization,p in predictions.items():
            path=RUN/(family+'-'+realization+'.f32');p.tofile(path)
            records.append(dict(family=family,realization=realization,rmse=score(p,y),
                prediction_sha256=u.sha(path),rows=len(p)))
        delta=predictions['B'].astype(np.float64)-predictions['A']
        u.write(REPORT/(family+'.json'),dict(records=records[-2:],
            prediction_delta_rms=float(np.sqrt(np.mean(delta**2))),
            maximum_prediction_delta=float(np.max(abs(delta))),all_original_folds_bitwise=True))
    for key,digest in bindings.items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Bound input changed during diagnostic')
    u.write(REPORT/'verification.json',dict(state='complete_frozen_item154_input_realization_sensitivity',
        observed=stamp(),records=records,source_sha256=bindings,
        all_ten_original_fold_predictions_bitwise=True,models_fitted=0,
        qualifying_labels_read=False,scope='Frozen-model sensitivity only; no new independently trained production-realization CV claim.'))

if __name__=='__main__':main()
