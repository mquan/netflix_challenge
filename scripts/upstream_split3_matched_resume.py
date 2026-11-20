#!/usr/bin/env python3
"""Preserve seven completed original folds and fit only three missing expanded folds."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import gc
from pathlib import Path
import numpy as np
import upstream_split3_matched_mps as study

u,n,data,backend,CONFIG=study.u,study.n,study.data,study.backend,study.CONFIG
RUN=u.ROOT/'artifacts/upstream-split3-matched-resume'
REPORT=u.ROOT/'experiments/upstream-split3-matched-resume'

def location(arm,fold):
    if arm not in study.ARMS or fold not in range(5):raise ValueError('Original arm/fold required')
    fresh=arm=='expanded' and fold>=2
    return (REPORT if fresh else study.REPORT)/arm/f'{fold}.json',(RUN if fresh else study.RUN)/arm/str(fold)

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing original-fold recovery preserved')
    old=u.read(study.REPORT/'manifest.json')
    if old['config']!=CONFIG or old['code_sha256']!=study.code_hashes():raise ValueError('Original complete numerical recipe must remain unchanged')
    failure=u.ROOT/'experiments/upstream-split3-matched-queue/failure.json'
    if 'Full matched resource guard exceeded' not in u.read(failure)['error']:raise ValueError('Actual terminal resource interruption required, not an observation timeout')
    sources=dict(old['source_sha256']);reused=[]
    for arm,folds in [('control',range(5)),('expanded',range(2))]:
        for fold in folds:
            receipt,root=location(arm,fold);r=u.read(receipt)
            if r['manifest_sha256']!=u.sha(study.REPORT/'manifest.json') or len(r['chosen_iterations'])!=3:raise ValueError('Complete original receipt required')
            if any(type(c)!=int or not 0<=c<=CONFIG['nn_epochs'] for c in r['chosen_iterations']):raise ValueError('Every original seed budget required')
            expected={'ridge.npz','scalers.npz','prediction.f32'}|{f'nn-{s}.pt' for s,c in zip(CONFIG['seeds'],r['chosen_iterations']) if c}
            if set(r['outputs'])!=expected or {p.name for p in root.iterdir()}!=expected:raise ValueError('Actual complete original saved pipeline required')
            for name,h in r['outputs'].items():
                p=root/name
                if u.sha(p)!=h:raise ValueError('Actual completed original model/prediction changed')
                sources[str(p.relative_to(u.ROOT))]=h
            sources[str(receipt.relative_to(u.ROOT))]=u.sha(receipt)
            reused.append(dict(arm=arm,fold=fold,receipt=str(receipt.relative_to(u.ROOT))))
    partial=study.RUN/'expanded/2'
    if {p.name for p in partial.iterdir()}!={'ridge.npz','scalers.npz'}:raise ValueError('Interrupted fold has completed model state requiring a different recovery; do not discard it')
    for p in partial.iterdir():sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    files=[Path(__file__),study.REPORT/'manifest.json',failure,u.ROOT/'SPLIT3_MATCHED_RESUME_PLAN.md',
        u.ROOT/'scripts/verify_upstream_split3_matched_resume.py',u.ROOT/'scripts/upstream_split3_matched_resume_queue.py']
    sources.update({str(p.relative_to(u.ROOT)):u.sha(p) for p in files});study.check_sources(sources)
    RUN.mkdir();REPORT.mkdir()
    u.write(REPORT/'manifest.json',dict(state='declared_preserved_seven_original_folds_and_three_missing_expanded_folds',config=CONFIG,
        original_manifest_sha256=u.sha(study.REPORT/'manifest.json'),source_sha256=sources,reused_original_folds=reused,
        pending=[dict(arm='expanded',fold=f) for f in [2,3,4]],versions=n.versions(),qualifying_labels_read=False,
        rule='Preserve all seven complete actual original fits without retraining. Original failed fold2 preprocessing remains untouched. Fit only missing expanded folds2/3/4 with exact same original full inputs/statistics, H128/r32,three seeds, max40/patience4, fresh selection/refit. Separate outputs and independent complete ten-fold original scaler/model replay.'))

def worker():
    import torch
    manifest=u.read(REPORT/'manifest.json')
    if manifest['config']!=CONFIG or manifest['versions']!=n.versions():raise ValueError('Frozen original recovery recipe/runtime changed')
    study.check_sources(manifest['source_sha256']);backend.MODEL_FACTORY=study.model_factory
    inputs=study.admission.Inputs();y,groups,canonical,mapping=study.arrays()
    with np.load(data.RUN/'stats.npz') as s:gram,rhs=s['gram'][1],s['rhs'][1]
    try:
        for fold in [2,3,4]:
            root=RUN/'expanded'/str(fold);root.mkdir(parents=True,exist_ok=False)
            train,fit,valid,iw,fw,ib,fb,inner,full,si,sf=study.fold_inputs(inputs,gram,rhs,groups,canonical,'expanded',fold)
            held=mapping[groups[mapping]//2==fold];correction=np.zeros(len(held));counts=[]
            np.savez(root/'ridge.npz',inner=iw,full=fw)
            np.savez(root/'scalers.npz',**{'inner_'+k:v for k,v in si.items()},**{'full_'+k:v for k,v in sf.items()})
            for seed in CONFIG['seeds']:
                actual=seed+fold;n.event(arm='expanded',fold=fold,stage='select',seed=seed,recovery=True)
                selected,count=backend.train_nn(inner[fit],y[fit]-ib[fit],CONFIG,actual,CONFIG['nn_epochs'],valid=(inner[valid],y[valid]-ib[valid]))
                del selected;torch.mps.empty_cache()
                if count:
                    model,_=backend.train_nn(full[train],y[train]-fb[train],CONFIG,actual,count)
                    correction+=backend.predict(model,full[held])/3
                    torch.save(dict(state=model.state_dict(),config=CONFIG,feature_width=210,seed=actual,epochs=count),root/f'nn-{seed}.pt')
                    del model;torch.mps.empty_cache()
                counts.append(count);n.event(arm='expanded',fold=fold,stage='refit_complete',seed=seed,iterations=count,recovery=True)
            prediction=np.clip(fb[held]+correction,1.05,4.95).astype('<f4');prediction.tofile(root/'prediction.f32')
            u.write(REPORT/'expanded'/f'{fold}.json',dict(arm='expanded',fold=fold,canonical_rows=len(held),chosen_iterations=counts,
                manifest_sha256=u.sha(study.REPORT/'manifest.json'),recovery_manifest_sha256=u.sha(REPORT/'manifest.json'),
                outputs={p.name:u.sha(p) for p in root.iterdir()},rmse=float(np.sqrt(np.mean((prediction.astype(np.float64)-y[held])**2)))))
            del train,fit,valid,ib,fb,inner,full,si,sf;gc.collect();torch.mps.empty_cache()
        study.check_sources(manifest['source_sha256'])
        u.write(REPORT/'fit-completion.json',dict(state='complete_three_missing_expanded_folds_preserving_seven_original_fits',qualifying_labels_read=False))
    finally:inputs.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args();worker() if a.worker else prepare()
