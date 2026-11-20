#!/usr/bin/env python3
"""Matched canonical-only versus full split3 holdout neural fitting and replay."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import gc
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_repro as u
import upstream_nonlinear as n
import upstream_mps_context as backend
import upstream_exact_neural_scaling as scaling
import upstream_split3_matched_data as data
import upstream_split3_matched_development_inputs as admission
import upstream_warm166_mps_context as previous

RUN=u.ROOT/'artifacts/upstream-split3-matched-mps'
REPORT=u.ROOT/'experiments/upstream-split3-matched-mps'
CONFIG=dict(previous.CONFIG,architecture='context-residual-full-split3-matched-H128-r32-v1')
SIGNALS=list(range(63))+list(range(68,79))+[209]
CONTEXT=list(range(63,68))+list(range(79,209))
ARMS=['control','expanded']

def model_factory(width,cfg):
    import torch
    if width!=210 or cfg!=CONFIG or sorted(SIGNALS+CONTEXT)!=list(range(width)):raise ValueError('Frozen complete matched network required')
    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__();h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(SIGNALS));self.register_buffer('context',torch.tensor(CONTEXT))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(SIGNALS),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,x):return self.base(x)+(self.project(x.index_select(1,self.signals))*self.gate(x.index_select(1,self.context))).sum(1,keepdim=True)/math.sqrt(CONFIG['context_rank'])
    return Residual()

def code_hashes():
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(data.__file__),Path(admission.__file__),Path(n.__file__),Path(backend.__file__),Path(scaling.__file__),Path(previous.__file__),u.ROOT/'scripts/verify_upstream_split3_matched_mps.py']}

def check_sources(sources):
    cache={}
    for key,h in sources.items():
        p=u.ROOT/key;s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity not in cache:cache[identity]=u.sha(p)
        if cache[identity]!=h:raise ValueError('Frozen original input changed: '+key)

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing full matched study preserved')
    metadata=u.read(data.REPORT/'manifest.json');stats=u.read(data.REPORT/'statistics-verification.json')
    if (stats['state']!='complete_all_original_full_input_blocks_and_both_arm_group_statistics' or
        stats['manifest_sha256']!=u.sha(data.REPORT/'manifest.json') or stats['stats_sha256']!=u.sha(data.RUN/'stats.npz') or
        stats['voting_verification_sha256']!=u.sha(data.VOTES) or not metadata['all_additional_rows_of_held_users_excluded']):raise ValueError('Complete full matched data admission required')
    sources=dict(metadata['source_sha256']);sources.update(stats['source_sha256']);sources.update(u.read(u.ROOT/'experiments/upstream-split3-voting-stream/plan.json')['source_sha256'])
    for p in [data.REPORT/'manifest.json',data.REPORT/'statistics-verification.json',data.VOTES,data.RUN/'stats.npz',u.ROOT/'experiments/upstream-native-nn-pool6-tree31/independent-verification.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    sources.update({str((data.RUN/name).relative_to(u.ROOT)):h for name,h in metadata['prepared_sha256'].items()})
    controls=[]
    for family,name in [('nn','upstream-item154-mps-context'),('tree','upstream-item154-tree')]:
        root=u.ROOT/'artifacts'/name;report=u.ROOT/'experiments'/name;proof=u.read(report/'independent-verification.json');replay=u.read(report/(family+'-replay.json'))
        if proof['state']!='complete' or proof['evaluation_sha256']!=u.sha(report/'evaluation.json') or not all(r['bitwise_equal'] for r in replay['records']) or len(replay['records'])!=5:raise ValueError('Actual fixed control replay required')
        row=next(r for r in proof['records'] if r['arm']==family)
        p=root/(family+'.f32')
        if u.sha(p)!=row['prediction_sha256']:raise ValueError('Original control vector changed')
        for p in [p,report/'independent-verification.json',report/(family+'-replay.json'),report/'evaluation.json']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
        controls.append(dict(family=family,path=str((root/(family+'.f32')).relative_to(u.ROOT)),rmse=row['independent_rmse']))
    sources.update(code_hashes());check_sources(sources)
    RUN.mkdir();REPORT.mkdir()
    best=u.read(u.ROOT/'experiments/upstream-native-nn-pool6-tree31/independent-verification.json')
    u.write(REPORT/'manifest.json',dict(state='declared_full_matched_split3_original_five_fold_three_seed_study',config=CONFIG,
        code_sha256=code_hashes(),source_sha256=sources,controls=controls,preceding_global_development_rmse=best['rmse'],
        arms=ARMS,evaluation_rows=1128164,available_known_rows=data.ROWS,qualifying_labels_read=False,
        rule='Both arms identical complete73 own native raw plus130 original voting features and5metadata, H128/r32,5outer user folds,3seeds,max40/patience4,batch8192. All held-user additional labels excluded. Inner selection followed by fresh full eligible refit. Evaluate same canonical rows; fixed half with existing tree154. No automatic qualifying fit.'))

def arrays():return [np.load(data.RUN/(name+'.npy')) for name in ['labels','groups','canonical','mapping']]

def fold_inputs(inputs,gram,rhs,groups,canonical,arm,fold):
    training,fit,valid=data.masks(groups,canonical,arm,fold);eligible=np.arange(10)//2!=fold
    iw=n.ridge_weights(gram,rhs,eligible&(np.arange(10)%2==0),CONFIG['ridge_lambda'])
    fw=n.ridge_weights(gram,rhs,eligible,CONFIG['ridge_lambda'])
    inner=np.empty((data.ROWS,210),dtype=np.float32);full=np.empty_like(inner)
    ib=np.empty(data.ROWS);fb=np.empty(data.ROWS);hashes=[hashlib.sha256(),hashlib.sha256()]
    for lo in range(0,data.ROWS,8192):
        hi=min(lo+8192,data.ROWS);z,f=inputs.block(lo,hi)
        hashes[0].update(z.tobytes(order='C'));hashes[1].update(f.tobytes(order='C'))
        ib[lo:hi]=np.clip(z@iw,1.05,4.95);fb[lo:hi]=np.clip(z@fw,1.05,4.95)
        inner[lo:hi,:-1]=f;full[lo:hi,:-1]=f
        inner[lo:hi,-1]=ib[lo:hi];full[lo:hi,-1]=fb[lo:hi]
    stats=u.read(data.REPORT/'statistics-verification.json')
    if [h.hexdigest() for h in hashes]!=[stats['design_payload_sha256'],stats['feature_payload_sha256']]:raise ValueError('Every full original input bit must match frozen stream')
    si=scaling.fit_scaler(inner,fit);sf=scaling.fit_scaler(full,training)
    scaling.apply_scaler(inner,si,output=inner);scaling.apply_scaler(full,sf,output=full)
    return training,fit,valid,iw,fw,ib,fb,inner,full,si,sf

def worker():
    import torch
    manifest=u.read(REPORT/'manifest.json')
    if manifest['config']!=CONFIG or manifest['code_sha256']!=code_hashes():raise ValueError('Frozen actual study changed')
    check_sources(manifest['source_sha256']);backend.MODEL_FACTORY=model_factory
    inputs=admission.Inputs();y,groups,canonical,mapping=arrays()
    with np.load(data.RUN/'stats.npz') as s:grams,rhs=s['gram'],s['rhs']
    try:
        for ai,arm in enumerate(ARMS):
            for fold in range(5):
                root=RUN/arm/str(fold);root.mkdir(parents=True,exist_ok=False)
                train,fit,valid,iw,fw,ib,fb,inner,full,si,sf=fold_inputs(inputs,grams[ai],rhs[ai],groups,canonical,arm,fold)
                held=mapping[groups[mapping]//2==fold];correction=np.zeros(len(held));counts=[]
                np.savez(root/'ridge.npz',inner=iw,full=fw)
                np.savez(root/'scalers.npz',**{'inner_'+k:v for k,v in si.items()},**{'full_'+k:v for k,v in sf.items()})
                for seed in CONFIG['seeds']:
                    actual=seed+fold;n.event(arm=arm,fold=fold,stage='select',seed=seed)
                    selected,count=backend.train_nn(inner[fit],y[fit]-ib[fit],CONFIG,actual,CONFIG['nn_epochs'],valid=(inner[valid],y[valid]-ib[valid]))
                    del selected;torch.mps.empty_cache()
                    if count:
                        model,_=backend.train_nn(full[train],y[train]-fb[train],CONFIG,actual,count)
                        correction+=backend.predict(model,full[held])/len(CONFIG['seeds'])
                        torch.save(dict(state=model.state_dict(),config=CONFIG,feature_width=210,seed=actual,epochs=count),root/f'nn-{seed}.pt')
                        del model;torch.mps.empty_cache()
                    counts.append(count);n.event(arm=arm,fold=fold,stage='refit_complete',seed=seed,iterations=count)
                prediction=np.clip(fb[held]+correction,1.05,4.95).astype('<f4');prediction.tofile(root/'prediction.f32')
                u.write(REPORT/arm/f'{fold}.json',dict(arm=arm,fold=fold,canonical_rows=len(held),chosen_iterations=counts,
                    rmse=float(np.sqrt(np.mean((prediction.astype(np.float64)-y[held])**2))),manifest_sha256=u.sha(REPORT/'manifest.json'),
                    outputs={p.name:u.sha(p) for p in root.iterdir()}))
                del train,fit,valid,ib,fb,inner,full,si,sf;gc.collect();torch.mps.empty_cache()
        check_sources(manifest['source_sha256']);u.write(REPORT/'fit-completion.json',dict(state='complete_both_matched_arms_all_ten_outer_folds',qualifying_labels_read=False))
    finally:inputs.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
    worker() if args.worker else prepare()
