#!/usr/bin/env python3
"""Full original-budget own176/all90-context neural study without large disk caches."""
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
import upstream_mps_context as mps
import upstream_exact_neural_scaling as scaling
import upstream_own176_recent10_data as data
import upstream_warm166_mps_context as previous

RUN=u.ROOT/'artifacts/upstream-own176-recent10-mps'
REPORT=u.ROOT/'experiments/upstream-own176-recent10-mps'
CONFIG=dict(previous.CONFIG,architecture='context-weighted-residual-own176-recent10-all90-mps-v1')
SIGNALS=list(previous.SIGNALS[:-1])+list(range(392,402))+[402]
CONTEXT=list(previous.CONTEXT)

def model_factory(width,cfg):
    import torch
    if width!=403 or cfg!=CONFIG or sorted(SIGNALS+CONTEXT)!=list(range(width)):raise ValueError('Complete frozen own176 partition/recipe required')
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

def hashes():
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(data.__file__),Path(n.__file__),Path(mps.__file__),Path(scaling.__file__),Path(previous.__file__),
        u.ROOT/'scripts/verify_upstream_own176_recent10_mps.py',u.ROOT/'scripts/upstream_own176_recent10_queue.py']}

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing complete own176 study retained')
    manifest=u.read(data.REPORT/'manifest.json');p=data.REPORT/'independent-verification.json';proof=u.read(p)
    novelty=data.REPORT/'canonical-novelty-verification.json';distinct=u.read(novelty)
    if (proof['state']!='complete_all_own176_retained_added_operator_bits_original_statistics_and_ridge_replay' or
        proof['manifest_sha256']!=u.sha(data.REPORT/'manifest.json') or not proof['all_actual_ridge_prediction_bits_equal'] or
        distinct['manifest_sha256']!=u.sha(data.REPORT/'manifest.json') or not distinct['all_ten_canonical_additions_distinct_from_original']):raise ValueError('Complete actual own176 operator/ridge and novelty gates required')
    sources=dict(manifest['source_sha256']);sources.update(hashes())
    for path in [p,novelty,data.REPORT/'manifest.json',u.ROOT/'OWN176_RECENT10_PLAN.md']:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,h in manifest['prepared_sha256'].items():sources[str((data.RUN/name).relative_to(u.ROOT))]=h
    controls=[]
    for ident,family,name in [('nn166','nn','upstream-warm166-mps-context'),('nn154','nn','upstream-item154-mps-context'),('tree154','tree','upstream-item154-tree')]:
        root=u.ROOT/'artifacts'/name;report=u.ROOT/'experiments'/name
        verified=u.read(report/'independent-verification.json');replay=u.read(report/(family+'-replay.json'))
        if verified['state']!='complete' or verified['evaluation_sha256']!=u.sha(report/'evaluation.json') or len(replay['records'])!=5 or not all(r['bitwise_equal'] for r in replay['records']):raise ValueError('Actual full original control model replay required')
        for key in ['labels','users','groups']:
            if not np.array_equal(np.load(root/(key+'.npy')),np.load(data.RUN/(key+'.npy'))):raise ValueError('Original control canonical populations/partitions differ')
        row=next(r for r in verified['records'] if r['arm']==family);vector=root/(family+'.f32')
        if u.sha(vector)!=row['prediction_sha256']:raise ValueError('Actual original control vector changed')
        for path in [report/'manifest.json',report/'evaluation.json',report/'independent-verification.json',report/(family+'-replay.json'),vector]:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        controls.append(dict(id=ident,family=family,origin=name,rmse=row['independent_rmse'],prediction=str(vector.relative_to(u.ROOT))))
    global_path=u.ROOT/'experiments/upstream-native-nn-pool6-tree31/independent-verification.json';global_proof=u.read(global_path)
    if global_proof['state']!='complete_separate_fixed_tree25_pool75_and_canonical_score':raise ValueError('Original declared global reference must remain verified')
    sources[str(global_path.relative_to(u.ROOT))]=u.sha(global_path);data.check(sources)
    RUN.mkdir();REPORT.mkdir()
    for name in ['labels.npy','users.npy','groups.npy','stats.npz']:u.link_or_copy(data.RUN/name,RUN/name)
    u.write(REPORT/'manifest.json',dict(state='declared_complete_own176_recent10_five_fold_three_seed_study',config=CONFIG,versions=n.versions(),code_sha256=hashes(),source_sha256=sources,
        controls=controls,declared_global_reference_rmse=global_proof['rmse'],raw_predictions=176,context_columns=90,rows=1128164,design_columns=252,nonlinear_columns=402,nn_inputs=403,
        signal_indices=SIGNALS,context_indices=CONTEXT,fixed_tree_weight=.5,qualifying_labels_read=False,
        rule='All ten new own scalar pairs added to original own166/all90; original complete data/userfolds/inner masks, H128/rank32,three seeds,max40/patience4,batch8192,AdamW. Independent original scaler/statistic/model replay mandatory. One fixed tree154 half; no subset/weight search or automatic qualifying refit.'))

def originals(inner_weights,full_weights,rows):
    inp=data.Inputs(rows);inner=np.empty((data.ROWS,403),dtype=np.float32);full=np.empty_like(inner)
    ib=np.empty(data.ROWS);fb=np.empty(data.ROWS);digests=[hashlib.sha256(),hashlib.sha256()]
    for lo in range(0,data.ROWS,8192):
        hi=min(lo+8192,data.ROWS);z,f=inp.block(lo,hi);digests[0].update(z.tobytes(order='C'));digests[1].update(f.tobytes(order='C'))
        ib[lo:hi]=np.clip(z@inner_weights,1.05,4.95);fb[lo:hi]=np.clip(z@full_weights,1.05,4.95)
        inner[lo:hi,:-1]=f;full[lo:hi,:-1]=f;inner[lo:hi,-1]=ib[lo:hi];full[lo:hi,-1]=fb[lo:hi]
    if [h.hexdigest() for h in digests]!=u.read(data.REPORT/'manifest.json')['virtual_c_order_sha256']:raise ValueError('Every full original own176 input bit differs')
    return inner,full,ib,fb

def worker():
    import torch
    manifest=u.read(REPORT/'manifest.json');mps.MODEL_FACTORY=model_factory
    if manifest['code_sha256']!=hashes() or manifest['config']!=CONFIG or manifest['versions']!=n.versions():raise ValueError('Frozen complete own176 recipe/runtime changed')
    data.check(manifest['source_sha256']);rows=u.read(data.REPORT/'manifest.json')['recent_rows']
    y=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy')
    with np.load(RUN/'stats.npz') as saved:grams,rhs=saved['gram'],saved['rhs']
    for fold in range(5):
        root=RUN/'nn'/str(fold);root.mkdir(parents=True,exist_ok=False)
        train=groups//2!=fold;held=~train;fit=train&(groups%2==0);valid=train&(groups%2==1);eligible=np.arange(10)//2!=fold
        iw=n.ridge_weights(grams,rhs,eligible&(np.arange(10)%2==0),CONFIG['ridge_lambda']);fw=n.ridge_weights(grams,rhs,eligible,CONFIG['ridge_lambda'])
        inner,full,ib,fb=originals(iw,fw,rows)
        si=scaling.fit_scaler(inner,fit);scaling.apply_scaler(inner,si,output=inner)
        sf=scaling.fit_scaler(full,train);scaling.apply_scaler(full,sf,output=full)
        np.savez(root/'ridge.npz',inner=iw,full=fw);np.savez(root/'scalers.npz',**{'inner_'+k:v for k,v in si.items()},**{'full_'+k:v for k,v in sf.items()})
        correction=np.zeros(int(held.sum()));counts=[]
        for seed in CONFIG['seeds']:
            actual=seed+fold;n.event(stage='select',fold=fold,seed=seed)
            selected,count=mps.train_nn(inner[fit],y[fit]-ib[fit],CONFIG,actual,CONFIG['nn_epochs'],valid=(inner[valid],y[valid]-ib[valid]))
            del selected;torch.mps.empty_cache()
            if count:
                model,_=mps.train_nn(full[train],y[train]-fb[train],CONFIG,actual,count)
                correction+=mps.predict(model,full[held])/3
                torch.save(dict(state=model.state_dict(),config=CONFIG,feature_width=403,seed=actual,epochs=count),root/f'nn-{seed}.pt')
                del model;torch.mps.empty_cache()
            counts.append(count);n.event(stage='refit_complete',fold=fold,seed=seed,iterations=count)
        prediction=np.clip(fb[held]+correction,1.05,4.95).astype('<f4');prediction.tofile(root/'prediction.f32')
        u.write(REPORT/'nn'/f'{fold}.json',dict(fold=fold,canonical_rows=len(prediction),chosen_iterations=counts,manifest_sha256=u.sha(REPORT/'manifest.json'),
            outputs={p.name:u.sha(p) for p in root.iterdir()},rmse=data.score(prediction,y[held])))
        del inner,full,ib,fb;gc.collect();torch.mps.empty_cache()
    data.check(manifest['source_sha256']);u.write(REPORT/'fit-completion.json',dict(state='complete_all_five_original_budget_own176_fits',qualifying_labels_read=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args();worker() if a.worker else prepare()
