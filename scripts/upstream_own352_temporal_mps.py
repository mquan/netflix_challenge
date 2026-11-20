#!/usr/bin/env python3
"""Full original-budget own352/all90-context neural study without large disk caches."""
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
import upstream_own352_temporal_data as data
import upstream_own330_mps as previous

RUN=u.ROOT/'artifacts/upstream-own352-temporal-mps'
REPORT=u.ROOT/'experiments/upstream-own352-temporal-mps'
CONFIG=dict(previous.CONFIG,architecture='context-weighted-residual-own352-temporal22-all90-mps-v1')
SIGNALS=list(previous.SIGNALS[:-1])+list(range(556,578))+[578]
CONTEXT=list(previous.CONTEXT)

def model_factory(width,cfg):
    import torch
    if width!=579 or cfg!=CONFIG or sorted(SIGNALS+CONTEXT)!=list(range(width)):raise ValueError('Complete frozen own352 partition/recipe required')
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
        u.ROOT/'scripts/verify_upstream_own352_temporal_mps.py',u.ROOT/'scripts/upstream_own352_temporal_queue.py']}


def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing complete or partial own352 study retained')
    manifest=u.read(data.REPORT/'manifest.json');p=data.REPORT/'independent-verification.json';proof=u.read(p)
    if (proof['state']!='complete_all_original352_temporal_inputs_statistics_solves_and_prediction_bits' or
        proof['manifest_sha256']!=u.sha(data.REPORT/'manifest.json') or
        not proof['all_original330_input_bits_preserved'] or not proof['all22_original_temporal_scalar_transforms_equal'] or
        not proof['full_group_statistics_and_all_eligible_solves_equal'] or not proof['every_actual_stored_prediction_bit_equal'] or
        manifest['nn_inputs']!=579 or len(manifest['recent_rows'])!=22):
        raise ValueError('Complete actual own352 operator/statistic/ridge replay required')
    sources=dict(manifest['source_sha256'])
    def bind(path,expected=None):
        key=str(path.relative_to(u.ROOT));digest=u.sha(path) if expected is None else expected
        if key in sources and sources[key]!=digest:raise ValueError('Original source expectation conflicts: '+key)
        sources[key]=digest
    for key,digest in hashes().items():bind(u.ROOT/key,digest)
    for path in [p,data.REPORT/'manifest.json',u.ROOT/'OWN352_TEMPORAL_PLAN.md']:bind(path)
    for name,digest in manifest['prepared_sha256'].items():bind(data.RUN/name,digest)
    prior_report=u.ROOT/'experiments/upstream-own330-mps';prior_run=u.ROOT/'artifacts/upstream-own330-mps'
    cp=prior_report/'independent-canonical-verification.json';canonical=u.read(cp)
    mp=prior_report/'independent-verification.json';actual=u.read(mp)
    pm=u.read(prior_report/'manifest.json')
    if (canonical['state']!='complete_separate_original330_tree_half_arithmetic_and_full_canonical_score' or
        canonical['complete_model_preprocessing_verification_sha256']!=u.sha(mp) or
        canonical['manifest_sha256']!=u.sha(prior_report/'manifest.json') or
        actual['state']!='complete_actual_own330_all_fits_original_statistics_scalers_and_saved_model_replays' or
        not canonical['all15_actual_saved_models_rehashed'] or not canonical['every_fixed_average_float32_bit_equal'] or
        len(actual['records'])!=5 or not all(r['actual_models_reopened']==3 and r['actual_saved_pipeline_bitwise_equal'] for r in actual['records'])):
        raise ValueError('Actual complete preceding own330 models and canonical control required')
    for key,digest in canonical['source_sha256'].items():bind(u.ROOT/key,digest)
    for path in [cp,mp,prior_report/'manifest.json']:bind(path)
    bind(prior_run/'nn.f32',actual['nn_sha256'])
    bind(prior_run/'fixed-half-tree154.f32',canonical['prediction_sha256'])
    for key in ['labels','users','groups']:
        if not np.array_equal(np.load(prior_run/(key+'.npy')),np.load(data.RUN/(key+'.npy'))):
            raise ValueError('Original own330 and new own352 canonical rows/userfolds differ')
    controls=[dict(id='nn330',family='nn',origin='upstream-own330-mps',rmse=actual['nn_rmse'],prediction=str((prior_run/'nn.f32').relative_to(u.ROOT))),
              next(r for r in pm['controls'] if r['id']=='tree154')]
    data.check(sources);RUN.mkdir();REPORT.mkdir()
    for name in ['labels.npy','users.npy','groups.npy','stats.npz']:u.link_or_copy(data.RUN/name,RUN/name)
    u.write(REPORT/'manifest.json',dict(state='declared_complete_own352_temporal_five_fold_three_seed_study',config=CONFIG,versions=n.versions(),code_sha256=hashes(),source_sha256=sources,
        controls=controls,declared_global_reference_rmse=canonical['rmse'],raw_predictions=352,context_columns=90,rows=1128164,design_columns=428,nonlinear_columns=578,nn_inputs=579,
        signal_indices=SIGNALS,context_indices=CONTEXT,fixed_tree_weight=.5,qualifying_labels_read=False,
        rule='Retain all original own330/all90 inputs; append all22 completed original temporal100r pairs without epoch selection. Full original data/userfolds/inner masks, H128/rank32,three seeds,max40/patience4,batch8192,AdamW. Independent complete eligible statistics/scalers/normalized inputs/actual saved models mandatory. One fixed tree154 half; no weight search, qualifying outcomes or automatic qualifying refit.'))

def originals(inner_weights,full_weights,rows):
    inp=data.Inputs(u.read(data.REPORT/'manifest.json')['original_rows'],rows);inner=np.empty((data.ROWS,579),dtype=np.float32);full=np.empty_like(inner)
    ib=np.empty(data.ROWS);fb=np.empty(data.ROWS);digests=[hashlib.sha256(),hashlib.sha256()]
    for lo in range(0,data.ROWS,8192):
        hi=min(lo+8192,data.ROWS);z,f=inp.block(lo,hi);digests[0].update(z.tobytes(order='C'));digests[1].update(f.tobytes(order='C'))
        ib[lo:hi]=np.clip(z@inner_weights,1.05,4.95);fb[lo:hi]=np.clip(z@full_weights,1.05,4.95)
        inner[lo:hi,:-1]=f;full[lo:hi,:-1]=f;inner[lo:hi,-1]=ib[lo:hi];full[lo:hi,-1]=fb[lo:hi]
    inp.close()
    if [h.hexdigest() for h in digests]!=u.read(data.REPORT/'manifest.json')['virtual_c_order_sha256']:raise ValueError('Every full original own352 input bit differs')
    return inner,full,ib,fb

def worker():
    import torch
    manifest=u.read(REPORT/'manifest.json');mps.MODEL_FACTORY=model_factory
    if manifest['code_sha256']!=hashes() or manifest['config']!=CONFIG or manifest['versions']!=n.versions():raise ValueError('Frozen complete own352 recipe/runtime changed')
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
                torch.save(dict(state=model.state_dict(),config=CONFIG,feature_width=579,seed=actual,epochs=count),root/f'nn-{seed}.pt')
                del model;torch.mps.empty_cache()
            counts.append(count);n.event(stage='refit_complete',fold=fold,seed=seed,iterations=count)
        prediction=np.clip(fb[held]+correction,1.05,4.95).astype('<f4');prediction.tofile(root/'prediction.f32')
        u.write(REPORT/'nn'/f'{fold}.json',dict(fold=fold,canonical_rows=len(prediction),chosen_iterations=counts,manifest_sha256=u.sha(REPORT/'manifest.json'),
            outputs={p.name:u.sha(p) for p in root.iterdir()},rmse=data.score(prediction,y[held])))
        del inner,full,ib,fb;gc.collect();torch.mps.empty_cache()
    data.check(manifest['source_sha256']);u.write(REPORT/'fit-completion.json',dict(state='complete_all_five_original_budget_own352_fits',qualifying_labels_read=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args();worker() if a.worker else prepare()
