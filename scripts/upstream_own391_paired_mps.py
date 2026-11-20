#!/usr/bin/env python3
"""Full original-budget matched B391 neural study with declared H128/H512 arms."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import gc
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_own391_paired_data as data
import upstream_exact391_scaler as exact_stats
import upstream_own352_temporal_mps as previous

u,n,mps,scaling=data.u,previous.n,previous.mps,previous.scaling
SIGNALS=list(previous.SIGNALS[:-1])+list(range(578,617))+[617]
CONTEXT=list(previous.CONTEXT)
CONFIGS={name:dict(previous.CONFIG,nn_hidden=hidden,architecture='context-weighted-residual-own391-paired-'+name+'-v1')
    for name,hidden in [('h128',128),('h512',512)]}


def paths(variant):
    if variant not in CONFIGS:raise ValueError('Only the two originally declared neural arms permitted')
    return (u.ROOT/('artifacts/upstream-own391-paired-'+variant+'-mps'),u.ROOT/('experiments/upstream-own391-paired-'+variant+'-mps'))


def model_factory(width,cfg):
    import torch
    if width!=618 or cfg not in CONFIGS.values() or sorted(SIGNALS+CONTEXT)!=list(range(width)):
        raise ValueError('Original complete618-input paired391 recipe required')
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__();h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(SIGNALS));self.register_buffer('context',torch.tensor(CONTEXT))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(SIGNALS),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,values):
            factors=self.project(values.index_select(1,self.signals));weights=self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()


def hashes():
    files=[Path(__file__),Path(data.__file__),Path(exact_stats.__file__),Path(previous.__file__),Path(mps.__file__),
        Path(scaling.__file__),Path(n.__file__),u.ROOT/'scripts/verify_upstream_own391_paired_mps.py',data.DECL]
    files.append(u.ROOT/'scripts/verify_upstream_own391_paired_canonical.py')
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in files}


def prepare(variant):
    run,report=paths(variant)
    if run.exists() or report.exists():raise ValueError('Original complete or partial paired neural arm retained')
    if variant=='h512':
        proof=u.ROOT/'experiments/upstream-high-capacity-successor/completion.json'
        if not proof.exists() or u.read(proof)['state']!='complete_original1600_and_independent_full_output_verification':
            raise ValueError('Declared H512 control must wait for original1600 completion')
    mp=data.REPORT/'manifest.json';vp=data.REPORT/'independent-verification.json';m=u.read(mp);v=u.read(vp)
    if (v['state']!='complete_independent_matched_B391_all_operators_statistics_ridge_and_stored_predictions' or
        v['manifest_sha256']!=u.sha(mp) or not v['matched_B_known_and_qualifying'] or
        not v['every_full_operator_statistic_solve_and_prediction_bit_equal'] or v['qualifying_labels_read'] is not False or
        v['neural_models_fitted']!=0 or v['source_sha256']!=m['source_sha256'] or v['nn_inputs']!=618):
        raise ValueError('Full matched391 data/scalar/ridge replay required')
    bindings=dict(m['source_sha256'])
    reference_path=u.ROOT/'experiments/upstream-own352-bounded-replay/independent-canonical-verification.json'
    reference=u.read(reference_path)
    if reference['state']!='complete_separate_bounded_original352_tree_half_arithmetic_and_full_canonical_score':
        raise ValueError('Original full canonical development reference required')
    tree=next(row for row in u.read(previous.REPORT/'manifest.json')['controls'] if row['id']=='tree154')
    if tree['prediction'] not in bindings or u.sha(u.ROOT/tree['prediction'])!=bindings[tree['prediction']]:
        raise ValueError('Original fully replayed tree154 OOF control required')
    for collection in [hashes(),{str(mp.relative_to(u.ROOT)):u.sha(mp),str(vp.relative_to(u.ROOT)):u.sha(vp)}]:
        for key,value in collection.items():
            if key in bindings and bindings[key]!=value:raise ValueError('Original matched neural source expectation conflicts')
            bindings[key]=value
    for name,digest in m['prepared_sha256'].items():bindings[str((data.RUN/name).relative_to(u.ROOT))]=digest
    data.check(bindings);run.mkdir();report.mkdir()
    for name in ['labels.npy','users.npy','groups.npy','stats.npz']:u.link_or_copy(data.RUN/name,run/name)
    u.write(report/'manifest.json',dict(state='declared_original_budget_matched_B391_neural_arm',variant=variant,
        config=CONFIGS[variant],versions=n.versions(),code_sha256=hashes(),source_sha256=bindings,
        input_manifest_sha256=u.sha(mp),raw_predictions=391,design_columns=467,nonlinear_columns=617,nn_inputs=618,
        signal_indices=SIGNALS,context_indices=CONTEXT,matched_B_known_and_qualifying=True,
        controls=[tree],declared_global_reference_rmse=reference['rmse'],fixed_tree_weight=.5,
        qualifying_labels_read=False,rule='All391 original scalar predictors/all90 learned contexts; complete225 context positions. Original5userfolds/3seeds/max40pat4/batch8192/AdamW. Exact full eligible statistics with bounded scratch. No nonlinear OOF component inputs. Known-only architecture selection before any new qualifying candidate.'))


def inputs():
    m=u.read(data.REPORT/'manifest.json')
    return data.Inputs('probex',m['original_rows'],m['temporal_rows'],m['new_rows'])


def originals(inner_weights,full_weights):
    reader=inputs();inner=np.empty((1128164,618),dtype=np.float32);full=np.empty_like(inner)
    ib=np.empty(1128164);fb=np.empty(1128164);digests=[hashlib.sha256(),hashlib.sha256()]
    try:
        for lo in range(0,len(inner),8192):
            hi=min(lo+8192,len(inner));z,f=reader.block(lo,hi);digests[0].update(z.tobytes());digests[1].update(f.tobytes())
            ib[lo:hi]=np.clip(z@inner_weights,1.05,4.95);fb[lo:hi]=np.clip(z@full_weights,1.05,4.95)
            inner[lo:hi,:-1]=f;full[lo:hi,:-1]=f;inner[lo:hi,-1]=ib[lo:hi];full[lo:hi,-1]=fb[lo:hi]
    finally:reader.close()
    if [h.hexdigest() for h in digests]!=u.read(data.REPORT/'manifest.json')['virtual_c_order_sha256']['probex']:
        raise ValueError('Every complete matched paired391 input bit must match')
    return inner,full,ib,fb


def worker(variant):
    import torch
    run,report=paths(variant);cfg=CONFIGS[variant];manifest=u.read(report/'manifest.json');mps.MODEL_FACTORY=model_factory
    if manifest['config']!=cfg or manifest['code_sha256']!=hashes() or manifest['versions']!=n.versions():raise ValueError('Original matched recipe/runtime changed')
    if (report/'fit-completion.json').exists():raise ValueError('Existing original complete fit retained')
    data.check(manifest['source_sha256']);y=np.load(run/'labels.npy');groups=np.load(run/'groups.npy')
    with np.load(run/'stats.npz') as saved:grams,rhs=saved['gram'],saved['rhs']
    for fold in range(5):
        root=run/'nn'/str(fold);root.mkdir(parents=True,exist_ok=False)
        train=groups//2!=fold;held=~train;fit=train&(groups%2==0);valid=train&(groups%2==1);eligible=np.arange(10)//2!=fold
        iw=n.ridge_weights(grams,rhs,eligible&(np.arange(10)%2==0),cfg['ridge_lambda']);fw=n.ridge_weights(grams,rhs,eligible,cfg['ridge_lambda'])
        inner,full,ib,fb=originals(iw,fw)
        si=exact_stats.fit_scaler(inner,fit);scaling.apply_scaler(inner,si,output=inner)
        sf=exact_stats.fit_scaler(full,train);scaling.apply_scaler(full,sf,output=full)
        np.savez(root/'ridge.npz',inner=iw,full=fw);np.savez(root/'scalers.npz',**{'inner_'+k:v for k,v in si.items()},**{'full_'+k:v for k,v in sf.items()})
        correction=np.zeros(int(held.sum()));counts=[]
        for seed in cfg['seeds']:
            actual=seed+fold;n.event(stage='select',variant=variant,fold=fold,seed=seed)
            selected,count=mps.train_nn(inner[fit],y[fit]-ib[fit],cfg,actual,cfg['nn_epochs'],valid=(inner[valid],y[valid]-ib[valid]))
            del selected;torch.mps.empty_cache()
            if count:
                model,_=mps.train_nn(full[train],y[train]-fb[train],cfg,actual,count)
                correction+=mps.predict(model,full[held])/3
                torch.save(dict(state=model.state_dict(),config=cfg,feature_width=618,seed=actual,epochs=count),root/('nn-%s.pt'%seed))
                del model;torch.mps.empty_cache()
            counts.append(count);n.event(stage='refit_complete',variant=variant,fold=fold,seed=seed,iterations=count)
        prediction=np.clip(fb[held]+correction,1.05,4.95).astype('<f4');prediction.tofile(root/'prediction.f32')
        u.write(report/'nn'/('%s.json'%fold),dict(fold=fold,canonical_rows=len(prediction),chosen_iterations=counts,
            manifest_sha256=u.sha(report/'manifest.json'),outputs={p.name:u.sha(p) for p in root.iterdir()},rmse=data.score(prediction,y[held])))
        del inner,full,ib,fb;gc.collect();torch.mps.empty_cache()
    data.check(manifest['source_sha256']);u.write(report/'fit-completion.json',dict(state='complete_all_five_original_budget_matched_B391_neural_fits',variant=variant,qualifying_labels_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=CONFIGS,default='h128');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    worker(args.variant) if args.worker else prepare(args.variant)
