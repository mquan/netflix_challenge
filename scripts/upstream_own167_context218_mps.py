#!/usr/bin/env python3
"""Full nested own167/context218 study with exact RAM-based preprocessing."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import datetime as dt
import gc
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import upstream_repro as u
import upstream_nonlinear as n
import upstream_mps_context as mps
import upstream_exact_neural_scaling as scaling
import upstream_own167_context218_features as data
import upstream_warm166_mps_context as previous

RUN=u.ROOT/'artifacts/upstream-own167-context218-mps'
REPORT=u.ROOT/'experiments/upstream-own167-context218-mps'
CONFIG=dict(previous.CONFIG,architecture='context-weighted-residual-own167-context218-mps-v1')
SIGNALS=list(range(63))+list(range(68,91))+list(range(221,303))+[521]
CONTEXT=list(previous.base.CONTEXT)+list(range(303,521))
CONTROLS=[('nn154','nn','upstream-item154-mps-context'),('nn166','nn','upstream-warm166-mps-context'),
    ('tree154','tree','upstream-item154-tree')]

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def model_factory(width,cfg):
    import torch
    if width!=522 or cfg!=CONFIG:raise ValueError('Frozen full own167/context218 recipe required')
    if sorted(SIGNALS+CONTEXT)!=list(range(width)):raise ValueError('Exact disjoint complete signal/context partition required')
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__();h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(SIGNALS));self.register_buffer('context',torch.tensor(CONTEXT))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),torch.nn.Dropout(d),
                torch.nn.Linear(h,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(SIGNALS),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(CONTEXT),h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:
                torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,values):
            factors=self.project(values.index_select(1,self.signals));weights=self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()

def configure():mps.MODEL_FACTORY=model_factory

def code_hashes():
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(data.__file__),Path(n.__file__),
        Path(mps.__file__),Path(scaling.__file__),Path(previous.__file__),
        u.ROOT/'scripts/verify_upstream_own167_context218_mps.py']}

def verify_sources(sources):
    cache={};signatures={}
    for key,expected in sources.items():
        p=u.ROOT/key;s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity not in cache:cache[identity]=u.sha(p)
        if cache[identity]!=expected:raise ValueError('Original source changed: '+key)
        signatures[key]=identity
    return signatures

def unchanged(signatures):
    for key,expected in signatures.items():
        s=(u.ROOT/key).stat()
        if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=tuple(expected):raise ValueError('Source changed during worker: '+key)

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing full study preserved')
    sources,arrays=data.sources();del arrays
    dp=data.REPORT/'independent-verification.json';d=u.read(dp)
    norm=u.ROOT/'experiments/upstream-own167-context218-normalization/verification.json';p=u.read(norm)
    if (d['state']!='complete_independent_all_columns_and_original_group_statistics' or
        d['manifest_sha256']!=u.sha(data.REPORT/'manifest.json') or
        not d['original_reference_all_ridge_group_statistic_bits_equal'] or
        p['state']!='complete_actual522_input_eligible_statistics_and_normalization_parity' or
        [r['arm'] for r in p['records']]!=['inner','full'] or not all(r['columns']==522 and
            r['original_all_statistical_bits_equal'] and r['every_normalized_float32_bit_equal'] for r in p['records']) or
        p['qualifying_labels_read'] is not False):raise ValueError('Complete actual original input/normalization parity required')
    for collection in [p['source_sha256'],code_hashes()]:
        for key,digest in collection.items():
            if key in sources and sources[key]!=digest:raise ValueError('Conflicting source provenance')
            sources[key]=digest
    for path in [dp,norm,data.REPORT/'manifest.json',u.ROOT/'OWN167_CONTEXT218_PLAN.md']:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    dm=u.read(data.REPORT/'manifest.json')
    for name,digest in dm['prepared_sha256'].items():sources[str((data.RUN/name).relative_to(u.ROOT))]=digest
    controls=[]
    for ident,family,name in CONTROLS:
        report=u.ROOT/'experiments'/name;cache=u.ROOT/'artifacts'/name
        proof=u.read(report/'independent-verification.json');replay=u.read(report/(family+'-replay.json'))
        if proof['state']!='complete' or proof['evaluation_sha256']!=u.sha(report/'evaluation.json') or not proof['fixed_average_bitwise_equal']:
            raise ValueError('Actual original control verification required')
        if [r['fold'] for r in replay['records']]!=list(range(5)) or not all(r['bitwise_equal'] for r in replay['records']):raise ValueError('Actual original control model replays required')
        for key in ['labels','users','groups']:
            if not np.array_equal(np.load(cache/(key+'.npy')),np.load(data.RUN/(key+'.npy'))):raise ValueError('Control rows/folds differ')
        row=next(r for r in proof['records'] if r['arm']==family);path=cache/(family+'.f32')
        if u.sha(path)!=row['prediction_sha256']:raise ValueError('Actual control vector changed')
        for path in [report/'manifest.json',report/'evaluation.json',report/'independent-verification.json',report/(family+'-replay.json'),path]:
            sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        controls.append(dict(id=ident,family=family,origin=name,rmse=row['independent_rmse'],prediction=str((cache/(family+'.f32')).relative_to(u.ROOT))))
    global_path=u.ROOT/'experiments/upstream-native-nn-pool6-tree31/independent-verification.json';global_best=u.read(global_path)
    if global_best['state']!='complete_separate_fixed_tree25_pool75_and_canonical_score':raise ValueError('Verified preceding global development best required')
    sources[str(global_path.relative_to(u.ROOT))]=u.sha(global_path)
    verify_sources(sources);RUN.mkdir();REPORT.mkdir()
    for name in ['labels.npy','users.npy','groups.npy','stats.npz']:u.link_or_copy(data.RUN/name,RUN/name)
    manifest=dict(state='declared_full_five_fold_own167_context218_study',created=stamp(),config=CONFIG,
        source_commit=u.PIN,versions=n.versions(),code_sha256=code_hashes(),source_sha256=sources,controls=controls,
        signal_indices=SIGNALS,context_indices=CONTEXT,rows=1128164,raw_predictions=167,learned_context_columns=218,
        nonlinear_columns=521,nn_inputs=522,design_columns=243,qualifying_labels_read=False,
        preceding_global_development_rmse=global_best['rmse'],fixed_tree_weight=.5,
        prepared_sha256={name:u.sha(RUN/name) for name in ['labels.npy','users.npy','groups.npy','stats.npz']},
        rule='Original full data,five outer user folds,original inner user partitions,seeds2026/7/13,H128/rank32,AdamW,batch8192,max40/patience4. Fresh select then refit each seed as original; no persistent large input cache. One fixed half with actual tree154. No qualifying access or automatic production.')
    u.write(REPORT/'manifest.json',manifest);return manifest

def open_arrays():
    pair=u.read(u.ROOT/'experiments/upstream-mps-item-autorec/pair-00050.json')
    paths=[data.PARENT/'design.npy',data.PARENT/'features.npy',u.ROOT/pair['pairs']['probex'],
        u.ROOT/'artifacts/upstream-item64-support/probex.npy',u.ROOT/'artifacts/upstream-itemae50-user64-support/probex.npy']
    return tuple(np.load(p,mmap_mode='r') for p in paths)

def design():
    arrays=open_arrays();out=np.empty((1128164,243),dtype=np.float64)
    for lo in range(0,len(out),8192):
        hi=min(lo+8192,len(out));out[lo:hi,:241]=arrays[0][lo:hi,:241]
        out[lo:hi,241]=np.clip(arrays[2][lo:hi],0,6);out[lo:hi,242]=arrays[0][lo:hi,241]
    expected=u.read(data.REPORT/'manifest.json')['virtual_c_order_sha256'][0]
    if hashlib.sha256(memoryview(out).cast('B')).hexdigest()!=expected:raise ValueError('Complete original ridge design bits differ')
    return out

def inputs(inner_base,full_base):
    arrays=open_arrays();inner=np.empty((1128164,522),dtype=np.float32);full=np.empty_like(inner);digest=hashlib.sha256()
    for lo in range(0,len(inner),8192):
        hi=min(lo+8192,len(inner));_,f=data.block(arrays,lo,hi);digest.update(f.tobytes(order='C'))
        inner[lo:hi,:-1]=f;full[lo:hi,:-1]=f
        inner[lo:hi,-1]=inner_base[lo:hi];full[lo:hi,-1]=full_base[lo:hi]
    if digest.hexdigest()!=u.read(data.REPORT/'manifest.json')['virtual_c_order_sha256'][1]:raise ValueError('Complete original nonlinear feature bits differ')
    return inner,full

def fit_fold(y,groups,grams,rhs,fold):
    import torch
    configure();root=RUN/'nn'/str(fold);root.mkdir(parents=True,exist_ok=False)
    training=groups//2!=fold;held=~training;fit=training&(groups%2==0);valid=training&(groups%2==1)
    eligible=np.arange(10)//2!=fold
    iw=n.ridge_weights(grams,rhs,eligible&(np.arange(10)%2==0),CONFIG['ridge_lambda'])
    fw=n.ridge_weights(grams,rhs,eligible,CONFIG['ridge_lambda'])
    z=design();inner_base=np.clip(z@iw,1.05,4.95);base=np.clip(z@fw,1.05,4.95);del z;gc.collect()
    np.savez(root/'ridge.npz',inner=iw,full=fw)
    inner_x,full_x=inputs(inner_base,base);gc.collect()
    inner_scaler=scaling.fit_scaler(inner_x,fit);scaling.apply_scaler(inner_x,inner_scaler,output=inner_x)
    full_scaler=scaling.fit_scaler(full_x,training);scaling.apply_scaler(full_x,full_scaler,output=full_x)
    np.savez(root/'scalers.npz',**{'inner_'+k:v for k,v in inner_scaler.items()},**{'full_'+k:v for k,v in full_scaler.items()})
    correction=np.zeros(int(held.sum()),dtype=np.float64);counts=[]
    for seed in CONFIG['seeds']:
        actual_seed=seed+fold;n.event(stage='select',fold=fold,seed=seed)
        selected,count=mps.train_nn(inner_x[fit],y[fit]-inner_base[fit],CONFIG,actual_seed,
            CONFIG['nn_epochs'],valid=(inner_x[valid],y[valid]-inner_base[valid]))
        del selected;torch.mps.empty_cache()
        if count:
            model,_=mps.train_nn(full_x[training],y[training]-base[training],CONFIG,actual_seed,count)
            correction+=mps.predict(model,full_x[held])/len(CONFIG['seeds'])
            torch.save(dict(state=model.state_dict(),config=CONFIG,feature_width=522,seed=actual_seed,epochs=count),root/f'nn-{seed}.pt')
            del model;torch.mps.empty_cache()
        counts.append(count);n.event(stage='refit_complete',fold=fold,seed=seed,iterations=count)
    result=np.clip(base[held]+correction,1.05,4.95).astype('<f4');result.tofile(root/'prediction.f32')
    return dict(fold=fold,rows=len(result),family='nn',chosen_iterations=counts,
        rmse=float(np.sqrt(np.mean((result.astype(np.float64)-y[held])**2))),
        outputs={p.name:u.sha(p) for p in root.iterdir()})

def worker():
    manifest=u.read(REPORT/'manifest.json')
    if manifest['config']!=CONFIG or manifest['code_sha256']!=code_hashes() or manifest['versions']!=n.versions():raise ValueError('Frozen full study recipe changed')
    bindings=dict(manifest['source_sha256'])
    bindings.update({str((RUN/name).relative_to(u.ROOT)):digest for name,digest in manifest['prepared_sha256'].items()})
    signatures=verify_sources(bindings)
    y=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy')
    with np.load(RUN/'stats.npz') as stats:grams,rhs=stats['gram'],stats['rhs']
    for fold in range(5):
        path=REPORT/'nn'/f'{fold}.json'
        if path.exists():raise ValueError('Existing complete fold preserved; no implicit restart')
        unchanged(signatures);start=time.monotonic();record=fit_fold(y,groups,grams,rhs,fold)
        record.update(manifest_sha256=u.sha(REPORT/'manifest.json'),seconds=time.monotonic()-start)
        u.write(path,record);n.event(stage='fold_complete',**record);gc.collect()
    verify_sources(bindings)
    u.write(REPORT/'fit-completion.json',dict(state='complete_all_five_original_budget_nested_folds',qualifying_labels_read=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    worker() if args.worker else prepare()
