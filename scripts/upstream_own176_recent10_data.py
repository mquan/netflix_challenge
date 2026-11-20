#!/usr/bin/env python3
"""Bounded complete own176 virtual inputs and canonical user-fold ridge diagnostics."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_repro as u
import upstream_nonlinear as n
import upstream_warm166 as original

RUN=u.ROOT/'artifacts/upstream-own176-recent10-data'
REPORT=u.ROOT/'experiments/upstream-own176-recent10-data'
RECENT=u.ROOT/'experiments/upstream-recent10-own-pairs-v2/verification.json'
ROWS=1128164
DESIGN=252
FEATURES=402

def score(p,y):
    return math.sqrt(math.fsum(float(v)*float(v) for v in np.clip(np.asarray(p,dtype=np.float64),1.05,4.95)-y)/len(y))

def check(sources):
    cache={}
    for key,h in sources.items():
        p=u.ROOT/key;s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity not in cache:cache[identity]=u.sha(p)
        if cache[identity]!=h:raise ValueError('Frozen original own input changed: '+key)

def sources():
    base=u.read(original.REPORT/'manifest.json');proof=u.read(original.REPORT/'cache-column-verification.json')
    recent=u.read(RECENT)
    if (base['prediction_columns']!=166 or base['support_columns']!=90 or base['design_columns']!=242 or base['nonlinear_features']!=392 or
        proof['state']!='complete_cache_columns' or proof['manifest_sha256']!=u.sha(original.REPORT/'manifest.json') or
        not proof['original_support_floats_preserved_bitwise'] or not proof['all_retained_and_new_columns_checked']):raise ValueError('Complete actual own166 source/operator proof required')
    if (recent['state']!='complete_ten_fresh_own_pairs_and_original_producer_lineage' or len(recent['rows'])!=10 or
        recent['qualifying_labels_read'] or recent['public_prediction_values_read']):raise ValueError('All ten independently verified complete own pairs required')
    bindings=dict(base['source_manifests_sha256'])
    def bind(key,h):
        if key in bindings and bindings[key]!=h:raise ValueError('Conflicting own source bindings')
        bindings[key]=h
    for key,h in recent['source_sha256'].items():bind(key,h)
    for name,h in base['prepared_sha256'].items():bind(str((original.RUN/name).relative_to(u.ROOT)),h)
    for p in [original.REPORT/'manifest.json',original.REPORT/'cache-column-verification.json',RECENT,Path(__file__),
        Path(n.__file__),u.ROOT/'OWN176_RECENT10_PLAN.md',u.ROOT/'scripts/verify_upstream_own176_recent10_data.py']:
        bind(str(p.relative_to(u.ROOT)),u.sha(p))
    return bindings,recent['rows']

class Inputs:
    def __init__(self,rows):
        self.design=np.load(original.RUN/'design.npy',mmap_mode='r');self.features=np.load(original.RUN/'features.npy',mmap_mode='r')
        if self.design.shape!=(ROWS,242) or self.features.shape!=(ROWS,392):raise ValueError('Original complete own166 input dimensions differ')
        self.raw=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in rows]
        if any(a.shape!=(ROWS,) or a.dtype!=np.float32 for a in self.raw):raise ValueError('Complete ten canonical own inputs required')
    def block(self,lo,hi):
        added=np.column_stack([np.clip(a[lo:hi],0,6) for a in self.raw])
        z=np.column_stack([self.design[lo:hi,:-1],added,self.design[lo:hi,-1:]])
        f=np.column_stack([self.features[lo:hi],added-np.float32(3.6)]).astype(np.float32)
        if z.shape[1]!=DESIGN or f.shape[1]!=FEATURES or not np.isfinite(z).all() or not np.isfinite(f).all():raise ValueError('Complete finite own176 operators required')
        return z,f

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing full own176 input/ridge evidence preserved')
    bindings,rows=sources();check(bindings);RUN.mkdir();REPORT.mkdir()
    for key in ['labels','users','groups']:u.link_or_copy(original.RUN/(key+'.npy'),RUN/(key+'.npy'))
    y=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy');inputs=Inputs(rows)
    grams=np.zeros((10,DESIGN,DESIGN));rhs=np.zeros((10,DESIGN));hashes=[hashlib.sha256(),hashlib.sha256()]
    for lo in range(0,ROWS,8192):
        hi=min(lo+8192,ROWS);z,f=inputs.block(lo,hi);hashes[0].update(z.tobytes(order='C'));hashes[1].update(f.tobytes(order='C'))
        for g in range(10):
            keep=groups[lo:hi]==g;a=z[keep];grams[g]+=a.T@a;rhs[g]+=a.T@y[lo:hi][keep]
    np.savez(RUN/'stats.npz',gram=grams,rhs=rhs)
    weights=np.stack([n.ridge_weights(grams,rhs,np.arange(10)//2!=fold,1000) for fold in range(5)])
    prediction=np.empty(ROWS,dtype='<f4')
    for lo in range(0,ROWS,8192):
        hi=min(lo+8192,ROWS);z,_=inputs.block(lo,hi)
        for fold in range(5):
            keep=groups[lo:hi]//2==fold;prediction[lo:hi][keep]=np.clip(z[keep]@weights[fold],1.05,4.95)
    prediction.tofile(RUN/'ridge.f32');np.save(RUN/'ridge-weights.npy',weights);check(bindings)
    u.write(REPORT/'manifest.json',dict(state='complete_all_own176_virtual_inputs_group_statistics_and_ridge_diagnostic',
        rows=ROWS,raw_predictions=176,learned_context_columns=90,design_columns=DESIGN,nonlinear_columns=FEATURES,nn_inputs=FEATURES+1,
        recent_rows=rows,source_sha256=bindings,virtual_c_order_sha256=[h.hexdigest() for h in hashes],
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},ridge_rmse=score(prediction,y),
        ridge_fold_rmse=[score(prediction[groups//2==fold],y[groups//2==fold]) for fold in range(5)],
        all_original_users_and_groups_retained=True,qualifying_labels_read=False,neural_models_fitted=0,
        scope='Complete canonical population, all original own166 inputs plus all ten recent own columns. Bounded exact virtual design/features, full original grouped sufficient statistics and five-fold ridge only. Not a neural CV promotion or qualifying score.'))
    print('Own176 full-data ridge diagnostic',score(prediction,y),'NN study remains mandatory',flush=True)

if __name__=='__main__':main()
