#!/usr/bin/env python3
"""Separate exact weighted-vector and canonical compensated score check."""
from pathlib import Path
import math
import numpy as np
import upstream_native_nn_pool6_tree31 as source
u=source.u

def main():
    p=source.REPORT/'verification.json';r=u.read(p)
    if r['state']!='complete_fixed_tree25_pool75_development' or r['weights']!=dict(tree=.25,nn_pool=.75) or r['qualifying_labels_read'] is not False:
        raise ValueError('Exact fixed known-only follow-up required')
    for name,digest in r['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Actual original model/source evidence changed')
    pool=np.fromfile(source.pool.RUN/'nn-pool.f32',dtype='<f4')
    tree=np.fromfile(u.ROOT/'artifacts'/source.pool.TREE/'tree.f32',dtype='<f4')
    expected=np.fromiter((math.fsum([float(a)/4,3*float(b)/4]) for a,b in zip(tree,pool)),dtype='<f4',count=1128164)
    path=source.RUN/'fixed-tree-pool.f32';stored=np.fromfile(path,dtype='<f4')
    if not np.array_equal(expected.view(np.uint32),stored.view(np.uint32)) or u.sha(path)!=r['prediction_sha256']:
        raise ValueError('Every stored weighted float32 prediction differs')
    y=np.load(u.FRESH/'data/probex/ratings.npy');values=np.clip(stored.astype(np.float64),1.05,4.95)
    rmse=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(values,y))/len(y))
    if abs(rmse-r['rmse'])>1e-12:raise ValueError('Canonical compensated score differs')
    result=dict(state='complete_separate_fixed_tree25_pool75_and_canonical_score',observed=source.pool.stamp(),
        rmse=rmse,verification_sha256=u.sha(p),verifier_sha256=u.sha(Path(__file__)),
        prediction_sha256=u.sha(path),rows=len(y),every_weighted_float32_bit_equal=True,
        all_original_model_replay_proofs_and_sources_rehashed=True,qualifying_labels_read=False,
        goal_complete=False,scope='Separate pure arithmetic and full canonical known score, bound to all actual original source models/folds/replays. Adaptive CV, no qualifying score or refit claimed.')
    out=source.REPORT/'independent-verification.json'
    if out.exists():raise ValueError('Existing independent fixed-weight proof preserved')
    u.write(out,result);print(result,flush=True)

if __name__=='__main__':main()
