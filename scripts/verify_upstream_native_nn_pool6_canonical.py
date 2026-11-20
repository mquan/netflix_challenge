#!/usr/bin/env python3
"""Independent compensated arithmetic, canonical scoring and source replay binding."""
from pathlib import Path
import math
import numpy as np
import upstream_native_nn_pool6 as source
u=source.u

def main():
    report=source.REPORT;run=source.RUN;p=report/'verification.json';v=u.read(p)
    if (v['state']!='complete_fixed_six_native_nn_pool_development' or v['origins']!=source.ORIGINS or
        v['neural_weights']!=[1/6]*6 or v['tree_pool_weights']!=[.5,.5] or
        not v['all35_origin_fold_receipts_and_actual_models_bound'] or v['qualifying_labels_read'] is not False):
        raise ValueError('Fixed all-six original replayed family proof required')
    for name,digest in v['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original source/model/vector binding changed')
    arrays=[np.fromfile(u.ROOT/'artifacts'/name/'nn.f32',dtype='<f4') for name in source.ORIGINS]
    expected=np.fromiter((math.fsum(float(value) for value in row)/6 for row in zip(*arrays)),dtype='<f4',count=1128164)
    tree=np.fromfile(u.ROOT/'artifacts'/source.TREE/'tree.f32',dtype='<f4')
    hybrid=((tree.astype(np.float64)+expected.astype(np.float64))/2).astype('<f4')
    y=np.load(u.FRESH/'data/probex/ratings.npy');groups=np.load(u.ROOT/'artifacts'/source.ORIGINS[0]/'groups.npy');records=[]
    def score(a,b):
        values=np.clip(np.asarray(a,dtype=np.float64),1.05,4.95)
        return math.sqrt(math.fsum((float(x)-float(z))**2 for x,z in zip(values,b))/len(b))
    for arm,values in [('nn-pool',expected),('fixed-half-tree-pool',hybrid)]:
        path=run/(arm+'.f32');stored=np.fromfile(path,dtype='<f4');row=next(r for r in v['records'] if r['arm']==arm)
        if not np.array_equal(stored.view(np.uint32),values.view(np.uint32)) or u.sha(path)!=row['prediction_sha256']:
            raise ValueError('Every actual pooled prediction must agree bitwise')
        rmse=score(stored,y)
        if abs(rmse-row['rmse'])>1e-10:raise ValueError('Canonical compensated score differs')
        records.append(dict(arm=arm,rmse=rmse,prediction_sha256=u.sha(path)))
    base=arrays[0];control=score(base,y)
    folds=[dict(fold=f,control_nn_rmse=score(base[groups//2==f],y[groups//2==f]),
        pooled_nn_rmse=score(expected[groups//2==f],y[groups//2==f])) for f in range(5)]
    result=dict(state='complete_independent_canonical_fixed_six_native_nn_pool',observed=source.stamp(),
        verification_sha256=u.sha(p),verifier_sha256=u.sha(Path(__file__)),records=records,
        matched_nn_control_rmse=control,matched_nn_improves=records[0]['rmse']<control,
        matched_nn_fold_results=folds,all_five_matched_nn_folds_improve=all(r['pooled_nn_rmse']<r['control_nn_rmse'] for r in folds),
        global_development_best_improved=records[1]['rmse']<v['reference_development_rmse'],
        every_pooled_float32_bit_recomputed_with_compensated_sum=True,
        all_original_source_models_and_complete_replay_receipts_rehashed=True,
        canonical_float64_reclip_before_scoring=True,original_unreclipped_scores_preserved=True,
        rows=len(y),qualifying_labels_read=False,qualifying_score_claimed=False,goal_complete=False)
    path=report/'independent-canonical-verification.json'
    if path.exists():raise ValueError('Existing independent fixed pool proof preserved')
    u.write(path,result);print(result,flush=True)

if __name__=='__main__':main()
