#!/usr/bin/env python3
"""Separate final own330/tree arithmetic, full canonical score and source binding."""
import datetime as dt
import math
from pathlib import Path
import numpy as np
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-own330-mps'
RUN=u.ROOT/'artifacts/upstream-own330-mps'


def main():
    destination=REPORT/'independent-canonical-verification.json'
    if destination.exists():raise ValueError('Existing separate canonical proof preserved')
    mp=REPORT/'manifest.json';manifest=u.read(mp)
    vp=REPORT/'independent-verification.json';verified=u.read(vp)
    if (verified['state']!='complete_actual_own330_all_fits_original_statistics_scalers_and_saved_model_replays' or
        verified['manifest_sha256']!=u.sha(mp) or len(verified['records'])!=5 or
        sum(r['actual_models_reopened'] for r in verified['records'])!=15 or
        not all(r['actual_saved_pipeline_bitwise_equal'] and r['both_original_eligible_scalers_recomputed'] and
                r['all_original_normalized_input_bits_equal'] for r in verified['records'])):
        raise ValueError('All15 actual models and complete eligible preprocessing replays required')
    bindings=dict(manifest['source_sha256'])
    for record in verified['records']:
        receipt=u.ROOT/record['receipt_path']
        if u.sha(receipt)!=record['receipt_sha256']:raise ValueError('Original model receipt changed')
        bindings[str(receipt.relative_to(u.ROOT))]=record['receipt_sha256']
        for name,digest in record['outputs'].items():
            path=RUN/'nn'/str(record['fold'])/name
            bindings[str(path.relative_to(u.ROOT))]=digest
    tree=u.ROOT/next(c['prediction'] for c in manifest['controls'] if c['id']=='tree154')
    nn=RUN/'nn.f32';half=RUN/'fixed-half-tree154.f32';labels=RUN/'labels.npy'
    if u.sha(nn)!=verified['nn_sha256'] or u.sha(half)!=verified['fixed_half_sha256']:
        raise ValueError('Actual complete candidate vectors changed')
    for path in [mp,vp,nn,half,tree,labels,RUN/'users.npy',RUN/'groups.npy',Path(__file__)]:
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    for rel,digest in bindings.items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original actual source/model changed: '+rel)
    a=np.fromfile(nn,dtype='<f4');b=np.fromfile(tree,dtype='<f4');stored=np.fromfile(half,dtype='<f4')
    y=np.load(labels)
    if any(v.shape!=(1128164,) for v in [a,b,stored,y]) or not all(np.isfinite(v).all() for v in [a,b,stored,y]):
        raise ValueError('Complete finite canonical population required')
    reference=np.empty_like(stored)
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y))
        reference[lo:hi]=np.clip((np.asarray(a[lo:hi],dtype=np.float64)+np.asarray(b[lo:hi],dtype=np.float64))/2,1.05,4.95)
    if not np.array_equal(reference.view('u4'),stored.view('u4')):
        raise ValueError('Every final fixed-average float32 bit must separately match')
    rmse=math.sqrt(math.fsum((float(p)-float(t))**2 for p,t in zip(np.clip(stored.astype(np.float64),1.05,4.95),y))/len(y))
    if abs(rmse-verified['fixed_half_tree154_rmse'])>1e-14:raise ValueError('Separate full canonical score differs')
    u.write(destination,dict(state='complete_separate_original330_tree_half_arithmetic_and_full_canonical_score',
            observed=dt.datetime.now(dt.timezone.utc).isoformat(),rmse=rmse,rows=len(y),
            complete_model_preprocessing_verification_sha256=u.sha(vp),manifest_sha256=u.sha(mp),
            verifier_sha256=u.sha(Path(__file__)),prediction_sha256=u.sha(half),source_sha256=bindings,
            all15_actual_saved_models_rehashed=True,every_fixed_average_float32_bit_equal=True,
            previous_declared_global_rmse=manifest['declared_global_reference_rmse'],
            improves_previous=rmse<manifest['declared_global_reference_rmse'],qualifying_labels_read=False,
            production_started=False,goal_complete=False,
            scope='Adaptive original canonical known CV. All complete original model/scaler replays and exact final arithmetic verified; no qualifying score, official or untouched Test claim.'))
    print('Independent complete own330/tree canonical score',rmse,flush=True)


if __name__=='__main__':main()
