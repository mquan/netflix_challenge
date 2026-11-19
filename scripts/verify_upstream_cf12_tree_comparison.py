#!/usr/bin/env python3
"""Verify the complete matching-tree result against the preceding fixed mean."""
import math
from pathlib import Path
import numpy as np
import upstream_cf12_tree as experiment
from verify_upstream_context_partial import independent_rmse


def verify():
    e=experiment;u=e.u;e.configure();e.n.verify_prepared();old=e.parent
    proofs={};predictions={}
    for name,run,report in [('previous',old.RUN,old.REPORT),('matching',e.RUN,e.REPORT)]:
        proof=u.read(report/'independent-verification.json');evaluation=u.read(report/'evaluation.json')
        if proof['state']!='complete' or proof['evaluation_sha256']!=u.sha(report/'evaluation.json') or not proof['fixed_average_bitwise_equal']:
            raise ValueError('Complete independent evaluation binding required')
        row=next(r for r in evaluation['records'] if r['arm']=='fixed-half-tree-nn')
        path=run/'fixed-half-tree-nn.f32'
        if u.sha(path)!=row['prediction_sha256']:raise ValueError('Comparison vector changed')
        predictions[name]=np.fromfile(path,dtype='<f4').astype(np.float64);proofs[name]=proof
    for name in ['labels.npy','users.npy','groups.npy']:
        if u.sha(e.RUN/name)!=u.sha(old.RUN/name):raise ValueError('Comparison row identity differs')
    if u.sha(e.RUN/'nn.f32')!=u.sha(old.RUN/'nn.f32'):raise ValueError('Reused neural predictions changed')
    replay=u.read(e.REPORT/'tree-replay.json');nn_replay=u.read(old.REPORT/'nn-replay.json')
    for record in [replay,nn_replay]:
        if [r['fold'] for r in record['records']]!=list(range(5)) or not all(r['bitwise_equal'] for r in record['records']):
            raise ValueError('All five saved-model replays required')
    for family,run,report in [('tree',e.RUN,e.REPORT),('nn',old.RUN,old.REPORT)]:
        for fold in range(5):
            receipt=u.read(report/family/(str(fold)+'.json'))
            if receipt['manifest_sha256']!=u.sha(report/'manifest.json'):raise ValueError('Fold manifest changed')
            for name,digest in receipt['outputs'].items():
                if u.sha(run/family/str(fold)/name)!=digest:raise ValueError('Saved model changed')
    y=np.load(e.RUN/'labels.npy');users=np.load(e.RUN/'users.npy');folds=np.load(e.RUN/'groups.npy')//2
    scores={name:independent_rmse(p,y) for name,p in predictions.items()}
    for name,proof in proofs.items():
        checked=next(r for r in proof['records'] if r['arm']=='fixed-half-tree-nn')
        if abs(scores[name]-checked['independent_rmse'])>1e-12:raise ValueError('Independent comparison score differs')
    changes=[independent_rmse(predictions['matching'][folds==f],y[folds==f])-independent_rmse(predictions['previous'][folds==f],y[folds==f]) for f in range(5)]
    paired=(np.clip(predictions['matching'],1.05,4.95)-y)**2-(np.clip(predictions['previous'],1.05,4.95)-y)**2
    delta=math.fsum(map(float,paired))/len(y)
    _,inverse=np.unique(users,return_inverse=True);counts=np.bincount(inverse)
    centered=np.bincount(inverse,weights=paired)-counts*delta;clusters=len(counts)
    se=math.sqrt(clusters/(clusters-1)*float(centered@centered))/len(y)
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in [e.REPORT/'manifest.json',e.REPORT/'evaluation.json',
        e.REPORT/'independent-verification.json',e.REPORT/'tree-replay.json',old.REPORT/'manifest.json',
        old.REPORT/'evaluation.json',old.REPORT/'independent-verification.json',old.REPORT/'nn-replay.json']}
    result=dict(state='verified_complete_comparison',rows=len(y),users=clusters,independent_rmse=scores,
        delta_rmse=scores['matching']-scores['previous'],fold_rmse_deltas=changes,all_folds_improve=all(v<0 for v in changes),
        paired_delta_mse=delta,conditional_user_cluster_95_mse_interval=[delta-1.96*se,delta+1.96*se],
        all_tree_and_neural_artifacts_rehashed=True,neural_vector_unchanged=True,source_sha256=sources,
        verifier_sha256=u.sha(Path(__file__)),qualifying_labels_read=False,goal_complete=False,
        scope='Complete adaptive known-ProbeX development comparison. Intervals condition on selected models and this bank; no selection correction, official score or untouched Test claim.')
    u.write(e.REPORT/'comparison-previous.json',result);print(result,flush=True);return result


if __name__=='__main__':verify()
