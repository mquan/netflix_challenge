#!/usr/bin/env python3
"""Replay one completed matching-tree fold; no full-result promotion."""
import argparse
from pathlib import Path
import lightgbm as lgb
import numpy as np
import upstream_cf12_tree as experiment
from verify_upstream_context_partial import independent_rmse


def verify(fold):
    e=experiment;u=e.u;e.configure();manifest=e.n.verify_prepared()
    receipt_path=e.REPORT/'tree'/(str(fold)+'.json');receipt=u.read(receipt_path)
    root=e.RUN/'tree'/str(fold)
    if receipt['manifest_sha256']!=u.sha(e.REPORT/'manifest.json'):
        raise ValueError('Completed tree fold recipe changed')
    for name,digest in receipt['outputs'].items():
        if u.sha(root/name)!=digest:raise ValueError('Completed tree artifact changed')
    groups=np.load(e.RUN/'groups.npy');held=groups//2==fold
    z=np.load(e.RUN/'design.npy',mmap_mode='r')
    with np.load(root/'ridge.npz') as saved:base=np.clip(z@saved['full'],1.05,4.95)
    features=np.load(e.RUN/'features.npy',mmap_mode='r')
    inputs=np.column_stack([features[held],base[held]]).astype(np.float32)
    correction=np.zeros(int(held.sum()))
    for seed,iterations in zip(e.CONFIG['seeds'],receipt['chosen_iterations']):
        if iterations:
            model=lgb.Booster(model_file=str(root/f'tree-{seed}.txt'))
            correction+=model.predict(inputs,num_threads=e.CONFIG['threads'])/len(e.CONFIG['seeds'])
    actual=np.clip(base[held]+correction,1.05,4.95).astype('<f4')
    stored=np.fromfile(root/'prediction.f32',dtype='<f4')
    if actual.shape!=(receipt['rows'],) or not np.array_equal(actual,stored):
        raise ValueError('Completed tree fold replay differs')
    old=e.parent;old_report=u.read(old.REPORT/'evaluation.json');proof=u.read(old.REPORT/'independent-verification.json')
    if proof['evaluation_sha256']!=u.sha(old.REPORT/'evaluation.json') or not proof['fixed_average_bitwise_equal']:
        raise ValueError('Preceding best verification changed')
    for name in ['labels.npy','users.npy','groups.npy']:
        if u.sha(e.RUN/name)!=u.sha(old.RUN/name):raise ValueError('Preceding best row identity differs')
    for arm in ['nn','fixed-half-tree-nn']:
        row=next(r for r in old_report['records'] if r['arm']==arm)
        if u.sha(old.RUN/(arm+'.f32'))!=row['prediction_sha256']:raise ValueError('Preceding vector changed')
    if u.sha(e.RUN/'nn.f32')!=manifest['reused_nn_prediction_sha256']:
        raise ValueError('Reused NN vector changed')
    nn=np.fromfile(e.RUN/'nn.f32',dtype='<f4')[held].astype(np.float64)
    fixed=((actual.astype(np.float64)+nn)/2).astype('<f4')
    labels=np.load(e.RUN/'labels.npy')[held]
    scores=dict(tree=independent_rmse(actual,labels),fixed_half=independent_rmse(fixed,labels),
        preceding_fixed_half=independent_rmse(np.fromfile(old.RUN/'fixed-half-tree-nn.f32',dtype='<f4')[held],labels))
    if abs(scores['tree']-receipt['rmse'])>1e-12:raise ValueError('Independent tree score differs')
    result=dict(fold=fold,rows=len(labels),saved_tree_models_bitwise_equal=True,independent_rmse=scores,
        delta_vs_preceding_fixed_half=scores['fixed_half']-scores['preceding_fixed_half'],
        receipt_sha256=u.sha(receipt_path),manifest_sha256=u.sha(e.REPORT/'manifest.json'),
        preceding_evaluation_sha256=u.sha(old.REPORT/'evaluation.json'),verifier_sha256=u.sha(Path(__file__)),
        qualifying_labels_read=False,goal_complete=False,
        scope='One completed adaptive known-ProbeX fold. All five folds, full saved-model replay and independent full scoring required before any promotion. No qualifying inference or target claim.')
    u.write(e.REPORT/f'partial-fold-{fold}-verification.json',result);print(result,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fold',type=int,choices=range(5),required=True)
    verify(parser.parse_args().fold)
