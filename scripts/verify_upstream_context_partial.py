#!/usr/bin/env python3
"""Replay one completed context fold; partial development evidence only."""
import argparse
import math
from pathlib import Path
import numpy as np
import upstream_context_blend as c


def independent_rmse(value, labels):
    value=np.asarray(value,dtype=np.float64);labels=np.asarray(labels,dtype=np.float64)
    if value.shape!=labels.shape or value.ndim!=1 or not len(labels) or not np.isfinite(value).all() or not np.isfinite(labels).all():
        raise ValueError('Malformed partial comparison prediction')
    return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(value,1.05,4.95),labels))/len(labels))


def verify(fold):
    c.configure();c.n.verify_prepared()
    receipt_path=c.REPORT/'nn'/(str(fold)+'.json')
    receipt=c.u.read(receipt_path);root=c.RUN/'nn'/str(fold)
    if receipt['manifest_sha256']!=c.u.sha(c.REPORT/'manifest.json'):
        raise ValueError('Completed fold recipe changed')
    for name,digest in receipt['outputs'].items():
        if c.u.sha(root/name)!=digest:raise ValueError('Completed fold artifact changed')
    groups=np.load(c.RUN/'groups.npy');held=groups//2==fold
    z=np.load(c.RUN/'design.npy',mmap_mode='r')
    with np.load(root/'ridge.npz') as saved:base=np.clip(z@saved['full'],1.05,4.95)
    features=np.load(c.RUN/'features.npy',mmap_mode='r')
    inputs=np.column_stack([features[held],base[held]]).astype(np.float32)
    with np.load(root/'scalers.npz') as saved:
        scaler={key:saved['full_'+key] for key in ['lo','hi','mean','std']}
    inputs,_=c.n.scaled(inputs,None,scaler=scaler)
    correction=np.zeros(int(held.sum()))
    for seed,epochs in zip(c.CONFIG['seeds'],receipt['chosen_iterations']):
        if epochs:
            model=c.neural_model(root/f'nn-{seed}.pt')
            correction+=c.n.nn_predict(model,inputs)/len(c.CONFIG['seeds'])
    actual=np.clip(base[held]+correction,1.05,4.95).astype('<f4')
    expected=np.fromfile(root/'prediction.f32',dtype='<f4')
    if actual.shape!=(receipt['rows'],) or not np.array_equal(actual,expected):
        raise ValueError('Completed fold saved-model replay differs')
    parent_nn=c.parent.RUN/'nn'/str(fold)/'prediction.f32'
    parent_receipt=c.u.read(c.parent.REPORT/'nn'/(str(fold)+'.json'))
    if c.u.sha(parent_nn)!=parent_receipt['outputs']['prediction.f32']:
        raise ValueError('Parent neural prediction changed')
    parent_evaluation=c.u.read(c.parent.REPORT/'evaluation.json')
    for arm in ['tree','fixed-half-tree-nn']:
        digest=next(r['prediction_sha256'] for r in parent_evaluation['records'] if r['arm']==arm)
        if c.u.sha(c.parent.RUN/(arm+'.f32'))!=digest:raise ValueError('Parent comparison changed')
    labels=np.load(c.RUN/'labels.npy')[held]
    tree=np.fromfile(c.RUN/'tree.f32',dtype='<f4')[held].astype(np.float64)
    values=dict(nn=actual,parent_nn=np.fromfile(parent_nn,dtype='<f4'),
        fixed_half=((tree+actual.astype(np.float64))/2).astype('<f4'),
        parent_fixed_half=np.fromfile(c.parent.RUN/'fixed-half-tree-nn.f32',dtype='<f4')[held])
    scores={name:independent_rmse(value,labels) for name,value in values.items()}
    if abs(scores['nn']-receipt['rmse'])>1e-12:raise ValueError('Independent fold scorer differs')
    result=dict(fold=fold,rows=len(labels),manifest_sha256=c.u.sha(c.REPORT/'manifest.json'),
        receipt_sha256=c.u.sha(receipt_path),verifier_sha256=c.u.sha(Path(__file__)),
        saved_models_bitwise_equal=True,independent_rmse=scores,
        scope='One completed adaptive ProbeX fold only. Remaining folds and final independent replay/score required. No qualifying prediction or goal-success claim.')
    c.u.write(c.REPORT/f'partial-fold-{fold}-verification.json',result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fold',type=int,choices=range(5),required=True)
    verify(parser.parse_args().fold)
