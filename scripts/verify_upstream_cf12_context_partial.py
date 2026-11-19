#!/usr/bin/env python3
"""Replay one completed CF12 context fold and compare the preceding context mean."""
import argparse
from pathlib import Path
import numpy as np
import upstream_cf12_context as experiment
import verify_upstream_context_partial as core


def verify(fold):
    # The original verifier operates on a module interface. Bind that interface
    # only in this interpreter, retaining the existing replay/scoring algorithm.
    core.c=experiment
    core.verify(fold)
    u=experiment.u
    path=experiment.REPORT/f'partial-fold-{fold}-verification.json'
    result=u.read(path)
    old_report=experiment.CONTEXT_REPORT
    old_run=experiment.context.RUN
    old_evaluation=u.read(old_report/'evaluation.json')
    old_row=next(r for r in old_evaluation['records'] if r['arm']=='fixed-half-tree-nn')
    previous=old_run/'fixed-half-tree-nn.f32'
    if u.sha(previous)!=old_row['prediction_sha256']:raise ValueError('Preceding context mean changed')
    for name in ['users','groups','labels']:
        if not np.array_equal(np.load(old_run/(name+'.npy')),np.load(experiment.RUN/(name+'.npy'))):
            raise ValueError('Preceding context comparison identities differ')
    groups=np.load(experiment.RUN/'groups.npy');held=groups//2==fold
    labels=np.load(experiment.RUN/'labels.npy')[held]
    old=np.fromfile(previous,dtype='<f4')[held]
    score=core.independent_rmse(old,labels)
    result['independent_rmse']['preceding_context_fixed_half']=score
    result['delta_vs_preceding_context']=result['independent_rmse']['fixed_half']-score
    result['preceding_context_evaluation_sha256']=u.sha(old_report/'evaluation.json')
    result['preceding_context_prediction_sha256']=u.sha(previous)
    result['adapter_sha256']=u.sha(Path(__file__))
    u.write(path,result)
    print(result,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,choices=range(5),required=True)
    verify(parser.parse_args().fold)
