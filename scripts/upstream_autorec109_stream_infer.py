#!/usr/bin/env python3
"""Use the original full NN inference arithmetic with exact streamed chunks."""
import numpy as np
import upstream_production as core

def infer(family,plan,root):
    if family!='nn':raise ValueError('This adapter only fits the new own109 NN')
    n,u,bank=core.n,core.u,core.bank
    w=np.load(core.RUN/'ridge.npy')
    if u.sha(core.RUN/'ridge.npy')!=plan['ridge_sha256']:raise ValueError('Final ridge changed')
    models=[None if not count else core.replay.neural_model(root/f'nn-{seed}.pt') for seed,count in zip(plan['seeds'],plan['iterations']['nn'])]
    with np.load(root/'scaler.npz') as archive:scaler={key:archive[key] for key in archive.files}
    prediction=np.empty(2817131,dtype='<f4')
    for lo in range(0,len(prediction),core.CHUNK):
        hi=min(lo+core.CHUNK,len(prediction));z,f=bank.qual_chunk(lo,hi)
        base=np.clip(z@w,1.05,4.95);features=np.column_stack([f,base]).astype(np.float32)
        features,_=n.scaled(features,None,scaler);correction=np.zeros(hi-lo,dtype=np.float64)
        for model in models:
            if model is not None:correction+=n.nn_predict(model,features)/len(models)
        prediction[lo:hi]=np.clip(base+correction,1.05,4.95).astype('<f4')
    return prediction
