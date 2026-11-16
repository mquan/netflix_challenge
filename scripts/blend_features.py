"""Prediction-mean/difference encoding shared by neural and tree blenders."""
import numpy as np


def nn_features(raw, columns, output=None):
    shape=(len(raw),raw.shape[1]+1)
    x=(np.lib.format.open_memmap(output,mode='w+',dtype='<f4',shape=shape,fortran_order=True)
       if output else np.empty(shape,dtype=np.float32,order='F'))
    p=raw[:,:columns];mean=p.mean(1)
    x[:,0]=mean-3.6
    for j in range(columns):x[:,j+1]=p[:,j]-mean
    x[:,columns+1:]=raw[:,columns:]
    if output:x.flush()
    return x
