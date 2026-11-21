#!/usr/bin/env python3
"""Complete dynamic-width original percentile/ordered-statistic scaler with bounded scratch."""
import numpy as np


def fit_scaler(features,eligible,block_rows=8192):
    if (features.ndim!=2 or features.dtype!=np.float32 or not features.flags.c_contiguous or
        not 618<=features.shape[1]<=813 or eligible.shape!=(len(features),) or eligible.dtype!=np.bool_ or
        not eligible.any() or not isinstance(block_rows,int) or block_rows<=0):
        raise ValueError('Complete dynamic-width C-order float32 input and original eligible mask required')
    lo=np.empty(features.shape[1]);hi=np.empty(features.shape[1])
    for column in range(features.shape[1]):lo[column],hi[column]=np.percentile(features[eligible,column],[1,99])
    def total(mean=None):
        carry=np.zeros(features.shape[1],dtype=np.float64)
        for start in range(0,len(features),block_rows):
            stop=min(start+block_rows,len(features));selected=features[start:stop][eligible[start:stop]]
            if not len(selected):continue
            values=np.clip(selected,lo,hi)
            if values.dtype!=np.float64 or not values.flags.c_contiguous:raise ValueError('Original ordered C-array statistics required')
            if mean is not None:
                np.subtract(values,mean,out=values);np.multiply(values,values,out=values)
            np.add(carry,values[0],out=values[0]);np.add.accumulate(values,axis=0,out=values);carry=values[-1].copy()
        return carry
    count=int(eligible.sum());mean=total()/count
    return dict(lo=lo,hi=hi,mean=mean,std=np.maximum(np.sqrt(total(mean)/count),1e-6))
