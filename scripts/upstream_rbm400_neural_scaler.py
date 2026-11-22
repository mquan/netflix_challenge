"""Guarded exact original scaler arithmetic for all four declared neural widths.

No files, model fitting, data admission or device operations occur here. Full
eligible percentiles and row-ordered float64 sums are retained at every width.
"""
import numpy as np

WIDTHS=(713,714,863,864)


def validate(features,eligible,guard,block_rows):
    if (features.ndim!=2 or features.dtype!=np.float32 or not features.flags.c_contiguous
            or features.shape[1] not in WIDTHS or not len(features)
            or eligible.shape!=(len(features),) or eligible.dtype!=np.bool_ or not eligible.any()
            or not callable(guard) or type(block_rows) is not int or block_rows<1):
        raise ValueError('Complete declared float32 width, eligible mask and external guard required')
    for start in range(0,len(features),block_rows):
        guard()
        if not np.isfinite(features[start:start+block_rows]).all():raise ValueError('Every original raw feature must be finite')


def fit_scaler(features,eligible,*,guard,block_rows=8192):
    validate(features,eligible,guard,block_rows)
    lo=np.empty(features.shape[1]);hi=np.empty(features.shape[1])
    for column in range(features.shape[1]):
        guard();lo[column],hi[column]=np.percentile(features[eligible,column],[1,99])
    def total(mean=None):
        carry=np.zeros(features.shape[1],dtype=np.float64)
        for start in range(0,len(features),block_rows):
            guard();stop=min(start+block_rows,len(features));selected=features[start:stop][eligible[start:stop]]
            if not len(selected):continue
            values=np.clip(selected,lo,hi)
            if values.dtype!=np.float64 or not values.flags.c_contiguous:raise ValueError('Original ordered C-array statistics required')
            if mean is not None:
                np.subtract(values,mean,out=values);np.multiply(values,values,out=values)
            np.add(carry,values[0],out=values[0]);np.add.accumulate(values,axis=0,out=values);carry=values[-1].copy()
        return carry
    count=int(eligible.sum());mean=total()/count
    result=dict(lo=lo,hi=hi,mean=mean,std=np.maximum(np.sqrt(total(mean)/count),1e-6))
    guard()
    if any(not np.isfinite(value).all() for value in result.values()):raise ValueError('Finite full original scaler required')
    return result


def apply_scaler(features,scaler,*,guard,block_rows=8192,output=None):
    if not callable(guard) or type(block_rows) is not int or block_rows<1:
        raise ValueError('External guard and positive row block required')
    if (features.ndim!=2 or features.dtype!=np.float32 or not features.flags.c_contiguous
            or features.shape[1] not in WIDTHS or set(scaler)!={'lo','hi','mean','std'}
            or any(v.shape!=(features.shape[1],) or v.dtype!=np.float64 or not np.isfinite(v).all() for v in scaler.values())
            or np.any(scaler['std']<1e-6) or np.any(scaler['lo']>scaler['hi'])):
        raise ValueError('Exact complete raw matrix and finite float64 scaler required')
    out=np.empty(features.shape,dtype=np.float32) if output is None else output
    if out.shape!=features.shape or out.dtype!=np.float32 or not out.flags.c_contiguous:
        raise ValueError('Exact complete contiguous float32 output required')
    if np.shares_memory(features,out) and (features.ctypes.data!=out.ctypes.data or features.strides!=out.strides):
        raise ValueError('Transform may alias only the identical complete matrix')
    for start in range(0,len(features),block_rows):
        guard();stop=min(start+block_rows,len(features))
        if not np.isfinite(features[start:stop]).all():raise ValueError('Finite complete raw features required')
        out[start:stop]=np.asarray((np.clip(features[start:stop],scaler['lo'],scaler['hi'])-scaler['mean'])/
                                   scaler['std'],dtype=np.float32,order='C')
    guard();return out
