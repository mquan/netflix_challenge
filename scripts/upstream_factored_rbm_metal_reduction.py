"""Stable ordered row additions for deterministic MPS RBM equations.

No atomics: one Metal thread owns each destination scalar and visits its
source rows in original order. CPU work sorts only integer addresses; all
learned floating-point values remain on MPS. Actual device admission is separate.
"""
import numpy as np

SOURCE=r'''
#include <metal_stdlib>
using namespace metal;
kernel void ordered_add(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    const device int* order [[buffer(2)]],
    const device int* offsets [[buffer(3)]],
    constant int& width [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  const uint row=tid/uint(width);
  const uint col=tid%uint(width);
  float total=dst[tid];
  for (int k=offsets[row];k<offsets[row+1];++k) {
    total=total+src[uint(order[k])*uint(width)+col];
  }
  dst[tid]=total;
}
'''
_LIBRARY=None


def addresses(indices,rows):
    ids=np.asarray(indices)
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or type(rows) is not int or rows<1
            or len(ids)>=2**31 or rows>=2**31 or np.any(ids<0) or np.any(ids>=rows)):
        raise ValueError('Complete bounded integer row addresses required')
    order=np.argsort(ids,kind='stable').astype(np.int32)
    counts=np.bincount(ids.astype(np.int64),minlength=rows)
    offsets=np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)].astype(np.int32)
    return order,offsets


def add_rows_(destination,indices,values):
    import torch
    if (destination.ndim<1 or indices.ndim!=1 or indices.dtype!=torch.int64
            or values.shape!=(len(indices),*destination.shape[1:])
            or destination.dtype!=torch.float32 or values.dtype!=torch.float32
            or destination.device!=values.device or destination.device!=indices.device
            or not destination.is_contiguous() or destination.numel()>=2**32):
        raise ValueError('Exact same-device contiguous float32 row-add shapes required')
    if destination.device.type=='cpu':
        destination.index_add_(0,indices,values);return destination
    if destination.device.type!='mps':raise ValueError('Only declared CPU test or actual MPS device permitted')
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError('Strict deterministic algorithms required')
    if not len(indices):return destination
    order,offsets=addresses(indices.detach().cpu().numpy(),len(destination))
    width=int(destination.numel()//len(destination))
    if width<1:return destination
    global _LIBRARY
    if _LIBRARY is None:_LIBRARY=torch.mps.compile_shader(SOURCE)
    permutation=torch.tensor(order,dtype=torch.int32,device='mps')
    bounds=torch.tensor(offsets,dtype=torch.int32,device='mps')
    columns=torch.tensor([width],dtype=torch.int32,device='mps')
    _LIBRARY.ordered_add(destination,values.contiguous(),permutation,bounds,columns,
                         threads=[destination.numel(),1,1],group_size=[256,1,1])
    return destination
