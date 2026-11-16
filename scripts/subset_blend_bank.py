#!/usr/bin/env python3
"""Remove a trailing base from an existing bank and its exact ridge statistics."""
import argparse
import json
from pathlib import Path
import numpy as np
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    src=json.loads(Path(a.source+'.inputs.json').read_text())
    if len(src['config']['extra'])!=1: raise ValueError('expected one trailing extra base')
    if Path(a.output+'.inputs.json').exists(): raise ValueError('output exists')
    x=np.load(a.source+'.x.npy',mmap_mode='r');n=src['columns'];drop=n-1
    out=np.lib.format.open_memmap(a.output+'.x.npy',mode='w+',dtype='<f4',
                                shape=(len(x),x.shape[1]-1),fortran_order=True)
    keep=np.delete(np.arange(x.shape[1]),drop)
    for j,k in enumerate(keep): out[:,j]=x[:,k]
    out.flush()
    stat=np.load(a.source+'.stats.npz');width=n+1
    nc=stat['rhs'].shape[1]//width
    ix=np.concatenate([np.arange(c*width,(c+1)*width-1) for c in range(nc)])
    # Design block is [intercept, base0,...,baseN]; last base is removed.
    np.savez(a.output+'.stats.npz',gram=stat['gram'][:,ix][:,:,ix],
             rhs=stat['rhs'][:,ix],count=stat['count'])
    removed=src['names'].pop(drop)
    del src['inputs'][removed]
    src['columns']-=1;src['config'].update(extra=[],output=a.output,ridge_only=True)
    src.update(derived_from=a.source,parent_manifest_sha256=digest(a.source+'.inputs.json'),
               derivation='Exact column/statistics submatrix; removed trailing extra base',removed=removed)
    Path(a.output+'.inputs.json').write_text(json.dumps(src,indent=2)+'\n')
    print(json.dumps(dict(state='complete',output=a.output,columns=src['columns'])))


if __name__=='__main__': main()
