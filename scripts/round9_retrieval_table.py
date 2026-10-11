#!/usr/bin/env python3
"""Build movie cosine similarities from a frozen, training-only joint checkpoint.

Selection must use the P2-trained checkpoint; refit uses the full-train checkpoint.
No probe ratings enter this exporter. Model provenance is recorded in the manifest.
"""
import argparse
import json
import pathlib
import numpy as np
from gbdt_blend import load
from round9_latent import read_model, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--model',required=True)
    p.add_argument('--output',required=True);p.add_argument('--dimensions',type=int,default=32)
    a=p.parse_args(); dest=pathlib.Path(a.output)
    if dest.exists(): raise FileExistsError(dest)
    d=load(a.data);m,info=read_model(a.model,d['users'],d['items'])
    q=np.array(m['item_factors'],dtype=np.float64); q-=q.mean(0)
    _,v=np.linalg.eigh(q.T@q)
    x=(q@v[:,-a.dimensions:]).astype('f4'); x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    table=np.lib.format.open_memmap(dest,mode='w+',dtype='f4',shape=(len(q),len(q)))
    for start in range(0,len(q),512): table[start:start+512]=x[start:start+512]@x.T
    table.flush(); del table
    manifest=dict(config=vars(a),model=info,inputs={path:digest(path) for path in (a.data,a.model)},
                  output_sha256=digest(dest),source_sha256=digest(__file__))
    pathlib.Path(str(dest)+'.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(dest),dimensions=a.dimensions)),flush=True)


if __name__=='__main__':main()
