#!/usr/bin/env python3
"""Prepare label-free query metadata/context, bound to the exact dataset."""
import argparse
from pathlib import Path
import numpy as np
from gbdt_blend import load, meta_features
from moonshot_nn_blend import contexts
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    p=Path(a.output)
    if p.exists(): ap.error('output exists')
    p.parent.mkdir(parents=True,exist_ok=True)
    d=load(a.data)
    meta=np.column_stack(list(meta_features(d).values())).astype(np.float32)
    context=contexts(d)
    np.savez(p,meta=meta,context=context,data_sha256=digest(a.data))
    print('Prepared metadata',meta.shape,'context',context.shape,flush=True)


if __name__=='__main__': main()
