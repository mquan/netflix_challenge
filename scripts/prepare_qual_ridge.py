#!/usr/bin/env python3
"""Prepare full/inner-development ridge baselines without training trees."""
import argparse
import json
from pathlib import Path
import numpy as np
from fit_qual_blend import solve
from streaming_blend import digest,predict_ridge


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True)
    ap.add_argument('--qual',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    b=json.loads(Path(a.bank+'.inputs.json').read_text())
    q=json.loads(Path(a.qual+'.json').read_text())
    if q['state']!='complete' or q['qualifying_labels_read'] is not False:
        raise ValueError('unverified qualifying features')
    if q['bank_sha256']!=digest(a.bank+'.inputs.json'):
        raise ValueError('different bank')
    s=np.load(a.bank+'.stats.npz');stats=tuple(s[k] for k in ('gram','rhs','count'))
    full=solve(stats,np.ones(10,dtype=bool));inner=solve(stats,np.arange(10)%2==0)
    x=np.load(a.bank+'.x.npy',mmap_mode='r');qx=np.load(a.qual+'.x.npy',mmap_mode='r')
    c=np.load(b['config']['meta'])['context'];qc=np.load(a.qual+'.context.npy',mmap_mode='r')
    np.savez(root/'ridge-weights.npz',full=full,inner=inner)
    for name,raw,context,w in [('development-base.npy',x,c,full),
                             ('development-inner-base.npy',x,c,inner),
                             ('qualifying-base.npy',qx,qc,full)]:
        np.save(root/name,predict_ridge(raw[:,:b['columns']],context,w),allow_pickle=False)
    report=dict(state='baselines_ready',config=vars(a),bank_sha256=digest(a.bank+'.inputs.json'),
                stats_sha256=digest(a.bank+'.stats.npz'),qualifying_labels_read=False)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
