#!/usr/bin/env python3
"""Reweight exact bank statistics; share immutable feature files with hard links."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from gbdt_blend import load,blend_fold
from streaming_blend import digest,grouped_statistics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True);ap.add_argument('--qual',required=True)
    ap.add_argument('--weights',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--metric-label',default='Alternative ProbeX with exact sampling correction (not exact full original Probe)')
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    dest=str(root/'bank');qd=str(root/'qual')
    b=json.loads(Path(a.bank+'.inputs.json').read_text());q=json.loads(Path(a.qual+'.json').read_text())
    if q['bank_sha256']!=digest(a.bank+'.inputs.json') or q['qualifying_labels_read'] is not False:
        raise ValueError('unverified qualifying schema')
    if digest(b['config']['data'])!=b['data_sha256']:raise ValueError('development data changed')
    sw=np.load(a.weights,allow_pickle=False)
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);users=d['probe']['user'].copy();del d
    if sw.shape!=y.shape or not np.isfinite(sw).all() or np.any(sw<=0):raise ValueError('invalid weights')
    outer=blend_fold(users);inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    raw=np.load(a.bank+'.x.npy',mmap_mode='r');c=np.load(b['config']['meta'])['context']
    stats=grouped_statistics(raw[:,:b['columns']],c,y,outer*2+inner,sample_weights=sw)
    np.savez(dest+'.stats.npz',gram=stats[0],rhs=stats[1],count=stats[2])
    os.link(a.bank+'.x.npy',dest+'.x.npy')
    b.update(parent_bank_sha256=digest(a.bank+'.inputs.json'),sample_weights_sha256=digest(a.weights))
    b['config'].update(output=dest,sample_weights=a.weights,
                       metric_label=a.metric_label)
    Path(dest+'.inputs.json').write_text(json.dumps(b,indent=2)+'\n')
    for suffix in ('.x.npy','.context.npy'):os.link(a.qual+suffix,qd+suffix)
    q.update(bank_sha256=digest(dest+'.inputs.json'),parent_qual_manifest_sha256=digest(a.qual+'.json'),
             derivation='Identical qualifying features; only development training weights changed')
    Path(qd+'.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(dict(state='complete',bank=dest,qual=qd,weights_sha256=digest(a.weights))),flush=True)


if __name__=='__main__':main()
