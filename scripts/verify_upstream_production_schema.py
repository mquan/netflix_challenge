#!/usr/bin/env python3
"""Verify production transformations against the frozen development cache."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
from pathlib import Path
import numpy as np
import upstream_repro as u
import upstream_expansion as e
import upstream_snapshot_blend as b
import upstream_remote_batch as remote
import upstream_voting_blend as voting
import upstream_production_bank as production


def main():
    voting.configure();voting.n.verify_prepared()
    labels,users,fresh=e.collect_expansion();frozen=u.read(b.REPORT/'manifest.json')
    current=b.freeze(labels,users,fresh)
    if any(current[key]!=frozen[key] for key in current):raise ValueError('Original fresh snapshot bank changed')
    arrays,receipt,_=remote.receive()
    rows,_=production.recipe()
    first=[fresh[r['name']] for r in frozen['entries'] if r['duplicate_of'] is None]
    first.append(np.load(u.FRESH/'preds_new/rbmx2-150__knnf.probex.npy',mmap_mode='r'))
    raw=first+[arrays[name] for name in u.read(remote.parent.REPORT/'manifest.json')['remote_columns']]
    raw.extend(arrays[name] for name in u.read(remote.REPORT/'manifest.json')['remote_columns'])
    raw.extend(np.load(u.FRESH/'preds_new'/(name+'.probex.npy'),mmap_mode='r') for name in voting.NEW)
    if len(raw)!=85:raise ValueError('Golden source raw bank differs')
    for name,digest in frozen['context_inputs'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Golden context metadata changed')
    context=b.context_features(*[np.load(u.ROOT/name,mmap_mode='r') for name in frozen['context_inputs']])
    features=np.load(voting.RUN/'features.npy',mmap_mode='r');design=np.load(voting.RUN/'design.npy',mmap_mode='r')
    vf=[np.load(u.FRESH/path,mmap_mode='r') for path in u.read(voting.REPORT/'manifest.json')['published_features']]
    # Spread across the whole native row order; include scalar/chunk-edge cases.
    indices=np.unique(np.r_[np.arange(8193),np.linspace(0,len(labels)-1,12000,dtype=int),len(labels)-1])
    counts=[]
    for chunk in [1,8192]:
        sample=indices[-1:] if chunk==1 else indices
        for lo in range(0,len(sample),chunk):
            ix=sample[lo:lo+chunk]
            z,f=production.blocks(np.column_stack([a[ix] for a in raw]),np.array([r['clip'] for r in rows]),
                context[ix],np.column_stack([a[ix] for a in vf]))
            if not np.array_equal(z,design[ix]):raise ValueError('Production ridge transformation differs from frozen development')
            if not np.array_equal(f.view(np.uint32),features[ix].view(np.uint32)):
                raise ValueError('Production nonlinear transformation differs from frozen development')
        counts.append(dict(chunk_size=chunk,rows_checked=len(sample),bitwise_equal=True))
    report=dict(state='complete',scope='Transformation schema only: same frozen fresh development predictions, not a claim that retrained model outputs equal remote outputs.',
        production_code_sha256=u.sha(Path(production.__file__)),development_manifest_sha256=u.sha(voting.REPORT/'manifest.json'),
        verifier_sha256=u.sha(Path(__file__)),prediction_columns=85,design_columns=161,nonlinear_features=221,
        checks=counts,qualifying_labels_read=False)
    u.write(production.REPORT/'schema-verification.json',report);print(report)


if __name__=='__main__':main()
