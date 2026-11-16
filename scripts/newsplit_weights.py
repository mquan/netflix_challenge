#!/usr/bin/env python3
"""Exact inverse inclusion probabilities for the rating-independent ProbeX split."""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load
from streaming_blend import digest


def inclusion_weights(d):
    q=d['probe'];users=q['user'];counts=np.bincount(users,minlength=d['users'])
    order=np.argsort(users,kind='stable');offset=np.r_[0,np.cumsum(counts)]
    weights=np.ones(len(q),dtype=np.float64)
    for u in np.flatnonzero(counts>1):
        rows=order[offset[u]:offset[u+1]]
        lo,hi=map(int,d['train_offsets'][u:u+2]);history=d['train_days'][lo:hi]
        day=q['day'][rows];pos=np.searchsorted(history,day)
        exists=pos<len(history);exists[exists]&=history[pos[exists]]==day[exists]
        candidates=rows[~exists]
        if len(candidates):weights[candidates]=1/(1-.9/len(candidates))
    return weights


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True);ap.add_argument('--rows',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args();d=load(a.data);weights=inclusion_weights(d)
    rows=np.fromfile(a.rows,dtype='<u4');retained=weights[rows]
    np.save(a.output,retained,allow_pickle=False)
    report=dict(method='Exact inverse probability of retention, 1/(1-0.9/candidate_count) for eligible queries; otherwise1',
                probabilities_use_rating_values=False,source_data_sha256=digest(a.data),rows_sha256=digest(a.rows),
                output_sha256=digest(a.output),retained_rows=len(rows),original_rows=len(weights),
                reconstructed_population_mass=float(retained.sum()),maximum=float(retained.max()),
                nonunit_rows=int((retained>1).sum()),metric='Weighted ProbeX estimates original sampling distribution; not exact full-Probe or Test RMSE')
    Path(a.output+'.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__': main()
