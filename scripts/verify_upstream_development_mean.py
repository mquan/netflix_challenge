#!/usr/bin/env python3
"""Separately rehash source/model evidence and check stored development means."""
import argparse
import datetime as dt
import math
from pathlib import Path
import time
import numpy as np
import upstream_repro as u

def verify(name):
    report=u.ROOT/'experiments'/name;run=u.ROOT/'artifacts'/name
    path=report/'verification.json';proof=u.read(path);bindings=dict(proof['source_sha256'])
    if (proof['state']!='complete_verified_fixed_combination' or proof['weights']!=[.5,.5] or
            proof['qualifying_labels_read'] is not False or not proof['original_saved_model_replays_verified']):
        raise ValueError('Complete actual replayed known-only fixed mean required')
    bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    vectors=[];identities=[]
    for family,origin in proof['origins']:
        cache=u.ROOT/'artifacts'/origin;origin_report=u.ROOT/'experiments'/origin
        replay_path=origin_report/(family+'-replay.json');replay=u.read(replay_path)
        if ([r['fold'] for r in replay['records']]!=list(range(5)) or
                not all(r['bitwise_equal'] for r in replay['records']) or
                bindings[str(replay_path.relative_to(u.ROOT))]!=u.sha(replay_path)):
            raise ValueError('Original complete actual saved-model replay differs')
        p=cache/(family+'.f32');vectors.append(np.fromfile(p,dtype='<f4').astype(np.float64))
        identities.append({k:np.load(cache/(k+'.npy'),allow_pickle=False) for k in ['labels','users','groups']})
        if bindings[str(p.relative_to(u.ROOT))]!=u.sha(p):raise ValueError('Actual family vector differs')
    if len(vectors)!=2 or any(not np.array_equal(identities[0][k],identities[1][k]) for k in identities[0]):
        raise ValueError('Family row identities differ')
    labels=np.load(u.FRESH/'data/probex/ratings.npy',allow_pickle=False)
    if not np.array_equal(labels,identities[0]['labels']):raise ValueError('Canonical known ratings differ')
    stored_path=run/'fixed-half-tree-nn.f32';stored=np.fromfile(stored_path,dtype='<f4')
    expected=np.multiply(np.add(vectors[0],vectors[1]),.5).astype('<f4')
    if stored.shape!=labels.shape or not np.array_equal(stored.view(np.uint32),expected.view(np.uint32)):
        raise ValueError('Stored float32 fixed mean differs')
    if u.sha(stored_path)!=proof['prediction_sha256']:raise ValueError('Stored mean binding differs')
    clipped=np.clip(stored.astype(np.float64),1.05,4.95)
    score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(clipped,labels))/len(labels))
    if abs(score-proof['rmse'])>1e-12:raise ValueError('Independent compensated score differs')
    bindings[str(stored_path.relative_to(u.ROOT))]=u.sha(stored_path)
    for relative,digest in bindings.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Original bound source/model evidence changed')
    result=dict(state='complete_separate_stored_mean_and_score_check',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        rows=len(labels),rmse=score,fixed_mean_bitwise_equal=True,verification_sha256=u.sha(path),
        verifier_sha256=u.sha(Path(__file__)),source_files_rehashed=len(bindings),qualifying_labels_read=False,
        scope='Separate exact stored-mean, canonical known ratings, original family identities, actual replay receipts and all source/model hashes checked. Compensated full-known scoring agrees. This checker does not rerun model inference or claim a qualifying score.')
    out=report/'independent-mean-check.json'
    if out.exists():raise ValueError('Existing separate proof preserved')
    u.write(out,result);print(name,score,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--watch',action='store_true');args=p.parse_args()
    if args.watch:
        while not (u.ROOT/'experiments'/args.name/'verification.json').exists():time.sleep(15)
    verify(args.name)
