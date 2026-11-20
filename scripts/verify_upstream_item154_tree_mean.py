#!/usr/bin/env python3
"""Separate exact known mean reconstruction and compensated canonical scoring."""
from pathlib import Path
import math
import numpy as np
import upstream_item154_tree_item154_mps as candidate
u=candidate.u

def verify():
    report=candidate.REPORT;proof=u.read(report/'verification.json')
    if (report/'independent-mean-check.json').exists():raise ValueError('Existing independent mean proof preserved')
    if proof['state']!='complete_verified_fixed_combination' or proof['weights']!=[.5,.5] or proof['qualifying_labels_read'] is not False:raise ValueError('Original known-only fixed combination required')
    for name,digest in proof['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original source or model changed')
    families=[];identity=None
    for family,name in candidate.ORIGINS:
        root=u.ROOT/'artifacts'/name;rr=u.ROOT/'experiments'/name
        values=np.fromfile(root/(family+'.f32'),dtype='<f4');arrays={key:np.load(root/(key+'.npy')) for key in ['labels','users','groups']}
        if identity is None:identity=arrays
        elif any(not np.array_equal(arrays[key],identity[key]) for key in arrays):raise ValueError('Original family row/fold identities differ')
        replay=u.read(rr/(family+'-replay.json'))
        if [r['fold'] for r in replay['records']]!=list(range(5)) or not all(r['bitwise_equal'] for r in replay['records']):raise ValueError('All original saved-model replays required')
        families.append(values)
    labels=np.load(u.FRESH/'data/probex/ratings.npy')
    if not np.array_equal(labels,identity['labels']) or not np.array_equal(np.load(u.FRESH/'data/probex/user_idxs.npy'),identity['users']):raise ValueError('Complete canonical known identities differ')
    path=candidate.RUN/'fixed-half-tree-nn.f32';stored=np.fromfile(path,dtype='<f4')
    expected=((families[0].astype(np.float64)+families[1].astype(np.float64))/2).astype('<f4')
    if stored.shape!=(1128164,) or not np.isfinite(stored).all() or not np.array_equal(stored.view(np.uint32),expected.view(np.uint32)) or u.sha(path)!=proof['prediction_sha256']:raise ValueError('Every original stored mean value must match')
    score=math.sqrt(math.fsum((min(4.95,max(1.05,float(p)))-float(y))**2 for p,y in zip(stored,labels))/len(labels))
    if abs(score-proof['rmse'])>1e-12:raise ValueError('Independent complete known score differs')
    u.write(report/'independent-mean-check.json',dict(state='complete_separate_stored_mean_and_score_check',observed=candidate.dt.datetime.now(candidate.dt.timezone.utc).isoformat(),
        rows=len(labels),rmse=score,fixed_mean_bitwise_equal=True,verification_sha256=u.sha(report/'verification.json'),verifier_sha256=u.sha(Path(__file__)),
        source_files_rehashed=len(proof['source_sha256']),qualifying_labels_read=False,
        scope='Separate exact stored half mean, complete canonical known labels/users, original family fold identities, actual saved-model replay receipts and all source/model hashes. Complete compensated known scoring; no new model fitting or qualifying score claimed.'))
    print(dict(rmse=score,rows=len(labels),exact_mean=True),flush=True)

if __name__=='__main__':verify()
