#!/usr/bin/env python3
"""Separate exact consensus reconstruction, source rehash and known-only score."""
import datetime as dt
import math
from pathlib import Path
import numpy as np
import upstream_autorec_consensus as experiment

u=experiment.u

def verify():
    path=experiment.REPORT/'verification.json';proof=u.read(path)
    plan=u.read(experiment.REPORT/'plan.json')
    if (proof['state']!='complete_fixed_post_study_consensus' or plan['weights']!=[.5,.5] or
            proof['origins']!=experiment.ORIGINS or proof['qualifying_labels_read'] is not False):
        raise ValueError('Declared fixed known-only consensus required')
    raw=[np.fromfile(u.ROOT/'artifacts'/name/'fixed-half-tree-nn.f32',dtype='<f4').astype(np.float64) for name in proof['origins']]
    predicted=np.fromfile(experiment.RUN/'consensus.f32',dtype='<f4')
    expected=(raw[0]*.5+raw[1]*.5).astype('<f4')
    if not np.array_equal(predicted.view(np.uint32),expected.view(np.uint32)) or u.sha(experiment.RUN/'consensus.f32')!=proof['prediction_sha256']:
        raise ValueError('Stored consensus differs')
    labels=np.load(u.FRESH/'data/probex/ratings.npy',allow_pickle=False)
    if predicted.shape!=labels.shape:raise ValueError('Different complete known population')
    # Convert the scalar before comparison: numpy.float32 equality would round
    # the Python4.95 bound and treat its lower float32 neighbor as equal.
    score=math.sqrt(math.fsum((min(4.95,max(1.05,float(p)))-float(y))**2 for p,y in zip(predicted,labels))/len(labels))
    if abs(score-proof['rmse'])>1e-12:raise ValueError('Independent score differs')
    for relative,digest in proof['source_sha256'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Original source/model evidence differs')
    u.write(experiment.REPORT/'independent-verification.json',dict(state='complete_independent_consensus_and_score',
        verification_sha256=u.sha(path),verifier_sha256=u.sha(Path(__file__)),rows=len(labels),rmse=score,
        stored_consensus_bitwise_equal=True,source_files_rehashed=len(proof['source_sha256']),qualifying_labels_read=False,
        scope='Separate exact stored consensus and compensated known-only score, with all original source/model hashes rechecked. Reuses actual originating model replay evidence; no qualifying outcomes or new model inference.'))
    print('Independent consensus',score,flush=True)

if __name__=='__main__':verify()
