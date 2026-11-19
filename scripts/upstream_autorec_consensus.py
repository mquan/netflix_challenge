#!/usr/bin/env python3
"""Fixed post-study consensus of two fully replayed own development ensembles."""
import datetime as dt
import math
from pathlib import Path
import numpy as np
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-autorec-consensus'
REPORT=u.ROOT/'experiments/upstream-autorec-consensus'
ORIGINS=['upstream-rbm50-tree-latent90-mps','upstream-rbm50-tree-autorec109-mps']

def run():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing consensus evidence preserved')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),origins=ORIGINS,weights=[.5,.5],
        script_sha256=u.sha(Path(__file__)),selection='Single post-study fixed equal mean of the two already verified own107/all90 and own109/all90 expanded-tree combinations. Both component scores were known before this declaration. No weight search or qualifying outcomes. Retain float32 rounding of each original mean, then average in float64 and store float32.',
        automatic_production=False,qualifying_labels_read=False))
    sources={};values=[];rows=None
    for name in ORIGINS:
        report=u.ROOT/'experiments'/name;cache=u.ROOT/'artifacts'/name
        verification=report/'verification.json';proof=u.read(verification)
        separate=report/'independent-mean-check.json';checked=u.read(separate)
        vector=cache/'fixed-half-tree-nn.f32'
        if (proof['state']!='complete_verified_fixed_combination' or proof['weights']!=[.5,.5] or
                proof['qualifying_labels_read'] is not False or not proof['original_saved_model_replays_verified'] or
                not proof['all_ten_original_fold_receipts_and_models_rehashed'] or
                checked['state']!='complete_separate_stored_mean_and_score_check' or
                checked['verification_sha256']!=u.sha(verification) or not checked['fixed_mean_bitwise_equal'] or
                abs(checked['rmse']-proof['rmse'])>1e-12 or u.sha(vector)!=proof['prediction_sha256']):
            raise ValueError('Actual complete own mean/replay evidence required')
        for relative,digest in proof['source_sha256'].items():
            if relative in sources and sources[relative]!=digest:raise ValueError('Conflicting source lineage')
            sources[relative]=digest
        for p in [verification,separate,vector]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
        if rows is None:rows=proof['rows']
        elif rows!=proof['rows']:raise ValueError('Different query population')
        a=np.fromfile(vector,dtype='<f4')
        if a.shape!=(rows,) or not np.isfinite(a).all():raise ValueError('Invalid original mean')
        values.append(a.astype(np.float64))
    sources[str(Path(__file__).relative_to(u.ROOT))]=u.sha(Path(__file__))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    label_path=u.FRESH/'data/probex/ratings.npy';labels=np.load(label_path,allow_pickle=False)
    groups_path=u.ROOT/'artifacts/upstream-rbm50-tree/groups.npy';groups=np.load(groups_path)
    for p in [label_path,groups_path]:
        if sources.get(str(p.relative_to(u.ROOT)))!=u.sha(p):raise ValueError('Canonical known identities differ')
    RUN.mkdir();prediction=((values[0]+values[1])/2).astype('<f4');prediction.tofile(RUN/'consensus.f32')
    def score(a,y):return math.sqrt(math.fsum((float(p)-float(t))**2 for p,t in zip(np.clip(a.astype(np.float64),1.05,4.95),y))/len(y))
    rmse=score(prediction,labels);best=score(values[0],labels)
    for relative,digest in sources.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Original source/model evidence changed')
    result=dict(state='complete_fixed_post_study_consensus',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        origins=ORIGINS,weights=[.5,.5],rows=rows,rmse=rmse,previous_best_rmse=best,improves_previous=rmse<best,
        delta_vs_previous_best=rmse-best,fold_rmse=[score(prediction[groups//2==f],labels[groups//2==f]) for f in range(5)],
        prediction_sha256=u.sha(RUN/'consensus.f32'),source_sha256=sources,qualifying_labels_read=False,goal_complete=False,
        scope='Fixed post-study equal consensus of independently verified own means. Each original float32 mean is retained before averaging. All original source/model bindings rehashed; actual saved-model replays are reused. No new inference replay, weight search, qualifying prediction or official Test claim.')
    u.write(REPORT/'verification.json',result);print({k:result[k] for k in ['rmse','delta_vs_previous_best','improves_previous']},flush=True)

if __name__=='__main__':run()
