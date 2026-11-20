#!/usr/bin/env python3
"""Prove every split3 holdout target is known data and excludes all qualifying keys."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import hashlib
import numpy as np
import upstream_repro as u
import upstream_split3_pilot as source

REPORT=u.ROOT/'experiments/upstream-split3-holdout-origin-audit'

def main():
    if REPORT.exists():raise ValueError('Existing complete origin audit preserved')
    REPORT.mkdir()
    origin=u.ROOT/'artifacts/upstream-split3-next3/source/data/probe_3'
    query=source.arrays(origin,labels=True);known=source.arrays(u.FRESH/'data/probex',labels=True)
    if len(query[0])!=1760494 or not np.isin(query[3],[1,2,3,4,5]).all():raise ValueError('Original complete known holdout domain differs')
    mapping=source.projection(query,known)
    keep=np.ones(len(query[0]),dtype=bool);keep[mapping]=False
    extras=[a[keep] for a in query];extra_keys=source.keys(*extras[:3]);order=np.argsort(extra_keys)
    sorted_keys=extra_keys[order];targets=extras[3][order]
    if len(extras[0])!=632330 or np.any(sorted_keys[1:]==sorted_keys[:-1]):raise ValueError('Exact complete additional-known population differs')
    # Ignore dates for the stronger qualifying exclusion check; outcomes are never loaded.
    qualifier=source.arrays(u.FRESH/'data/qual',labels=False)
    hold_pairs=np.sort(query[0].astype(np.int64)*17770+query[1].astype(np.int64))
    qual_pairs=qualifier[0].astype(np.int64)*17770+qualifier[1].astype(np.int64)
    indices=np.searchsorted(hold_pairs,qual_pairs);inside=indices<len(hold_pairs)
    if np.any(hold_pairs[indices[inside]]==qual_pairs[inside]):raise ValueError('Qualifying identity appears in proposed known training labels')
    canonical=source.arrays(u.FRESH/'data/trainx',labels=True);counts=np.zeros(len(sorted_keys),dtype=np.uint8)
    u.write(REPORT/'plan.json',dict(holdout_rows=1760494,canonical_benchmark_rows=1128164,additional_known_rows=632330,
        rule='Recompute every canonical benchmark projection and label. Scan all canonical trainx rows for every additional holdout identity/date/label; require exactly one match. Exclude all qualifying user/item keys without loading any qualifying ratings. No fitting or prediction-score selection.'))
    for lo in range(0,len(canonical[0]),524288):
        hi=min(lo+524288,len(canonical[0]));keys=source.keys(*[a[lo:hi] for a in canonical[:3]])
        indices=np.searchsorted(sorted_keys,keys);inside=indices<len(sorted_keys)
        matched=np.zeros(len(keys),dtype=bool);matched[inside]=sorted_keys[indices[inside]]==keys[inside]
        hits=indices[matched]
        if not np.array_equal(canonical[3][lo:hi][matched],targets[hits]):raise ValueError('Additional known target differs from canonical original known data')
        np.add.at(counts,hits,1)
        u.write(REPORT/'status.json',dict(state='scanning_complete_canonical_known_origin',canonical_rows_scanned=hi,
            canonical_rows_total=len(canonical[0]),additional_rows_matched=int(np.count_nonzero(counts))))
    if not np.all(counts==1):raise ValueError('Every additional label must have exactly one canonical known origin')
    paths=[Path(__file__),Path(source.__file__)]
    for directory,names in [(origin,['user_idxs','item_idxs','dates','ratings']),
        (u.FRESH/'data/probex',['user_idxs','item_idxs','dates','ratings']),
        (u.FRESH/'data/trainx',['user_idxs','item_idxs','dates','ratings']),
        (u.FRESH/'data/qual',['user_idxs','item_idxs','dates'])]:paths.extend(directory/(name+'.npy') for name in names)
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
    u.write(REPORT/'verification.json',dict(state='complete_all_split3_holdout_labels_have_disjoint_canonical_known_origin',
        holdout_rows=1760494,canonical_benchmark_rows=1128164,additional_known_rows=632330,
        complete_benchmark_projection_and_labels_recomputed=True,all_additional_keys_dates_targets_match_canonical_trainx_exactly_once=True,
        all_qualifying_user_item_keys_disjoint=True,qualifying_ratings_file_opened=False,qualifying_labels_read=False,
        source_sha256=bindings,models_fitted=0,
        scope='Origin/eligibility audit only, no accuracy claim. Future use must exclude canonical held users from all augmented meta-training labels; compare only matching evaluation populations.'))
    print('All1760494 split3 labels are known-only:1128164 benchmark +632330 additional; all qualifying keys excluded',flush=True)

if __name__=='__main__':main()
