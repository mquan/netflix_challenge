#!/usr/bin/env python3
"""Close original known-input and complete native training-disjointness gates."""
from pathlib import Path
import numpy as np
import upstream_repro as u
import upstream_split3_pilot as original

REPORT=u.ROOT/'experiments/upstream-split3-full-population-v2'

def main():
    REPORT.mkdir(exist_ok=False)
    origin=u.ROOT/'experiments/upstream-split3-holdout-origin-audit/verification.json';audit=u.read(origin)
    if (audit['state']!='complete_all_split3_holdout_labels_have_disjoint_canonical_known_origin' or
        audit['additional_known_rows']!=632330 or not audit['all_additional_keys_dates_targets_match_canonical_trainx_exactly_once'] or
        not audit['all_qualifying_user_item_keys_disjoint'] or audit['qualifying_ratings_file_opened']):raise ValueError('Complete known-only origin proof required')
    fixed=u.ROOT/'experiments/upstream-remote15h/plan.json';plan=u.read(fixed)
    bank_path=u.ROOT/'experiments/upstream-item154-stream-bank/manifest.json';bank=u.read(bank_path)
    bank_proof=u.read(bank_path.parent/'column-verification.json')
    if bank_proof['manifest_sha256']!=u.sha(bank_path):raise ValueError('Original paired-bank input binding differs')
    for key,digest in audit['source_sha256'].items():
        p=u.ROOT/key
        if u.sha(p)!=digest:raise ValueError('Origin audit source changed')
        if p.is_relative_to(u.FRESH/'data'):
            relative=str(p.relative_to(u.FRESH/'data'))
            expected=bank['source_sha256'].get(key) if relative.startswith('trainx/') else plan['data_sha256'].get(relative)
            if expected!=digest:raise ValueError('Canonical data must match original frozen known-data recipe')
    tree=u.ROOT/'artifacts/upstream-split3-next3/source/data'
    holdout=original.arrays(tree/'probe_3',labels=True);training=original.arrays(tree/'train_3')
    keys=np.sort(original.keys(*holdout[:3]))
    if len(keys)!=1760494 or np.any(keys[1:]==keys[:-1]):raise ValueError('Full unique holdout required')
    original.assert_disjoint(keys,training)
    if len(training[0])+len(holdout[0])!=100480507:raise ValueError('Complete original known split size differs')
    paths=[Path(__file__),Path(original.__file__),origin,fixed,bank_path,bank_path.parent/'column-verification.json',u.ROOT/'SPLIT3_FULL_HOLDOUT_PLAN.md']
    for stage in ['train_3','probe_3']:
        paths.extend(tree/stage/(name+'.npy') for name in ['user_idxs','item_idxs','dates','ratings'])
    u.write(REPORT/'verification.json',dict(state='complete_original_known_origin_and_all_split3_training_disjointness',
        native_training_rows=len(training[0]),full_holdout_rows=len(holdout[0]),canonical_rows=1128164,
        additional_known_rows=632330,all_original_native_training_identities_disjoint=True,
        original_frozen_canonical_input_hashes_verified=True,qualifying_labels_read=False,
        source_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths},models_fitted=0))
    print('All1760494 known holdout rows excluded from original98720013 training rows; frozen canonical origin verified',flush=True)

if __name__=='__main__':main()
