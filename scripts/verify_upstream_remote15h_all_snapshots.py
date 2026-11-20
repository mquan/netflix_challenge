#!/usr/bin/env python3
"""Independent full canonical projection and complete paired-alias replay."""
import hashlib
from pathlib import Path
import numpy as np
import upstream_remote15h_all_snapshots as experiment
import upstream_remote15h as original

u=experiment.u


def digest(array):
    h=hashlib.sha256()
    for lo in range(0,len(array),32768):h.update(np.clip(array[lo:lo+32768],0,6).astype('<f4').tobytes())
    return h.hexdigest()


def main():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-verification.json'
    if output.exists():raise ValueError('Existing independent complete native snapshot evidence retained')
    plan=u.read(report/'plan.json');manifest=u.read(report/'manifest.json')
    if (manifest['state']!='complete_all48_native_known_projections_and_exact_paired_aliases_pending_independent_replay' or
        manifest['plan_sha256']!=u.sha(report/'plan.json') or plan['script_sha256']!=u.sha(Path(experiment.__file__)) or
        plan['verifier_sha256']!=u.sha(Path(__file__)) or plan['declaration_sha256']!=u.sha(experiment.DECL) or
        manifest['qualifying_labels_read'] is not False or manifest['models_fitted']!=0):raise ValueError('Original complete native snapshot freeze required')
    proof,names,bindings=experiment.sources()
    if bindings!=plan['source_sha256'] or bindings!=manifest['source_sha256'] or names!=plan['original_names']:
        raise ValueError('Original complete native source closure differs')
    split=u.ROOT/'artifacts/upstream-split3-pilot/source/data'
    full=original.arrays(split/'probe_3',labels=True);known=original.arrays(u.FRESH/'data/probex',labels=True)
    mapping=original.projection(full,known)
    if mapping.shape!=(1128164,) or not np.array_equal(mapping,np.load(experiment.native.RUN/'probex-row-map.npy')):
        raise ValueError('Complete canonical identity/rating projection must independently replay')
    if [r['name'] for r in manifest['all_projected_rows']]!=names:raise ValueError('Every original native name required in original order')
    records=[];rows=[];aliases=[];seen={}
    for row in manifest['all_projected_rows']:
        name=row['name'];raw=np.load(experiment.native.RUN/'source/preds_3'/(name+'.probe_3.npy'),mmap_mode='r')
        counterpart=experiment.native.RUN/'source/preds_3'/(name+'.qual.npy')
        path=root/(name+'.probex.npy');stored=np.load(path,mmap_mode='r');q=np.load(counterpart,mmap_mode='r')
        if (row['native_known']!=str((experiment.native.RUN/'source/preds_3'/(name+'.probe_3.npy')).relative_to(u.ROOT)) or
            row['paired_files']!=dict(probex=str(path.relative_to(u.ROOT)),qual=str(counterpart.relative_to(u.ROOT))) or
            row['known_sha256']!=u.sha(path) or row['qual_sha256']!=u.sha(counterpart) or
            raw.shape!=(1760494,) or stored.shape!=(1128164,) or q.shape!=(2817131,) or
            any(a.dtype!=np.dtype('<f4') for a in [raw,stored,q]) or
            not all(np.isfinite(a).all() for a in [raw,stored,q])):raise ValueError('Every complete stored native pair required')
        for lo in range(0,len(mapping),32768):
            hi=min(lo+32768,len(mapping))
            if not np.array_equal(raw[mapping[lo:hi]].view('u4'),stored[lo:hi].view('u4')):
                raise ValueError('Every stored projected prediction bit must match original native rows')
        key=(digest(stored),digest(q))
        if list(key)!=row['clipped_pair_sha256']:raise ValueError('Independent complete clipped pair identity differs')
        if key in seen:aliases.append(dict(name=name,alias_of=seen[key],clipped_pair_sha256=list(key)))
        else:seen[key]=name;rows.append(row)
        records.append(dict(name=name,rows=1128164,all_projection_bits_equal=True,qualifying_rows=2817131,
            known_sha256=row['known_sha256'],qual_sha256=row['qual_sha256']))
    recent=u.read(u.ROOT/'experiments/upstream-recent15-own-pairs/verification.json');previous=[]
    for name in proof['previous_compact_pairs_replayed']:
        old=next(r for r in recent['rows'] if r['name']==name)
        key=tuple(digest(np.load(u.ROOT/old['paired_files'][query],mmap_mode='r')) for query in ['probex','qual'])
        if key not in seen:raise ValueError('Original complete compact pair must be independently identifiable')
        previous.append(dict(name=name,canonical_name=seen[key],clipped_pair_sha256=list(key)))
    old_names={r['canonical_name'] for r in previous};added=[r for r in rows if r['name'] not in old_names]
    if (rows!=manifest['rows'] or aliases!=manifest['aliases'] or previous!=manifest['previous_compact_pairs'] or
        added!=manifest['new_rows'] or len(previous)!=9 or len(old_names)!=9):raise ValueError('Complete original native and previous compact alias partition differs')
    experiment.checks.verify_bindings(bindings)
    u.write(output,dict(state='complete_independent_all48_native_known_projections_qualifying_counterparts_and_paired_aliases',
        observed=experiment.checks.stamp(),manifest_sha256=u.sha(report/'manifest.json'),plan_sha256=u.sha(report/'plan.json'),
        verifier_sha256=u.sha(Path(__file__)),source_sha256=bindings,records=records,
        all_canonical_identity_label_map_rows_rebuilt=True,exact_aliases=len(aliases),distinct_pairs=len(rows),
        previous_pairs=9,additional_pairs=len(added),qualifying_labels_read=False,models_fitted=0,
        novelty_against_all352_inputs_claimed=False))
    print('Independent complete native snapshot projection/alias replay:',len(rows),'distinct,',len(added),'additional',flush=True)


if __name__=='__main__':main()
