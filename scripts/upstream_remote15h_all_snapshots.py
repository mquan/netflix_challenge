#!/usr/bin/env python3
"""Project every completed remote native query pair without outcome selection."""
from pathlib import Path
import hashlib
import shutil
import numpy as np
import upstream_remote15h_native_stream as native
import upstream_own6_tight_calibration as checks

u=native.u
REPORT=u.ROOT/'experiments/upstream-remote15h-all-snapshots'
RUN=u.ROOT/'artifacts/upstream-remote15h-all-snapshots'
DECL=u.ROOT/'REMOTE15H_ALL_NATIVE_SNAPSHOTS_PLAN.md'


def clipped_sha(array):
    digest=hashlib.sha256()
    for lo in range(0,len(array),65536):digest.update(np.clip(array[lo:lo+65536],0,6).astype('<f4').tobytes())
    return digest.hexdigest()


def sources():
    path=native.REPORT/'independent-verification.json';proof=u.read(path)
    transport=native.REPORT/'transport-verification.json';manifest=native.REPORT/native.ARCHIVE
    if (proof['state']!='complete_independent_original_native_queries_source_and_all_previous_compact_projections' or
        proof['transport_verification_sha256']!=u.sha(transport) or proof['manifest_sha256']!=u.sha(manifest) or
        not proof['complete_native_holdout_absent_from_training'] or proof['native_holdout_rows']!=1760494 or
        proof['qualifying_labels_read'] is not False or proof['models_fitted']!=0 or len(proof['queries'])!=96):
        raise ValueError('Complete original native source/query and full holdout admission required')
    bindings=dict(proof['source_sha256'])
    def bind(p,expected=None):
        relative=str(p.relative_to(u.ROOT));digest=u.sha(p) if expected is None else expected
        if relative in bindings and bindings[relative]!=digest:raise ValueError('Original native source expectation conflicts')
        bindings[relative]=digest
    for row in proof['queries']:bind(native.RUN/row['path'],row['sha256'])
    for p in [Path(__file__),u.ROOT/'scripts/verify_upstream_remote15h_all_snapshots.py',DECL,path,transport,manifest]:bind(p)
    checks.verify_bindings(bindings)
    names=sorted(row['path'].removeprefix('source/preds_3/').removesuffix('.probe_3.npy')
        for row in proof['queries'] if row['path'].endswith('.probe_3.npy'))
    if len(names)!=48 or len(set(names))!=48:raise ValueError('All48 original complete native pairs required')
    return proof,names,bindings


def main():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing whole native snapshot evidence retained')
    proof,names,bindings=sources()
    if shutil.disk_usage(u.ROOT).free<18.75*2**30+48*1128164*4+600*2**20:
        raise ValueError('All native projections plus original disk floor/checkpoint reserve required')
    mapping=np.load(native.RUN/'probex-row-map.npy')
    if mapping.shape!=(1128164,) or not np.issubdtype(mapping.dtype,np.integer) or mapping.min()<0 or mapping.max()>=1760494:
        raise ValueError('Complete original canonical map required')
    REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,original_names=names,
        declaration_sha256=u.sha(DECL),script_sha256=u.sha(Path(__file__)),
        verifier_sha256=u.sha(u.ROOT/'scripts/verify_upstream_remote15h_all_snapshots.py'),
        qualifying_labels_read=False,rule='All48 pairs; exact complete clipped-pair aliases only; no scores or fitting.'))
    rows=[];aliases=[];seen={};all_rows=[]
    for name in names:
        original=native.RUN/'source/preds_3'/(name+'.probe_3.npy')
        counterpart=native.RUN/'source/preds_3'/(name+'.qual.npy')
        raw=np.load(original,mmap_mode='r');q=np.load(counterpart,mmap_mode='r')
        if raw.shape!=(1760494,) or q.shape!=(2817131,) or raw.dtype!=np.dtype('<f4') or q.dtype!=np.dtype('<f4'):
            raise ValueError('Both complete original native shapes/dtypes required')
        values=raw[mapping]
        if not np.isfinite(values).all() or not np.isfinite(q).all():raise ValueError('Finite complete native pair required')
        path=RUN/(name+'.probex.npy');np.save(path,values,allow_pickle=False)
        key=(clipped_sha(values),clipped_sha(q))
        row=dict(name=name,paired_files=dict(probex=str(path.relative_to(u.ROOT)),qual=str(counterpart.relative_to(u.ROOT))),
            native_known=str(original.relative_to(u.ROOT)),known_sha256=u.sha(path),qual_sha256=u.sha(counterpart),
            clipped_pair_sha256=list(key))
        all_rows.append(row)
        if key in seen:aliases.append(dict(name=name,alias_of=seen[key],clipped_pair_sha256=list(key)))
        else:seen[key]=name;rows.append(row)
        if shutil.disk_usage(u.ROOT).free<18.75*2**30+600*2**20:raise ValueError('Original checkpoint/disk reserve must remain')
    recent=u.read(u.ROOT/'experiments/upstream-recent15-own-pairs/verification.json');previous=[]
    for name in proof['previous_compact_pairs_replayed']:
        old=next(r for r in recent['rows'] if r['name']==name)
        key=tuple(clipped_sha(np.load(u.ROOT/old['paired_files'][query],mmap_mode='r')) for query in ['probex','qual'])
        if key not in seen:raise ValueError('Every previously verified compact pair must have its full numerical alias')
        previous.append(dict(name=name,canonical_name=seen[key],clipped_pair_sha256=list(key)))
    if len(previous)!=9 or len({r['canonical_name'] for r in previous})!=9:raise ValueError('All nine original compact pairs required')
    old_names={r['canonical_name'] for r in previous};added=[r for r in rows if r['name'] not in old_names]
    checks.verify_bindings(bindings)
    u.write(REPORT/'manifest.json',dict(state='complete_all48_native_known_projections_and_exact_paired_aliases_pending_independent_replay',
        source_sha256=bindings,plan_sha256=u.sha(REPORT/'plan.json'),rows=rows,all_projected_rows=all_rows,
        aliases=aliases,previous_compact_pairs=previous,new_rows=added,qualifying_labels_read=False,models_fitted=0,
        novelty_against_all352_inputs_claimed=False))
    print('Complete native snapshots:',len(rows),'distinct,',len(added),'beyond previous nine; independent replay remains',flush=True)


if __name__=='__main__':main()
