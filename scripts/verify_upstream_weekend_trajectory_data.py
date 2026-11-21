#!/usr/bin/env python3
"""Independent full candidate/operator/system/held-prediction reconstruction."""
import argparse
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_weekend_trajectory_data as study
import verify_upstream_complete_cf_delta as legacy

u, ops = study.u, study.ops


def independent_candidates():
    import verify_upstream_complete_trajectory_data as local
    rows=list(local.independent_candidates())
    weekend=u.read(u.ROOT/'experiments/upstream-remote-weekend/plan.json')
    for job in weekend['jobs']:
        prefix='artifacts/upstream-remote-weekend-return/'+job['name']+'/native/preds_3/'
        for column in job['published_columns']:
            rows.append(dict(id='weekend:'+column,paired_files={'probex':prefix+column+'.probe_3.npy',
                'qual':prefix+column+'.qual.npy'},known_row_map='artifacts/upstream-split3-pilot/probex-row-map.npy'))
    if len(rows)!=195:raise ValueError('Independent complete128+67 candidate order required')
    return rows


def scalar(row, query, lo, hi):
    values = np.load(u.ROOT/row['paired_files'][query], mmap_mode='r', allow_pickle=False)
    projected='known_row_map' in row and query=='probex'
    expected=1760494 if projected else ops.COUNTS[query]
    if values.shape != (expected,) or values.dtype != np.dtype('<f4'):
        raise ValueError('Exact complete original scalar schema required')
    if projected:
        mapping=np.load(u.ROOT/row['known_row_map'],mmap_mode='r',allow_pickle=False)
        if mapping.shape!=(1128164,) or mapping.dtype!=np.dtype('<i8'):raise ValueError('Original complete projection schema required')
        selected=mapping[lo:hi]
        if np.any(selected<0) or np.any(selected>=expected):raise ValueError('Original projection bounds required')
        raw=np.take(values,selected)
    else:raw=values[lo:hi]
    if not np.isfinite(raw).all():
        raise ValueError('Every original scalar must be finite before clipping')
    # Explicit comparisons preserve signed zero, as the original clip does.
    return np.where(raw < 0, np.float32(0), np.where(raw > 6, np.float32(6), raw)).astype('<f4')


def extend(z, f, rows, query, lo, hi):
    count = len(rows)
    design = np.empty((hi-lo, 467+count), dtype=np.float64)
    features = np.empty((hi-lo, 617+count), dtype='<f4')
    design[:, :466] = z[:, :466]; design[:, -1] = z[:, -1]
    features[:, :617] = f
    for index, row in enumerate(rows):
        value = scalar(row, query, lo, hi)
        design[:, 466+index] = value
        features[:, 617+index] = value-np.float32(3.6)
    return design, features


def evidence():
    m, candidates, bindings, cf_hashes = study.sources()
    plan = u.read(study.REPORT/'plan.json'); manifest = u.read(study.REPORT/'input-manifest.json')
    if (candidates != independent_candidates() or plan['candidates'] != candidates or
        plan['source_sha256'] != bindings or manifest['source_sha256'] != bindings or
        manifest['plan_sha256'] != ops.sha(study.REPORT/'plan.json') or
        plan['worker_sha256'] != ops.sha(Path(study.__file__)) or
        plan['verifier_sha256'] != ops.sha(Path(__file__)) or plan['declaration_sha256'] != ops.sha(study.DECL) or
        plan['lambda_fixed'] != 1000 or plan['folds'] != 5 or plan['groups'] != 10 or
        plan['qualifying_labels_read'] is not False or manifest['qualifying_labels_read'] is not False or
        manifest['models_fitted'] != 0 or not manifest['matched_B_known_and_qualifying'] or
        manifest['state'] != 'complete_all586_scalar_pairs_pending_independent_replay'):
        raise ValueError('Original full source/declaration/input list required')
    return m, candidates, bindings, cf_hashes, manifest


def admit():
    output = study.REPORT/'input-verification.json'
    if output.exists(): raise ValueError('Existing independent full input proof retained')
    guard = study.Guard(); m, candidates, bindings, cf_hashes, im = evidence()
    hashes = {}
    for query, count in ops.COUNTS.items():
        reader = legacy.Independent(query, m)
        original = [hashlib.sha256() for _ in range(391)]
        extra = [hashlib.sha256() for _ in candidates]
        full = [hashlib.sha256(), hashlib.sha256()]
        try:
            for block, lo in enumerate(range(0, count, 8192)):
                hi = min(lo+8192, count); z, f = reader.block(lo, hi)
                full[0].update(z.tobytes()); full[1].update(f.tobytes())
                for index, column in enumerate(list(range(62))+list(range(137, 466))):
                    raw = z[:, column].astype('<f4')
                    value = np.where(raw < 0, np.float32(0), np.where(raw > 6, np.float32(6), raw))
                    original[index].update(value.tobytes())
                for index, row in enumerate(candidates):
                    extra[index].update(scalar(row, query, lo, hi).tobytes())
                if block % 20 == 0: guard.check()
        finally: reader.close()
        if [h.hexdigest() for h in full] != m['virtual_c_order_sha256'][query]:
            raise ValueError('Every independent467/617 pairedB391 bit must match')
        hashes[query] = dict(original=[h.hexdigest() for h in original], candidates=[h.hexdigest() for h in extra])
        if hashes[query]['original']+hashes[query]['candidates'][:97] != cf_hashes[query]:
            raise ValueError('Full original391/CF97 hashes must remain equal')
        prior=u.read(u.ROOT/'experiments/upstream-complete-trajectory-data/input-manifest.json')['clipped_sha256'][query]
        expected=list(prior['original'])+list(prior['candidates'])
        for job in u.read(u.ROOT/'experiments/upstream-remote-weekend/plan.json')['jobs']:
            proof=u.read(u.ROOT/'experiments/upstream-remote-weekend-return'/job['name']/'independent-verification.json')
            expected.extend(proof['clipped_pair_sha256'][column][query] for column in job['published_columns'])
        if len(expected)!=586 or hashes[query]['original']+hashes[query]['candidates'] != expected:
            raise ValueError('All original local and remote admission hashes must replay independently')
        print('Independent complete586 scalar query:', query, flush=True)
    if hashes != im['clipped_sha256']:
        raise ValueError('All complete scalar bits must match the producer')
    seen = {}; kept = []; aliases = []
    for column in range(391):
        key = (hashes['probex']['original'][column], hashes['qual']['original'][column])
        if key not in seen: seen[key] = 'original391:%03d'%column
    for column, row in enumerate(candidates):
        key = (hashes['probex']['candidates'][column], hashes['qual']['candidates'][column])
        if key in seen:
            aliases.append(dict(id=row['id'], alias_of=seen[key], clipped_pair_sha256=list(key)))
        else:
            kept.append(row); seen[key] = row['id']
    if (kept != im['retained'] or aliases != im['aliases'] or im['raw_predictions'] != 391+len(kept) or
        [im[k] for k in ['design_columns', 'nonlinear_columns', 'nn_inputs']] != [467+len(kept), 617+len(kept), 618+len(kept)]):
        raise ValueError('Independent complete retained/alias partition differs')
    study.verify_bindings(bindings); guard.check()
    u.write(output, dict(state='complete_independent_all586_scalar_pairs_and_both_operators',
        observed=study.stamp(), input_manifest_sha256=ops.sha(study.REPORT/'input-manifest.json'), source_sha256=bindings,
        original391_retained=True, all195_candidates_replayed=True, retained_additions=len(kept), exact_paired_aliases=len(aliases),
        every_complete_parent_operator_bit_equal=True, qualifying_labels_read=False, models_fitted=0))


def fit():
    output = study.REPORT/'independent-verification.json'
    if output.exists(): raise ValueError('Existing full independent ridge proof retained')
    guard = study.Guard(); m, _, bindings, _, im = evidence()
    manifest = u.read(study.REPORT/'manifest.json'); iv = u.read(study.REPORT/'input-verification.json')
    if (manifest['state'] != 'complete_all_trajectory_operators_and_known_ridge_pending_independent_replay' or
        manifest['source_sha256'] != bindings or manifest['retained'] != im['retained'] or
        manifest['plan_sha256'] != ops.sha(study.REPORT/'plan.json') or
        manifest['input_manifest_sha256'] != ops.sha(study.REPORT/'input-manifest.json') or
        manifest['input_verification_sha256'] != ops.sha(study.REPORT/'input-verification.json') or
        iv['state'] != 'complete_independent_all586_scalar_pairs_and_both_operators' or
        iv['input_manifest_sha256'] != ops.sha(study.REPORT/'input-manifest.json') or iv['source_sha256'] != bindings or
        manifest['qualifying_labels_read'] is not False or iv['qualifying_labels_read'] is not False or
        manifest['neural_models_fitted'] != 0 or manifest['nonlinear_component_OOF_predictions_used_as_features'] is not False):
        raise ValueError('Complete original independently admitted ridge study required')
    if ([manifest[k] for k in ['raw_predictions', 'design_columns', 'nonlinear_columns', 'nn_inputs']] !=
        [im[k] for k in ['raw_predictions', 'design_columns', 'nonlinear_columns', 'nn_inputs']] or
        manifest['rows'] != 1128164 or manifest['qualifying_rows'] != 2817131 or
        not manifest['matched_B_known_and_qualifying'] or not manifest['complete_original_B391_operator_bits_preserved']):
        raise ValueError('Full dynamic dimensions, canonical rows and pairedB391 retention required')
    for name, digest in manifest['prepared_sha256'].items():
        if ops.sha(study.RUN/name) != digest: raise ValueError('Actual full ridge artifact changed')
    y, groups, users = [np.load(study.RUN/(name+'.npy')) for name in ['labels', 'groups', 'users']]
    order = np.argsort(users, kind='stable')
    if (y.shape != (1128164,) or groups.shape != y.shape or users.shape != y.shape or
        set(np.unique(groups)) != set(range(10)) or
        np.any((users[order][1:] == users[order][:-1]) & (groups[order][1:]//2 != groups[order][:-1]//2))):
        raise ValueError('Full canonical known labels and held-user folds required')
    d = 467+len(im['retained']); gram=np.zeros((10, d, d)); rhs=np.zeros((10, d))
    cg, cr = np.zeros((10, 467, 467)), np.zeros((10, 467)); streams={}
    for query, count in ops.COUNTS.items():
        reader = legacy.Independent(query, m); hashes=[hashlib.sha256(), hashlib.sha256()]
        try:
            for block, lo in enumerate(range(0, count, 8192)):
                hi=min(lo+8192, count); parent_z, parent_f=reader.block(lo, hi)
                z, f=extend(parent_z, parent_f, im['retained'], query, lo, hi)
                hashes[0].update(z.tobytes()); hashes[1].update(f.tobytes())
                if query == 'probex':
                    for group in range(10):
                        keep=groups[lo:hi] == group; a=z[keep]; control=parent_z[keep]
                        gram[group] += a.T@a; rhs[group] += a.T@y[lo:hi][keep]
                        cg[group] += control.T@control; cr[group] += control.T@y[lo:hi][keep]
                if block % 20 == 0: guard.check()
        finally: reader.close()
        streams[query]=[h.hexdigest() for h in hashes]
    with np.load(study.RUN/'stats.npz') as saved:
        for name, values in [('gram', gram), ('rhs', rhs), ('control_gram', cg), ('control_rhs', cr)]:
            if saved[name].shape != values.shape or not np.array_equal(saved[name].view('u8'), values.view('u8')):
                raise ValueError('Every original full statistic bit must match')
    if streams != manifest['virtual_c_order_sha256']:
        raise ValueError('Both complete independently expanded input operators must match')
    with np.load(study.parent.RUN/'stats.npz') as old:
        if not np.array_equal(cg.view('u8'), old['gram'].view('u8')) or not np.array_equal(cr.view('u8'), old['rhs'].view('u8')):
            raise ValueError('Full original matched391 control statistic bits must remain identical')
    weights=[]
    for g, r, name in [(gram, rhs, 'weights.npy'), (cg, cr, 'control391-weights.npy')]:
        penalty=np.eye(r.shape[1])*1000; penalty[-1, -1]=0
        expected=np.stack([np.linalg.solve(g[np.arange(10)//2 != fold].sum(0)+penalty,
                                          r[np.arange(10)//2 != fold].sum(0)) for fold in range(5)])
        actual=np.load(study.RUN/name)
        if actual.shape != expected.shape or not np.array_equal(actual.view('u8'), expected.view('u8')):
            raise ValueError('Every full eligible ridge coefficient bit must match')
        weights.append(actual)
    vectors=[np.fromfile(study.RUN/name, dtype='<f4') for name in ['ridge.f32', 'control391.f32']]
    if any(v.shape != y.shape or not np.isfinite(v).all() for v in vectors):
        raise ValueError('Full finite held-user prediction vectors required')
    reader=legacy.Independent('probex', m)
    try:
        for block, lo in enumerate(range(0, len(y), 8192)):
            hi=min(lo+8192, len(y)); parent_z, parent_f=reader.block(lo, hi)
            z, _=extend(parent_z, parent_f, im['retained'], 'probex', lo, hi)
            for fold in range(5):
                held=groups[lo:hi]//2 == fold
                for values, a, coefficients in zip(vectors, [z, parent_z], weights):
                    expected=np.clip(a[held]@coefficients[fold], 1.05, 4.95).astype('<f4')
                    if not np.array_equal(values[lo:hi][held].view('u4'), expected.view('u4')):
                        raise ValueError('Every actual stored held prediction bit must replay')
            if block % 20 == 0: guard.check()
    finally: reader.close()
    scores=[math.sqrt(math.fsum((float(a)-float(b))**2 for a, b in zip(np.clip(v.astype(np.float64), 1.05, 4.95), y))/len(y)) for v in vectors]
    if (abs(scores[0]-manifest['ridge_rmse']) > 1e-12 or abs(scores[1]-manifest['control391_rmse']) > 1e-12 or
        ops.sha(study.RUN/'control391.f32') != m['prepared_sha256']['ridge.f32']):
        raise ValueError('Full canonical expanded/control ridge scores and control bits must match')
    study.verify_bindings(bindings); guard.check()
    u.write(output, dict(state='complete_independent_all_trajectory_operators_statistics_ridge_and_stored_predictions',
        observed=study.stamp(), manifest_sha256=ops.sha(study.REPORT/'manifest.json'), source_sha256=bindings,
        input_verification_sha256=ops.sha(study.REPORT/'input-verification.json'), raw_predictions=391+len(im['retained']),
        design_columns=d, nonlinear_columns=617+len(im['retained']), nn_inputs=618+len(im['retained']),
        all_original_paired_B391_bits_preserved=True, matched_B_known_and_qualifying=True,
        every_full_operator_statistic_solve_and_prediction_bit_equal=True, ridge_rmse=scores[0], control391_rmse=scores[1],
        qualifying_labels_read=False, neural_models_fitted=0))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase', choices=['admit', 'fit']); args=parser.parse_args()
    {'admit': admit, 'fit': fit}[args.phase]()
