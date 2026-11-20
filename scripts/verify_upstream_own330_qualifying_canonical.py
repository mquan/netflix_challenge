#!/usr/bin/env python3
"""Separate complete frozen-vector and compensated recovered-group score check."""
from pathlib import Path
import gzip
import hashlib
import math
import numpy as np
import upstream_own330_assembly as study

u = study.u


def main():
    root, report = study.RUN, study.REPORT
    output = report / 'independent-canonical-verification.json'
    if output.exists():
        raise ValueError('Existing independent canonical qualifying proof retained')
    audit_path = report / 'independent-recovered-audit.json'; audit = u.read(audit_path)
    pre_path = report / 'independent-preaudit-verification.json'; pre = u.read(pre_path)
    frozen_path = report / 'frozen.json'; frozen = u.read(frozen_path)
    calibration_path = report / 'calibration.json'; calibration = u.read(calibration_path)
    plan_path = report / 'plan.json'; plan = u.read(plan_path)
    if (audit['state'] != 'complete_independent_two_frozen_own330_candidate_recovered_audit'
            or audit['preaudit_verification_sha256'] != u.sha(pre_path)
            or audit['freeze_sha256'] != u.sha(frozen_path)
            or pre['freeze_sha256'] != u.sha(frozen_path)
            or pre['calibration_sha256'] != u.sha(calibration_path)
            or frozen['plan_sha256'] != u.sha(plan_path)
            or frozen['calibration_sha256'] != u.sha(calibration_path)
            or frozen['qualifying_labels_read'] is not False
            or plan['candidates'] != study.CANDIDATES
            or audit['primary'] != plan['primary']
            or plan['primary'] != 'own7-own330-known-calibration'
            or pre['simplex_supports_examined'] != 127
            or pre['weights'] != calibration['weights']
            or not pre['all_stored_candidate_arithmetic_bitwise']
            or pre['qualifying_labels_read'] is not False
            or audit['qualifying_outcomes_used_for_fitting_or_selection'] is not False):
        raise ValueError('Original independent freeze/coefficient/arithmetic/audit chain required')
    rows, sources = study.checked_sources()
    if rows != calibration['families'] or sources != calibration['source_sha256'] or sources != pre['source_sha256']:
        raise ValueError('Complete original own lineage and actual pipeline evidence differs')
    weights = np.asarray(calibration['weights'])
    independent = np.asarray(pre['independent_weights'])
    if (weights.shape != (7,) or weights.min() < 0 or abs(weights.sum()-1) > 1e-12
            or np.max(abs(weights-independent)) > 1e-5):
        raise ValueError('Original independent all127-support solution differs')
    arrays = [np.fromfile(u.ROOT / r['qualifying_prediction'], dtype='<f4') for r in rows]
    tree = arrays[next(i for i,r in enumerate(rows) if r['id'] == 'tree154')]
    predictions = {}
    for name, ingredients, coefficients in [
            (study.CANDIDATES[0], [arrays[-1], tree], [.5, .5]),
            (study.CANDIDATES[1], arrays, weights)]:
        path = root / (name+'.f32')
        if u.sha(path) != frozen['predictions'][name]:
            raise ValueError('Original frozen qualifying vector changed')
        saved = np.fromfile(path, dtype='<f4')
        if saved.shape != (2817131,) or not np.isfinite(saved).all():
            raise ValueError('Complete finite canonical qualifying vector required')
        for lo in range(0, len(saved), 32768):
            hi = min(lo+32768, len(saved)); reference = np.zeros(hi-lo, dtype=np.float64)
            for values, coefficient in zip(ingredients, coefficients):
                reference += values[lo:hi].astype(np.float64)*float(coefficient)
            if not np.array_equal(reference.astype('<f4').view('u4'), saved[lo:hi].view('u4')):
                raise ValueError('Every frozen qualifying arithmetic bit must reconstruct')
        predictions[name] = np.clip(saved.astype(np.float64), 1.05, 4.95)
    path = u.ROOT / 'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz'
    payload = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob != '47ed4788ffba11aad029854a8e60d4dee04fc899' or audit['label_git_blob'] != blob:
        raise ValueError('Canonical recovered outcomes differ')
    with gzip.open(path, 'rt') as stream:
        labels = np.loadtxt(stream, delimiter=',', dtype=np.int16, skiprows=1)
    if (labels.shape != (2817131, 2) or not np.isin(labels[:, 0], [1,2,3,4,5]).all()
            or not np.isin(labels[:, 1], [0,1]).all()):
        raise ValueError('Complete canonical recovered domain differs')
    results = []
    for name, values in predictions.items():
        original = next(r for r in audit['results'] if r['candidate'] == name)
        groups = {}
        for group, count in [(0,1408789), (1,1408342)]:
            ids = np.flatnonzero(labels[:,1] == group)
            if len(ids) != count:
                raise ValueError('Complete canonical recovered group count differs')
            partials = []
            for lo in range(0, count, 16384):
                selected = ids[lo:lo+16384]
                errors = values[selected]-labels[selected,0]
                partials.append(math.fsum(float(e)*float(e) for e in errors))
            score = math.sqrt(math.fsum(partials)/count)
            key = 'source'+str(group)
            if abs(score-original['groups'][key]['rmse']) > 1e-12:
                raise ValueError('Separate complete canonical score differs')
            groups[key] = dict(rows=count, rmse=score)
        results.append(dict(candidate=name, groups=groups,
            both_groups_below_target=all(r['rmse'] < .85725 for r in groups.values())))
    primary = next(r for r in results if r['candidate'] == audit['primary'])
    prior_path = u.ROOT / 'experiments/upstream-own6-tight-calibration/independent-verification.json'
    prior = u.read(prior_path)['groups']
    study.own6.verify_bindings(sources)
    bindings = {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__), audit_path, pre_path,
        frozen_path, calibration_path, report / 'plan.json', path, prior_path]}
    u.write(output, dict(state='complete_independent_own330_calibration_and_frozen_group_scores',
        observed=study.own6.stamp(), primary=audit['primary'], groups=primary['groups'], results=results,
        previous_groups=prior, both_groups_improved=all(primary['groups'][k]['rmse'] < prior[k]['rmse'] for k in prior),
        both_groups_below_target=primary['both_groups_below_target'], weights=weights.tolist(),
        independent_simplex_weights=independent.tolist(), simplex_supports_examined=127,
        independently_solved_full_simplex=True, exact_known_and_full_stored_vectors_bitwise=True,
        all_original_development_and_full_family_replays_verified=True,
        actual_new_full_models_reopened=3, complete_known_statistics_scaler_normalized_bits_replayed=True,
        known_fit_rmse=pre['known_apparent_fit_rmse'], known_score_is_nested_cross_validation=False,
        qualifying_labels_used_for_fitting_or_selection=False, label_git_blob=blob,
        prediction_sha256=frozen['predictions'][audit['primary']], bindings=bindings,
        limitation='Adaptive community-recovered qualifying groups, not official or untouched Test. Original A development and paired B production remain distinct.'))
    print(dict(groups=primary['groups'], both_groups_improved=all(primary['groups'][k]['rmse'] < prior[k]['rmse'] for k in prior)), flush=True)


if __name__ == '__main__':
    main()
