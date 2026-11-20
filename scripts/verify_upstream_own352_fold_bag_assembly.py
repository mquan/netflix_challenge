#!/usr/bin/env python3
"""Independent127-support solution and exact frozen vectors, followed by audit."""
import argparse
import gzip
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_own352_fold_bag_assembly as experiment
from verify_upstream_nn_pool6_assembly import simplex, stored

u = experiment.u


def verify():
    report, root = experiment.REPORT, experiment.RUN
    output = report / 'independent-preaudit-verification.json'
    if output.exists():
        raise ValueError('Existing independent preaudit proof retained')
    plan = u.read(report / 'plan.json'); cal = u.read(report / 'calibration.json')
    frozen = u.read(report / 'frozen.json')
    if (frozen['state'] != 'complete_two_declared_own352_foldbag_candidates_frozen_before_audit'
            or frozen['plan_sha256'] != u.sha(report / 'plan.json')
            or frozen['calibration_sha256'] != u.sha(report / 'calibration.json')
            or frozen['qualifying_labels_read'] is not False
            or plan['candidates'] != experiment.CANDIDATES
            or plan['verifier_sha256'] != u.sha(Path(__file__))):
        raise ValueError('Original two-candidate freeze required')
    rows, sources = experiment.checked_sources()
    if rows != cal['families'] or sources != cal['source_sha256']:
        raise ValueError('Original independently checked family list/source expectations differ')
    known = [np.fromfile(u.ROOT / r['known_prediction'], dtype='<f4') for r in rows]
    labels = np.load(experiment.own6.LABELS).astype(np.float64)
    independent, gram, rhs = simplex(known, labels)
    weights = np.asarray(cal['weights'])
    if (weights.shape != (7,) or weights.min() < 0 or abs(weights.sum()-1) > 1e-12
            or np.max(abs(weights-independent)) > 1e-5
            or np.max(abs(gram-np.asarray(cal['gram']))) > 1e-12
            or np.max(abs(rhs-np.asarray(cal['rhs']))) > 1e-12):
        raise ValueError('Independent all127-support solution/statistics differ')
    fitted = stored(root / 'known-fit.f32', known, weights)
    arrays = [np.fromfile(u.ROOT / r['qualifying_prediction'], dtype='<f4') for r in rows]
    tree = arrays[next(i for i, r in enumerate(rows) if r['id'] == 'tree154')]
    stored(root / (experiment.CANDIDATES[0]+'.f32'), [arrays[-1], tree], [.5, .5])
    stored(root / (experiment.CANDIDATES[1]+'.f32'), arrays, weights)
    for name, digest in frozen['predictions'].items():
        if u.sha(root / (name+'.f32')) != digest:
            raise ValueError('Original frozen vector changed')
    experiment.own6.verify_bindings(sources)
    clipped = np.clip(fitted.astype(np.float64), 1.05, 4.95)
    apparent = math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(clipped, labels))/len(labels))
    u.write(output, dict(state='complete_independent_own352_foldbag_127_support_and_all_frozen_arithmetic_before_audit',
        observed=experiment.own6.stamp(), freeze_sha256=u.sha(report / 'frozen.json'),
        calibration_sha256=u.sha(report / 'calibration.json'), verifier_sha256=u.sha(Path(__file__)),
        simplex_supports_examined=127, weights=weights.tolist(), independent_weights=independent.tolist(),
        known_apparent_fit_rmse=apparent, known_score_is_nested_cross_validation=False,
        all_stored_candidate_arithmetic_bitwise=True, source_sha256=sources, qualifying_labels_read=False))


def audit():
    from verify_upstream_complete13_calibration import require_paired_preaudits
    require_paired_preaudits()
    report, root = experiment.REPORT, experiment.RUN
    output = report / 'independent-recovered-audit.json'
    if output.exists():
        raise ValueError('Existing recovered audit retained')
    pre = report / 'independent-preaudit-verification.json'; proof = u.read(pre)
    frozen = u.read(report / 'frozen.json')
    if (proof['state'] != 'complete_independent_own352_foldbag_127_support_and_all_frozen_arithmetic_before_audit'
            or proof['freeze_sha256'] != u.sha(report / 'frozen.json')
            or proof['calibration_sha256'] != u.sha(report / 'calibration.json')
            or proof['verifier_sha256'] != u.sha(Path(__file__))
            or proof['qualifying_labels_read'] is not False
            or frozen['plan_sha256'] != u.sha(report / 'plan.json')
            or frozen['calibration_sha256'] != u.sha(report / 'calibration.json')):
        raise ValueError('Original independent coefficient/vector proof before outcomes required')
    experiment.own6.verify_bindings(proof['source_sha256'])
    predictions = {}
    for name in experiment.CANDIDATES:
        path = root / (name+'.f32')
        if u.sha(path) != frozen['predictions'][name]:
            raise ValueError('Frozen qualifying vector changed')
        predictions[name] = np.fromfile(path, dtype='<f4')
        if predictions[name].shape != (2817131,) or not np.isfinite(predictions[name]).all():
            raise ValueError('Complete finite candidate required')
    path = u.ROOT / 'experiments/upstream-item154-stream-production/recovered-audit/qual_ratings.csv.gz'
    payload = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob != '47ed4788ffba11aad029854a8e60d4dee04fc899':
        raise ValueError('Canonical recovered label payload differs')
    with gzip.open(path, 'rt') as stream:
        labels = np.loadtxt(stream, delimiter=',', skiprows=1, dtype=np.int16)
    if (labels.shape != (2817131, 2) or not np.isin(labels[:, 0], [1,2,3,4,5]).all()
            or not np.isin(labels[:, 1], [0,1]).all()):
        raise ValueError('Complete canonical recovered label domain differs')
    results = []
    for name, values in predictions.items():
        values = np.clip(values.astype(np.float64), 1.05, 4.95)
        groups = {}
        for group, count in [(0, 1408789), (1, 1408342)]:
            ids = np.flatnonzero(labels[:, 1] == group)
            if len(ids) != count:
                raise ValueError('Complete recovered group count differs')
            score = math.sqrt(math.fsum((float(values[i])-float(labels[i,0]))**2 for i in ids)/count)
            groups['source'+str(group)] = dict(rows=count, rmse=score, below_target=score < .85725)
        results.append(dict(candidate=name, groups=groups,
            both_groups_below_target=all(v['below_target'] for v in groups.values()),
            prediction_sha256=frozen['predictions'][name]))
    experiment.own6.verify_bindings(proof['source_sha256'])
    u.write(output, dict(state='complete_independent_two_frozen_own352_foldbag_candidate_recovered_audit',
        observed=experiment.own6.stamp(), primary='own7-own352-foldbag-known-calibration', results=results,
        label_git_blob=blob, preaudit_verification_sha256=u.sha(pre),
        freeze_sha256=u.sha(report / 'frozen.json'), qualifying_outcomes_used_for_fitting_or_selection=False,
        limitation='Adaptive community-recovered groups, not official or untouched Test. Original development A and paired production B remain distinct.'))
    print(results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--audit', action='store_true'); args = parser.parse_args()
    audit() if args.audit else verify()
