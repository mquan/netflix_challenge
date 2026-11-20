#!/usr/bin/env python3
"""Freeze the two previously declared own330 candidates using known-only weights."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import upstream_own6_tight_calibration as own6
import upstream_own330_full_refit as full

u = own6.u
REPORT = u.ROOT / 'experiments/upstream-own330-assembly'
RUN = u.ROOT / 'artifacts/upstream-own330-assembly'
CANDIDATES = ['fixed-half-own330-tree154', 'own7-own330-known-calibration']


def checked_sources():
    rows, sources, _ = own6.checked_sources()
    def bind(path, expected=None):
        key = str(path.relative_to(u.ROOT)); actual = u.sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Original expected own source changed: '+key)
        if key in sources and sources[key] != actual:
            raise ValueError('Conflicting original own source expectation: '+key)
        sources[key] = actual
    proof_path = full.REPORT / 'independent-verification.json'
    proof = u.read(proof_path); plan = u.read(full.REPORT / 'plan.json')
    complete = u.read(full.REPORT / 'fit-completion.json')
    if (proof['state'] != 'complete_original_own330_full_known_preprocessing_and_both_actual_model_queries'
            or proof['plan_sha256'] != u.sha(full.REPORT / 'plan.json')
            or proof['completion_sha256'] != u.sha(full.REPORT / 'fit-completion.json')
            or not proof['complete_known_statistics_scaler_normalized_bits_replayed']
            or proof['qualifying_labels_read'] is not False
            or [r['query'] for r in proof['records']] != ['probex', 'qual']
            or not all(r['all_prediction_bits_equal'] and r['actual_models_reopened'] == 3 for r in proof['records'])
            or proof['outputs'] != complete['outputs']):
        raise ValueError('Complete independent original own330 full pipeline required')
    for key, expected in plan['source_sha256'].items():
        bind(u.ROOT / key, expected)
    for name, expected in proof['outputs'].items():
        bind(full.RUN / name, expected)
    development = u.ROOT / 'experiments/upstream-own330-mps'
    original = u.read(development / 'independent-verification.json')
    if original['state'] != 'complete_actual_own330_all_fits_original_statistics_scalers_and_saved_model_replays':
        raise ValueError('Complete original development predictions required')
    known = u.ROOT / 'artifacts/upstream-own330-mps/nn.f32'
    bind(known, original['nn_sha256'])
    for name in ['labels.npy', 'users.npy', 'groups.npy']:
        if not np.array_equal(np.load(u.ROOT / 'artifacts/upstream-own330-mps' / name),
                              np.load(u.ROOT / 'artifacts/upstream-item154-tree' / name)):
            raise ValueError('Own330 and original six families must share canonical rows/folds')
    for path in [Path(__file__), u.ROOT / 'scripts/verify_upstream_own330_assembly.py',
                 u.ROOT / 'scripts/verify_upstream_nn_pool6_assembly.py',
                 u.ROOT / 'OWN330_FULL_REFIT_PLAN.md', proof_path, full.REPORT / 'plan.json',
                 full.REPORT / 'fit-completion.json', development / 'independent-verification.json']:
        bind(path)
    rows.append(dict(id='nn330', family='nn', development='upstream-own330-mps',
        production='upstream-own330-full-refit', known_prediction=str(known.relative_to(u.ROOT)),
        qualifying_prediction=str((full.RUN / 'qual.f32').relative_to(u.ROOT)),
        original_full_iterations=proof['full_iterations']))
    return rows, sources


def freeze():
    if REPORT.exists() or RUN.exists():
        raise ValueError('Existing own330 assembly evidence retained')
    REPORT.mkdir(); RUN.mkdir()
    u.write(REPORT / 'plan.json', dict(created=own6.stamp(), candidates=CANDIDATES,
        primary='own7-own330-known-calibration', fixed_tree_weight=.5,
        original_declaration_sha256=u.sha(u.ROOT / 'OWN330_FULL_REFIT_PLAN.md'),
        rule='Exactly two previously declared candidates: fixed half own330/tree154 and seven-family known-OOF simplex adding own330 to established own6. Uniform initialization, SLSQP ftol1e-17/maxiter1000. Independently check all127 supports and every stored vector before recovered outcomes. No new fitting/stopping/selection using qualifying outcomes.',
        script_sha256=u.sha(Path(__file__)), verifier_sha256=u.sha(u.ROOT / 'scripts/verify_upstream_own330_assembly.py'),
        qualifying_labels_read=False, known_calibration_is_nested_cv=False,
        input_realization='Development/calibration A and paired full-refit production B remain distinct. No matched B-trained CV claim.'))
    rows, sources = checked_sources()
    known = [np.fromfile(u.ROOT / r['known_prediction'], dtype='<f4') for r in rows]
    labels = np.load(own6.LABELS).astype(np.float64)
    if any(a.shape != labels.shape or not np.isfinite(a).all() for a in known):
        raise ValueError('Complete finite known vectors required')
    x = np.column_stack(known).astype(np.float64); d = x-x[:, -1, None]
    residual = labels-x[:, -1]; gram = d.T @ d / len(labels); rhs = d.T @ residual / len(labels)
    size = len(rows)
    if size != 7:
        raise ValueError('All seven declared families required')
    optimum = minimize(lambda w:float(w@gram@w-2*rhs@w), np.full(size, 1/size),
        jac=lambda w:2*(gram@w-rhs), bounds=[(0., 1.)]*size,
        constraints=[dict(type='eq', fun=lambda w:float(w.sum()-1), jac=lambda w:np.ones(size))],
        method='SLSQP', options=dict(ftol=1e-17, maxiter=1000))
    if not optimum.success or optimum.x.min() < -1e-12 or abs(optimum.x.sum()-1) > 1e-10:
        raise ValueError('Known-only simplex fit did not converge')
    weights = np.maximum(optimum.x, 0); weights /= weights.sum()
    own6.combined(known, weights).tofile(RUN / 'known-fit.f32')
    u.write(REPORT / 'calibration.json', dict(families=rows, family_ids=[r['id'] for r in rows],
        weights=weights.tolist(), gram=gram.tolist(), rhs=rhs.tolist(), source_sha256=sources,
        known_score_is_nested_cross_validation=False, qualifying_labels_read=False))
    del x, d, residual
    arrays = [np.fromfile(u.ROOT / r['qualifying_prediction'], dtype='<f4') for r in rows]
    if any(a.shape != (2817131,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Complete finite qualifying vectors required')
    tree = arrays[next(i for i, r in enumerate(rows) if r['id'] == 'tree154')]
    own6.combined([arrays[-1], tree], [.5, .5]).tofile(RUN / (CANDIDATES[0]+'.f32'))
    own6.combined(arrays, weights).tofile(RUN / (CANDIDATES[1]+'.f32'))
    own6.verify_bindings(sources)
    u.write(REPORT / 'frozen.json', dict(state='complete_two_declared_own330_candidates_frozen_before_audit',
        observed=own6.stamp(), plan_sha256=u.sha(REPORT / 'plan.json'),
        calibration_sha256=u.sha(REPORT / 'calibration.json'), qualifying_labels_read=False,
        predictions={p.stem:u.sha(p) for p in RUN.glob('*.f32')},
        known_calibration_is_nested_cv=False, independent_coefficient_arithmetic_verification_remaining=True))


if __name__ == '__main__':
    freeze()
