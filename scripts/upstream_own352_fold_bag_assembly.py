#!/usr/bin/env python3
"""Freeze the two previously declared own352 fold bag candidates using known-only weights."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import upstream_own6_tight_calibration as own6
import upstream_own352_fold_bag as full

u = own6.u
REPORT = u.ROOT / 'experiments/upstream-own352-fold-bag-assembly'
RUN = u.ROOT / 'artifacts/upstream-own352-fold-bag-assembly'
CANDIDATES = ['fixed-half-own352-foldbag-tree154', 'own7-own352-foldbag-known-calibration']


def checked_sources():
    rows,sources,_=own6.checked_sources()
    def bind(path,expected=None):
        key=str(path.relative_to(u.ROOT));value=u.sha(path) if expected is None else expected
        if key in sources and sources[key]!=value:raise ValueError('Original expected own source conflicts: '+key)
        sources[key]=value
    proof_path=full.REPORT/'independent-verification.json'
    proof=u.read(proof_path);plan=u.read(full.REPORT/'plan.json')
    completion_path=full.REPORT/'inference-completion.json';complete=u.read(completion_path)
    if (proof['state']!='complete_independent_all15_original_own352_fold_qualifying_models_and_uniform_bag' or
        proof['plan_sha256']!=u.sha(full.REPORT/'plan.json') or
        proof['inference_completion_sha256']!=u.sha(completion_path) or
        not proof['all428_578_input_operators_independently_reconstructed'] or
        not proof['uniform_five_fold_bag_bitwise_equal'] or proof['original_models_reopened']!=15 or
        proof['qualifying_labels_read'] is not False or proof['models_fitted']!=0 or
        [r['fold'] for r in proof['records']]!=list(range(5)) or
        not all(r['rows']==2817131 and r['actual_models_reopened']==3 and r['every_qualifying_prediction_bit_equal'] for r in proof['records']) or
        proof['source_sha256']!=plan['source_sha256'] or
        proof['prediction_sha256']!=complete['outputs']['nn-bag.f32']):
        raise ValueError('All15 original models, full qualifying operators and uniform bag replay required')
    for key,value in plan['source_sha256'].items():bind(u.ROOT/key,value)
    for name,value in complete['outputs'].items():bind(full.RUN/name,value)
    original_path=full.REPLAY/'independent-verification.json';original=u.read(original_path)
    canonical_path=full.REPLAY/'independent-canonical-verification.json';canonical=u.read(canonical_path)
    if (original['state']!='complete_actual_own352_bounded_original_statistics_scalers_and_saved_model_replays' or
        canonical['complete_model_preprocessing_verification_sha256']!=u.sha(original_path) or
        not original['complete_original_statistics_bitwise_equal'] or not original['all_original_models_preserved_without_refitting']):
        raise ValueError('Complete actual original eligible known-model/scaler replay required')
    known=full.study.RUN/'nn.f32';bind(known,original['nn_sha256'])
    for name in ['labels.npy','users.npy','groups.npy']:
        if not np.array_equal(np.load(full.study.RUN/name),np.load(u.ROOT/'artifacts/upstream-item154-tree'/name)):
            raise ValueError('All seven families require same canonical known rows/userfolds')
    for path in [Path(__file__),u.ROOT/'scripts/verify_upstream_own352_fold_bag_assembly.py',
                 u.ROOT/'scripts/upstream_complete13_calibration.py',
                 u.ROOT/'scripts/verify_upstream_complete13_calibration.py',
                 u.ROOT/'ALL_COMPLETE13_CALIBRATION_PLAN.md',
                 u.ROOT/'scripts/verify_upstream_nn_pool6_assembly.py',
                 u.ROOT/'OWN352_FOLD_BAG_PRODUCTION_PLAN.md',proof_path,full.REPORT/'plan.json',
                 completion_path,original_path,canonical_path]:bind(path)
    rows.append(dict(id='nn352-fold-bag',family='nn',development='upstream-own352-temporal-mps',
        production='upstream-own352-fold-bag',known_prediction=str(known.relative_to(u.ROOT)),
        qualifying_prediction=str((full.RUN/'nn-bag.f32').relative_to(u.ROOT)),
        original_fold_iterations=[u.read(full.study.REPORT/'nn'/('%s.json'%f))['chosen_iterations'] for f in range(5)]))
    own6.verify_bindings(sources)
    return rows,sources


def freeze():
    if REPORT.exists() or RUN.exists():
        raise ValueError('Existing own352 fold bag assembly evidence retained')
    REPORT.mkdir(); RUN.mkdir()
    u.write(REPORT / 'plan.json', dict(created=own6.stamp(), candidates=CANDIDATES,
        primary='own7-own352-foldbag-known-calibration', fixed_tree_weight=.5,
        original_declaration_sha256=u.sha(u.ROOT / 'OWN352_FOLD_BAG_PRODUCTION_PLAN.md'),
        rule='Exactly two previously declared candidates: fixed half own352 fold bag/tree154 and seven-family known-OOF simplex adding own352 fold bag to established own6. Uniform initialization, SLSQP ftol1e-17/maxiter1000. Independently check all127 supports and every stored vector before recovered outcomes. No new fitting/stopping/selection using qualifying outcomes.',
        script_sha256=u.sha(Path(__file__)), verifier_sha256=u.sha(u.ROOT / 'scripts/verify_upstream_own352_fold_bag_assembly.py'),
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
    u.write(REPORT / 'frozen.json', dict(state='complete_two_declared_own352_foldbag_candidates_frozen_before_audit',
        observed=own6.stamp(), plan_sha256=u.sha(REPORT / 'plan.json'),
        calibration_sha256=u.sha(REPORT / 'calibration.json'), qualifying_labels_read=False,
        predictions={p.stem:u.sha(p) for p in RUN.glob('*.f32')},
        known_calibration_is_nested_cv=False, independent_coefficient_arithmetic_verification_remaining=True))


if __name__ == '__main__':
    freeze()
