"""Actual CPU tree fold reconstruction component; no production admission.

The future owned replay worker must independently rebuild and authenticate the
full raw matrix, ten ridge systems, original users/groups and saved receipts.
This component reopens every positive saved slot and checks every held bit.
"""
import hashlib
import os
from pathlib import Path
import numpy as np
import upstream_full_context_catboost as backend
from upstream_factored_rbm_admission import file_identity

SEEDS = (2026, 7, 13)


def sha_array(array, guard):
    h = hashlib.sha256()
    for lo in range(0, len(array), 8192):
        guard(); h.update(array[lo:lo + 8192].tobytes(order='C'))
    return h.hexdigest()


def authenticate_model(path, record, guard):
    path = Path(path)
    if path.resolve() != path:
        raise ValueError('Original canonical saved-model path required')
    h = hashlib.sha256()
    with path.open('rb') as source:
        original = file_identity(os.fstat(source.fileno()))
        if original != record['identity'] or original[2] != record['bytes']:
            raise ValueError('Original saved-model identity and byte count required')
        try:
            while True:
                guard(); block = source.read(1048576)
                if not block: break
                h.update(block)
            if h.hexdigest() != record['sha256']:
                raise ValueError('Every original saved-model byte must match its frozen digest')
        finally:
            if original != file_identity(os.fstat(source.fileno())) or original != file_identity(path.stat()):
                raise ValueError('Original model changed during complete hashing')


def verify(features, baselines, users, groups, receipt, models, expected_prediction, guard):
    """Reload all complete original positive slots; retain the zero denominator."""
    if not callable(guard): raise ValueError('External owned whole-group replay guard required')
    fold = receipt['fold']
    if (type(fold) is not int or not 0 <= fold < 5 or users.ndim != 1
            or users.dtype.kind not in 'iu' or np.any(users < 0) or groups.shape != users.shape
            or groups.dtype != np.int8 or set(np.unique(groups)) != set(range(10))
            or features.ndim != 2 or features.shape[0] != len(users)
            or features.shape[1] not in backend.WIDTHS or features.dtype != np.dtype('<f4')
            or not features.flags.c_contiguous or not features.flags.writeable
            or baselines.shape != (2, len(users)) or baselines.dtype != np.dtype('<f8')
            or np.may_share_memory(features, baselines)
            or receipt['seed_order'] != list(SEEDS) or receipt['seed_offset'] != fold
            or receipt['schedule'] != 'all_independent_inner_selections_then_fresh_full_refits'
            or receipt['correction_arithmetic'] != 'ordered_float64_sum_of_each_seed_prediction_divided_by_three'
            or receipt['qualifying_labels_read'] is not False):
        raise ValueError('Complete original raw fold and declared tree schedule required')
    order = np.argsort(users, kind='stable')
    if np.any((users[order][1:] == users[order][:-1]) & (groups[order][1:] != groups[order][:-1])):
        raise ValueError('Every user must retain its original group')
    outer = groups // 2
    held = outer == fold
    train = outer != fold
    inner = train & (groups % 2 == 0)
    if not held.any() or not inner.any() or not (train & (groups % 2 == 1)).any():
        raise ValueError('Every original inner and outer population required')
    counts = receipt['chosen_iterations']
    if (len(counts) != 3 or any(type(c) is not int or not 0 <= c <= backend.MAX_ROUNDS for c in counts)
            or len(receipt['inner_selection']) != 3
            or set(models) != {str(s) for s, c in zip(SEEDS, counts) if c}
            or expected_prediction.shape != (int(held.sum()),) or expected_prediction.dtype != np.dtype('<f4')
            or not np.isfinite(expected_prediction).all()):
        raise ValueError('Exact three slots, positive model inventory and complete held prediction required')
    for chosen, selection in zip(counts, receipt['inner_selection']):
        count = selection['best_iteration_zero_based'] + 1
        best = selection['best_validation_RMSE']; zero = selection['zero_correction_validation_MSE']
        if (type(count) is not int or not 1 <= count <= backend.MAX_ROUNDS
                or not np.isfinite(best) or best < 0 or not np.isfinite(zero) or zero < 0
                or selection['max_rounds'] != backend.MAX_ROUNDS or selection['patience'] != backend.PATIENCE
                or selection['qualifying_labels_read'] is not False
                or selection['selected_iterations'] != chosen
                or chosen != (count if best * best < zero else 0)):
            raise ValueError('Original complete known-only iteration selection arithmetic required')
    for index, stage in enumerate(('inner', 'full')):
        baseline = baselines[index]
        for lo in range(0, len(users), 8192):
            guard(); hi = min(lo + 8192, len(users)); values = baseline[lo:hi]
            if (not np.isfinite(features[lo:hi, :-1]).all() or not np.isfinite(values).all()
                    or np.any(values < 1.05) or np.any(values > 4.95)):
                raise ValueError('Complete finite raw operators and original clipped ridge baseline required')
            features[lo:hi, -1] = values.astype('<f4')
        expected = dict(raw_input_sha256=sha_array(features, guard), baseline_sha256=sha_array(baseline, guard),
                        eligible_rows=int((inner if index == 0 else train).sum()),
                        width=features.shape[1], rows=len(users))
        if receipt['populations'][stage] != expected:
            raise ValueError('Every original raw input and eligible baseline bit required')
    correction = np.zeros(int(held.sum()), dtype=np.float64)
    reopened = 0
    for seed, count in zip(SEEDS, counts):
        if not count: continue
        record = models[str(seed)]; path = Path(record['path'])
        if record['seed'] != seed + fold or record['iterations'] != count or record['width'] != features.shape[1]:
            raise ValueError('Exact original positive saved-model slot required')
        signature = record['signature']; parameters = dict(backend.PARAMETERS, random_seed=seed + fold, iterations=count)
        if (signature['tree_count'] != count or len(signature['feature_names']) != features.shape[1]
                or any(signature['flat_parameters'].get(k) != v for k, v in parameters.items())):
            raise ValueError('Original declared saved CPU configuration required')
        authenticate_model(path, record, guard)
        with backend.actual_model(path, record, signature, features.shape[1], guard) as model:
            # Independently reproduce the declared float64 seed/3 arithmetic.
            correction += backend.predict(model, features[held], guard) / 3
        reopened += 1
    reconstructed = np.clip(baselines[1, held] + correction, 1.05, 4.95).astype('<f4')
    if reconstructed.tobytes() != expected_prediction.tobytes():
        raise ValueError('Every independently reconstructed original held prediction bit required')
    guard()
    return dict(fold=fold, held_rows=int(held.sum()), actual_saved_models_reopened=reopened,
                seed_slots=3, every_held_prediction_bit_verified=True,
                every_original_raw_population_bit_verified=True, qualifying_labels_read=False)
