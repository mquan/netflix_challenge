"""Known-only five-fold/three-seed CatBoost residual schedule component.

No production namespace, source admission, resource owner or CLI is supplied.
The future owned consumer must authenticate every full stream and ridge
population before calling this routine and durably save every positive refit.
"""
import hashlib
import numpy as np
import upstream_full_context_catboost as backend
import upstream_full_context_catboost_stream as stream

SEEDS = (2026, 7, 13)


def masks(users, groups, fold):
    if (type(fold) is not int or not 0 <= fold < 5 or users.ndim != 1
            or groups.shape != users.shape or users.dtype.kind not in 'iu'
            or groups.dtype != np.int8 or np.any(users < 0)
            or set(np.unique(groups)) != set(range(10))):
        raise ValueError('Every original user group and a declared outer fold required')
    order = np.argsort(users, kind='stable')
    if np.any((users[order][1:] == users[order][:-1]) & (groups[order][1:] != groups[order][:-1])):
        raise ValueError('Each user must retain one original fold group')
    train = groups // 2 != fold
    result = dict(train=train, held=~train, fit=train & (groups % 2 == 0), valid=train & (groups % 2 == 1))
    if not all(mask.any() for mask in result.values()):
        raise ValueError('Every original nonempty training/held population required')
    return result


def digest(array, guard):
    result = hashlib.sha256()
    for lo in range(0, len(array), stream.BLOCK):
        guard(); result.update(array[lo:lo + stream.BLOCK].tobytes(order='C'))
    return result.hexdigest()


def run(features, baselines, labels, users, groups, fold, *, save_model, save_population,
        release, guard, train=backend.train, predict=backend.predict, select=backend.selected_iterations):
    """Independent inner selections, followed by fresh full positive refits.

    All training/validation calls receive only eligible raw rows. CatBoost fits
    each model's quantization on its training population. Held rows enter only
    prediction. Zero slots stay in the fixed three-seed denominator.
    """
    if (any(not callable(f) for f in (save_model, save_population, release, guard, train, predict, select))
            or labels.shape != users.shape or labels.dtype != np.float64
            or not np.isin(labels, [1, 2, 3, 4, 5]).all()
            or baselines.shape != (2, len(labels)) or baselines.dtype != np.dtype('<f8')
            or features.shape[0] != len(labels) or features.ndim != 2
            or features.shape[1] not in backend.WIDTHS or features.dtype != np.dtype('<f4')
            or not features.flags.c_contiguous or not features.flags.writeable
            or np.may_share_memory(features, baselines)):
        raise ValueError('Complete raw matrix, matching eligible baselines and known outcomes required')
    partition = masks(users, groups, fold)
    counts = []; inner_proofs = []; population_proofs = {}
    guard()
    for population, stage in enumerate(('inner', 'full')):
        baseline = baselines[population]
        stream.install_baseline(features, baseline, guard)
        # Seal every raw input bit before model callbacks; no normalization or
        # nonlinear OOF features are introduced by this learner.
        proof = dict(raw_input_sha256=digest(features, guard),
                     baseline_sha256=digest(baseline, guard),
                     eligible_rows=int(partition['fit' if population == 0 else 'train'].sum()),
                     width=features.shape[1], rows=len(features))
        save_population(stage, proof); population_proofs[stage] = proof
        if population == 0:
            for seed in SEEDS:
                guard()
                model = train(features[partition['fit']], labels[partition['fit']] - baseline[partition['fit']],
                              seed + fold, backend.MAX_ROUNDS, guard,
                              validation=(features[partition['valid']], labels[partition['valid']] - baseline[partition['valid']]))
                count, selected_proof = select(model, labels[partition['valid']] - baseline[partition['valid']])
                if type(count) is not int or not 0 <= count <= backend.MAX_ROUNDS:
                    raise ValueError('Original known-only iteration selection required')
                counts.append(count); inner_proofs.append(selected_proof)
                del model; release(); guard()
        else:
            correction = np.zeros(int(partition['held'].sum()), dtype=np.float64)
            for seed, count in zip(SEEDS, counts):
                if count:
                    guard()
                    model = train(features[partition['train']], labels[partition['train']] - baseline[partition['train']],
                                  seed + fold, count, guard)
                    if model.tree_count_ != count:
                        raise ValueError('Every original selected full-refit tree required')
                    values = predict(model, features[partition['held']], guard)
                    if values.shape != correction.shape or values.dtype != np.float64 or not np.isfinite(values).all():
                        raise ValueError('Complete finite float64 held tree predictions required')
                    correction += values / 3
                    save_model(model, seed, count)
                    del model, values; release(); guard()
            prediction = np.clip(baseline[partition['held']] + correction, 1.05, 4.95).astype('<f4')
        if (proof['raw_input_sha256'] != digest(features, guard)
                or proof['baseline_sha256'] != digest(baseline, guard)):
            raise ValueError('Original input matrix changed during model callbacks')
    guard()
    return dict(fold=fold, chosen_iterations=counts, seed_order=list(SEEDS), seed_offset=fold,
                prediction=prediction, inner_selection=inner_proofs, populations=population_proofs,
                schedule='all_independent_inner_selections_then_fresh_full_refits',
                correction_arithmetic='ordered_float64_sum_of_each_seed_prediction_divided_by_three',
                qualifying_labels_read=False)
