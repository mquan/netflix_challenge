"""Bounded complete-operator handoff for the planned full CatBoost study.

This numerical component neither creates an owned production namespace nor
admits a model. A future guarded producer must authenticate every source,
allocate durable sinks, seal their hashes, then exit before CatBoost starts.
No Torch or CatBoost import is needed in this handoff component.
"""
import hashlib
import numpy as np

BLOCK = 8192


def dimensions(retain_scalar, latent):
    if type(retain_scalar) is not bool or type(latent) is not bool:
        raise ValueError('Explicit original scalar-alias and context decisions required')
    return 562 + int(retain_scalar), 712 + int(retain_scalar) + 150 * int(latent)


def materialize(reader, weights, expected_streams, features, baselines, *,
                retain_scalar, latent, guard):
    """Stream every original row once; preserve ten separate ridge dot orders.

    weights and baselines use [outer fold, inner/full population, column/row].
    The final feature column is a zero placeholder; the consumer installs the
    matching population's float32 baseline before making its training Pool.
    Sinks may be caller-owned memmaps. Operator memory is bounded by8192-row blocks.
    """
    design, nonlinear = dimensions(retain_scalar, latent)
    count = len(reader.prediction)
    if (not callable(guard) or count <= 0
            or weights.shape != (5, 2, design) or weights.dtype != np.float64
            or not np.isfinite(weights).all()
            or features.shape != (count, nonlinear + 1) or features.dtype != np.dtype('<f4')
            or not features.flags.c_contiguous or not features.flags.writeable
            or baselines.shape != (5, 2, count) or baselines.dtype != np.dtype('<f8')
            or not baselines.flags.c_contiguous or not baselines.flags.writeable
            or np.may_share_memory(features, baselines)
            or np.may_share_memory(weights, features) or np.may_share_memory(weights, baselines)
            or len(expected_streams) != 2
            or any(type(h) is not str or len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
                   for h in expected_streams)):
        raise ValueError('Complete disjoint full-width sinks and all ten eligible ridge systems required')
    hashes = [hashlib.sha256(), hashlib.sha256()]
    for start in range(0, count, BLOCK):
        guard()
        stop = min(start + BLOCK, count)
        z, f = reader.block(start, stop)
        if (z.shape != (stop - start, design) or z.dtype != np.float64
                or f.shape != (stop - start, nonlinear) or f.dtype != np.dtype('<f4')
                or not np.isfinite(z).all() or not np.isfinite(f).all()):
            raise ValueError('Every complete finite original operator column required')
        hashes[0].update(z.tobytes(order='C'))
        hashes[1].update(f.tobytes(order='C'))
        features[start:stop, :-1] = f
        features[start:stop, -1] = 0
        for fold in range(5):
            for population in range(2):
                guard()
                # Preserve the original separate vector-dot and clipping order;
                # a single GEMM over ten columns can change rounding bits.
                values = np.clip(z @ weights[fold, population], 1.05, 4.95)
                if not np.isfinite(values).all():
                    raise ValueError('Finite complete eligible ridge predictions required')
                baselines[fold, population, start:stop] = values
    if [h.hexdigest() for h in hashes] != list(expected_streams):
        raise ValueError('Full original independent CPU operator stream proof required')
    guard()
    return dict(rows=count, width=nonlinear + 1, ridge_populations=10,
                operator_sha256=list(expected_streams), block_rows=BLOCK,
                final_feature_column='zero_placeholder_pending_matching_population',
                qualifying_labels_read=False, models_fitted=0)


def install_baseline(features, baseline, guard):
    """Use the same mutable raw matrix for one authenticated population.

    The future consumer must call this before constructing each inner/full
    training/validation Pool; quantization belongs to its training rows only.
    """
    if (not callable(guard) or features.ndim != 2 or features.shape[1] not in (713, 714, 863, 864)
            or features.dtype != np.dtype('<f4') or not features.flags.c_contiguous
            or not features.flags.writeable or baseline.shape != (len(features),)
            or baseline.dtype != np.dtype('<f8') or np.may_share_memory(features, baseline)):
        raise ValueError('One original full feature matrix and matching float64 baseline required')
    for start in range(0, len(features), BLOCK):
        guard(); stop = min(start + BLOCK, len(features))
        values = baseline[start:stop]
        if not np.isfinite(values).all() or np.any(values < 1.05) or np.any(values > 4.95):
            raise ValueError('Complete originally clipped eligible baseline required')
        features[start:stop, -1] = values
    guard()
