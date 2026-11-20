#!/usr/bin/env python3
"""Memory-bounded operations with the frozen scaler's complete statistical inputs."""
import numpy as np


def append_baseline(features, baseline):
    if features.ndim != 2 or baseline.shape != (len(features),):
        raise ValueError('Complete row-aligned features and baseline required')
    out = np.empty((len(features), features.shape[1] + 1), dtype=np.float32)
    out[:, :-1] = features
    out[:, -1] = baseline
    return out


def fit_scaler(features, fit):
    # Keep the original eligible-row order and exact percentile/reduction shapes.
    selected = features[fit]
    lo, hi = np.percentile(selected, [1, 99], axis=0)
    clipped = np.asarray(np.clip(selected, lo, hi), dtype=np.float64, order='C')
    del selected
    mean = clipped.mean(0)
    # NumPy's original std computes this same ordered squared-deviation array.
    # Reuse its private buffer rather than allocating another complete matrix.
    np.subtract(clipped, mean, out=clipped)
    np.multiply(clipped, clipped, out=clipped)
    variance = clipped.mean(0)
    return dict(lo=lo, hi=hi, mean=mean,
                std=np.maximum(np.sqrt(variance), 1e-6))


def apply_scaler(features, scaler, block_rows=8192, output=None):
    if not isinstance(block_rows, int) or block_rows <= 0:
        raise ValueError('Positive transform block size required')
    out = np.empty(features.shape, dtype=np.float32) if output is None else output
    if out.shape != features.shape or out.dtype != np.float32 or not out.flags.c_contiguous:
        raise ValueError('Exact complete contiguous float32 output required')
    for lo in range(0, len(features), block_rows):
        hi = min(lo + block_rows, len(features))
        out[lo:hi] = np.asarray((np.clip(features[lo:hi], scaler['lo'], scaler['hi'])-scaler['mean'])/
                               scaler['std'], dtype=np.float32, order='C')
    return out


def scaled(features, fit, scaler=None):
    scaler = fit_scaler(features, fit) if scaler is None else scaler
    return apply_scaler(features, scaler), scaler
