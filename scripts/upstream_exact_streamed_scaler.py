#!/usr/bin/env python3
"""Original C-order scaler reductions with complete column percentiles and bounded rows."""
import numpy as np


def fit_scaler(features, eligible, block_rows=8192):
    if (features.ndim != 2 or features.dtype != np.float32 or not features.flags.c_contiguous
            or features.shape[1] != 579 or eligible.shape != (len(features),)
            or eligible.dtype != np.bool_ or not eligible.any()
            or not isinstance(block_rows, int) or block_rows <= 0):
        raise ValueError('Complete original579-column contiguous float32 population and eligible mask required')
    width = features.shape[1]
    lo, hi = np.empty(width), np.empty(width)
    # Same complete eligible values, original float32 dtype and percentile
    # interpolation. Partition each full column instead of a complete matrix.
    for column in range(width):
        lo[column], hi[column] = np.percentile(features[eligible, column], [1, 99])
    count = int(eligible.sum())
    def total(mean=None):
        carry = np.zeros(width, dtype=np.float64)
        for start in range(0, len(features), block_rows):
            stop = min(start+block_rows, len(features))
            selected = features[start:stop][eligible[start:stop]]
            if not len(selected):
                continue
            values = np.clip(selected, lo, hi)
            if values.dtype != np.float64 or not values.flags.c_contiguous:
                raise ValueError('Original full C-order clipped float64 arithmetic required')
            if mean is not None:
                np.subtract(values, mean, out=values)
                np.multiply(values, values, out=values)
            # Original axis0 C-order reduce adds rows in order. Carry into the
            # first row before accumulating, avoiding different block sums.
            np.add(carry, values[0], out=values[0])
            np.add.accumulate(values, axis=0, out=values)
            carry = values[-1].copy()
        return carry
    mean = total()/count
    variance = total(mean)/count
    return dict(lo=lo, hi=hi, mean=mean, std=np.maximum(np.sqrt(variance), 1e-6))
