#!/usr/bin/env python3
"""Rank retained predictors by cheap user-fold development blend screening.

The reference is already out of fold, so this second-level screen is adaptive
development analysis, not an independent generalization estimate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load, blend_fold
from streaming_blend import digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--bank', required=True)
    ap.add_argument('--source-report', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    d = load(a.data)
    y = d['probe']['rating'].astype(np.float64)
    fold = blend_fold(d['probe']['user'])
    reference = np.fromfile(a.reference, '<f4').astype(np.float64)
    if reference.shape != y.shape or not np.isfinite(reference).all():
        raise ValueError('invalid reference')
    b = json.loads(Path(a.bank + '.inputs.json').read_text())
    source = json.loads(Path(a.source_report).read_text())
    by_hash = {}
    for name, sha in source['inputs'].items():
        by_hash.setdefault(sha, []).append(name)
    results = []
    for name in b['names']:
        if '/bank/columns/' not in name:
            continue
        sha = b['inputs'][name]
        if digest(name) != sha:
            raise ValueError('retained column changed')
        candidate = np.fromfile(name, '<f4').astype(np.float64)
        delta = candidate - reference
        numerator = np.bincount(fold, (y-reference)*delta, minlength=5)
        denominator = np.bincount(fold, delta*delta, minlength=5)
        weights = np.clip((numerator.sum()-numerator) /
                          np.maximum(denominator.sum()-denominator, 1e-15), -.5, 1.5)
        pred = np.clip(reference + weights[fold]*delta, 1, 5)
        results.append(dict(path=name, sha256=sha, original_paths=by_hash.get(sha, []),
                            blend_rmse=float(np.sqrt(np.mean((pred-y)**2))),
                            candidate_rmse=float(np.sqrt(np.mean((np.clip(candidate,1,5)-y)**2))),
                            weights=weights.tolist()))
    results.sort(key=lambda row: row['blend_rmse'])
    report = dict(metric='Adaptive original-Probe screening only; no independent Test claim',
                  reference_sha256=digest(a.reference), data_sha256=digest(a.data),
                  reference_rmse=float(np.sqrt(np.mean((np.clip(reference,1,5)-y)**2))),
                  qualifying_labels_read=False, results=results)
    Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(reference_rmse=report['reference_rmse'], best=results[:15])), flush=True)


if __name__ == '__main__':
    main()
