#!/usr/bin/env python3
"""Structural plan week 0: scoping the promoted bank (STRUCTURAL_PLAN.md).

Uses the 79 prediction columns of R9-LAT-A2h on the original probe and a
ridge proxy blender with 5 user folds (cross-fitted). Support, metadata and
latent inputs are excluded. The proxy is weaker than the production
LightGBM ⊕ NN2 blender; it is used for the *shape* of the curves. This reads
probe labels for diagnosis only; no column or setting is selected for the bank.

(i)  Diversity curve: blend RMSE vs number of random columns k, and vs number
     of whole families included (random family orders). A power law
     a + b * k**-c is fitted to the column curve and extrapolated.
(ii) Quality -> blend conversion: for each family with at least 3 members,
     the bank without that family plus exactly one member; blend RMSE is
     regressed on the member's standalone RMSE across members (the slope is
     blend gain per unit of standalone gain, when that family's representative
     improves).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from gbdt_blend import blend_fold, load

ROOT = Path('/Users/quan/workspace/netflix_round9')
SPEC = ROOT / 'artifacts/r9/R9-LAT-A2h-20261010-NN2.json'
FAMILIES = [
    ('a2_transformer', r'masked/A2'), ('masked_history', r'masked/(MH|MT)'), ('bayes', r'r8/B\d'),
    ('rbm', r'rbm'), ('autorec', r'autorec'), ('gp_local', r'(gp|local-)'),
    ('integrated', r'(I2|neighborhood64)'), ('knn', r'knn|uknn'),
    ('factor_sgd', r'(joint|/T\d|^T\d|T\d-|T6)'),
]
PENALTY = 1.
SEARCH = [ROOT, Path('/Users/quan/workspace/netflix_challenge'),
          Path('/Users/quan/workspace/netflix_challenge/artifacts/masked/reference')]


def resolve(name, sha):
    """Locate a bank input by its recorded SHA-256 (the spec's paths span several trees)."""
    from train_holdout import digest
    for root in SEARCH:
        path = root / name
        if path.exists() and digest(path) == sha:
            return path
    raise FileNotFoundError(name)


def family(name):
    for fam, pattern in FAMILIES:
        if re.search(pattern, name):
            return fam
    return 'other'


class Blender:
    """Cross-fitted ridge on any column subset from cached per-fold Gram matrices."""

    def __init__(self, x, y, folds):
        self.y, self.folds = y, folds
        self.x = np.column_stack([np.ones(len(y)), x])
        self.g = [self.x[folds == f].T @ self.x[folds == f] for f in range(5)]
        self.r = [self.x[folds == f].T @ y[folds == f] for f in range(5)]
        self.G, self.R = sum(self.g), sum(self.r)

    def rmse(self, cols):
        idx = np.concatenate([[0], np.asarray(cols, dtype=int) + 1])
        sse = 0.
        for f in range(5):
            g = (self.G - self.g[f])[np.ix_(idx, idx)]
            r = (self.R - self.r[f])[idx]
            w = np.linalg.solve(g + PENALTY * np.eye(len(idx)), r)
            pred = np.clip(self.x[self.folds == f][:, idx] @ w, 1, 5)
            sse += float(np.sum((pred - self.y[self.folds == f]) ** 2))
        return float(np.sqrt(sse / len(self.y)))


def run(output, draws=24, seed=20261031):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(SPEC.read_text())
    names = [n for n in spec['inputs'] if not re.search(r'support|count|meta-|latent/', n)]
    d = load('data/processed/netflix.bin')
    y, users = d['probe']['rating'].astype(float), d['probe']['user']
    paths = [resolve(n, spec['inputs'][n]) for n in names]
    x = np.column_stack([np.fromfile(p, dtype='<f4').astype(float) for p in paths])
    if x.shape[0] != len(y) or not np.isfinite(x).all():
        raise ValueError('column length mismatch')
    fams = [family(n) for n in names]
    standalone = [float(np.sqrt(np.mean((np.clip(x[:, j], 1, 5) - y) ** 2))) for j in range(len(names))]
    folds = blend_fold(users)
    b = Blender(x, y, folds)
    full = b.rmse(range(len(names)))
    rng = np.random.default_rng(seed)
    ks = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, len(names)]
    curve = {}
    for k in ks:
        vals = [b.rmse(rng.choice(len(names), k, replace=False)) for _ in range(1 if k == len(names) else draws)]
        curve[k] = dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))
        print(f'k={k:3d} mean={curve[k]["mean"]:.5f} min={curve[k]["min"]:.5f}', flush=True)
    # power-law fit on mean curve for k >= 4 (grid over c, linear least squares for a, b)
    kk = np.array([k for k in ks if k >= 4], float)
    vv = np.array([curve[int(k)]['mean'] for k in kk])
    best = None
    for c in np.linspace(.05, 2, 400):
        A = np.column_stack([np.ones(len(kk)), kk ** -c])
        coef, res, *_ = np.linalg.lstsq(A, vv, rcond=None)
        err = float(np.sum((A @ coef - vv) ** 2))
        if best is None or err < best[0]:
            best = (err, c, coef)
    _, c, (a, bb) = best
    extrapolate = {int(k): float(a + bb * k ** -c) for k in (79, 160, 320, 800)}
    # family curve
    fam_list = sorted(set(fams))
    fam_cols = {f: [j for j, g in enumerate(fams) if g == f] for f in fam_list}
    fam_curve = {}
    for m in range(1, len(fam_list) + 1):
        vals = []
        for _ in range(draws):
            chosen = rng.choice(fam_list, m, replace=False)
            vals.append(b.rmse([j for f in chosen for j in fam_cols[f]]))
        fam_curve[m] = float(np.mean(vals))
    # leave-one-family-out value
    lofo = {f: b.rmse([j for j in range(len(names)) if fams[j] != f]) - full for f in fam_list}
    # quality -> blend conversion
    conversion = {}
    for f in fam_list:
        members = fam_cols[f]
        if len(members) < 3:
            continue
        rest = [j for j in range(len(names)) if fams[j] != f]
        pts = [(standalone[m], b.rmse(rest + [m]), names[m]) for m in members]
        s = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
        slope = float(np.polyfit(s, v, 1)[0]) if np.ptp(s) > 1e-4 else None
        conversion[f] = dict(slope_blend_per_standalone=slope, standalone_range=float(np.ptp(s)),
                             blend_range=float(np.ptp(v)), members=[dict(name=n, standalone=a_, blend=b_)
                                                                    for a_, b_, n in pts])
    report = dict(columns=names, resolved_paths=[str(p) for p in paths], families=dict(zip(names, fams)), standalone=dict(zip(names, standalone)),
                  ridge_full_cv_rmse=full, production_reference=0.8668706425, column_curve=curve,
                  power_law=dict(a=float(a), b=float(bb), c=float(c)), extrapolated_mean_rmse=extrapolate,
                  family_curve=fam_curve, leave_one_family_out=lofo, quality_to_blend=conversion,
                  metric='Ridge proxy (5 user folds, cross-fitted) on the original probe; diagnostic only.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(ridge_full=full, power_law=report['power_law'], extrapolated=extrapolate,
                          family_curve=fam_curve, lofo={k: round(v, 6) for k, v in lofo.items()},
                          slopes={k: (v['slope_blend_per_standalone'], round(v['standalone_range'], 4))
                                  for k, v in conversion.items()}), indent=1))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    run(ap.parse_args().output)


if __name__ == '__main__':
    main()
