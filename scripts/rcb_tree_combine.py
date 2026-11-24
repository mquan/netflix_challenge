#!/usr/bin/env python3
"""Combine tree units: prediction = clip(base + mean of 3 seed-slot corrections). Scores all
five held folds against the same-bank global and context v2 known vectors on identical rows,
and writes the frozen qual vector from the 'final' units."""
import argparse, json
from pathlib import Path
import numpy as np
import rcb_blend as b

ap = argparse.ArgumentParser(); ap.add_argument('--bank', type=Path, required=True); ap.add_argument('--units', type=Path, required=True)
ap.add_argument('--blend', type=Path, required=True, help='dir with known-global.f32 and known-context.f32 for the same bank')
ap.add_argument('--out', type=Path, required=True); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
y = np.load(a.bank / 'known_y.npy'); u = np.load(a.bank / 'known_users.npy'); folds = b.splitmix_folds(u, 20261116, 5)
tree = np.full(len(y), np.nan); recs = []
for f in range(5):
    te = np.flatnonzero(folds == f); base = np.load(a.units / f'{f}.base.npy')
    corr = np.mean([np.load(a.units / f'{f}-s{s}.corr.npy') for s in range(3)], axis=0)
    tree[te] = np.clip(base + corr, 1.05, 4.95)
    recs.append([json.loads((a.units / f'{f}-s{s}.json').read_text()) for s in range(3)])
g = np.fromfile(a.blend / 'known-global.f32', '<f4'); c = np.fromfile(a.blend / 'known-context.f32', '<f4')
fr = lambda p: [b.rmse(p[folds == f], y[folds == f]) for f in range(5)]
res = dict(pooled=dict(tree=b.rmse(tree, y), glob=b.rmse(g, y), context=b.rmse(c, y)),
           folds=dict(tree=fr(tree), glob=fr(g), context=fr(c)), units=recs)
gain = [cc - t for cc, t in zip(res['folds']['context'], res['folds']['tree'])]
res['tree_vs_context_gain'] = res['pooled']['context'] - res['pooled']['tree']; res['fold_gains_vs_context'] = gain
res['tree_replaces_primary'] = bool(res['tree_vs_context_gain'] >= 0.001 and sum(x > 0 for x in gain) >= 4)
tree.astype('<f4').tofile(a.out / 'known-tree.f32'); res['known_tree_sha256'] = b.sha_arr(tree.astype('<f4'))
if all((a.units / f'final-s{s}.corr.npy').exists() for s in range(3)):
    q = np.clip(np.load(a.units / 'final.base.npy') + np.mean([np.load(a.units / f'final-s{s}.corr.npy') for s in range(3)], axis=0), 1.05, 4.95)
    q.astype('<f4').tofile(a.out / 'qual-tree.f32'); res['qual_tree_sha256'] = b.sha_arr(q.astype('<f4')); res['qual_rows'] = len(q)
(a.out / 'summary.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps({k: res[k] for k in ('pooled', 'tree_vs_context_gain', 'tree_replaces_primary')}))
