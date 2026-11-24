#!/usr/bin/env python3
"""Lane 1 comparison: global ridge (lambda 1000) on bank v2 +/- CF48 residual chains, 5 user folds.
All columns are native held-out (trainx->probex or train_3->probe_3). Writes held-out known vectors,
final all-known qual vectors per arm and summary. Never reads qualifying labels."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import rcb_blend as b
ROOT = b.ROOT; A = ROOT / 'artifacts'; P = A / 'rcb-r1/source/preds_new'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
bank = json.loads((ROOT / 'experiments/remote-context-blend/bank-v2.json').read_text())
names = [c['column'] for c in bank['columns']]
Xk = np.load(A / 'bank-v2/known.npy'); Xq = np.load(A / 'bank-v2/qual.npy', mmap_mode='r')
y = np.load(A / 'bank-v2/known_y.npy'); users = np.load(A / 'bank-v2/known_users.npy')
CH = ['knn3', 'knn3d', 'knn-25c', 'knn-25m']
extra = {c: (np.load(P / f'cfnade-48__{c}.probex.npy'), np.load(P / f'cfnade-48__{c}.qual.npy')) for c in CH}
Xk = np.column_stack([Xk] + [extra[c][0] for c in CH]); Xq = np.column_stack([np.asarray(Xq)] + [extra[c][1] for c in CH])
cf = names.index('cfnade-48'); K = len(names); idx = {c: K + j for j, c in enumerate(CH)}
base = [j for j in range(K) if j != cf]
arms = {'no-cf48': base, 'base': base + [cf]}
for c in CH: arms[f'base+{c}'] = base + [cf, idx[c]]
arms['base+all'] = base + [cf] + [idx[c] for c in CH]
folds = b.splitmix_folds(users, 20261116, 5); held = {a: np.full(len(y), np.nan) for a in arms}; rec = []
for f in range(5):
    tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f); r = dict(fold=f)
    for a, cols in arms.items():
        d, w = b.fit_global(Xk, y, cols, tr); held[a][te] = d.predict(te, w); r[a] = b.rmse(held[a][te], y[te])
    rec.append(r); print(json.dumps(r), flush=True)
pooled = {a: b.rmse(v, y) for a, v in held.items()}
standalone = {'cfnade-48': b.rmse(Xk[:, cf], y), **{f'cfnade-48__{c}': b.rmse(Xk[:, idx[c]], y) for c in CH}}
final = {}
for a, cols in arms.items():
    d, w = b.fit_global(Xk, y, cols, np.arange(len(y)))
    q = b.Design(Xq, None, None, cols).predict(np.arange(Xq.shape[0]), w).astype('<f4')
    q.tofile(out / f'qual-{a}.f32'); held[a].astype('<f4').tofile(out / f'known-{a}.f32'); np.save(out / f'coef-{a}.npy', w)
    final[a] = dict(qual_sha256=b.sha_arr(q), known_sha256=b.sha_arr(held[a].astype('<f4')), coef_sha256=b.sha_arr(w))
s = dict(bank='bank-v2 (65 cols) + 4 cfnade-48 chains', ridge_lambda=1000, folds=rec, pooled=pooled,
         delta_vs_base={a: pooled[a] - pooled['base'] for a in arms},
         folds_better_than_base={a: int(sum(r[a] < r['base'] for r in rec)) for a in arms},
         standalone_probex=standalone, final=final, qualifying_labels_read=False)
(out / 'summary.json').write_text(json.dumps(s, indent=1) + '\n'); print(json.dumps({k: s[k] for k in ('pooled', 'delta_vs_base', 'folds_better_than_base', 'standalone_probex')}, indent=1))
