#!/usr/bin/env python3
"""Lane 2 comparison (plan amendment 'Lane 2'): TX300 single-split control B vs
second-split arm C on probex known rows, 5 original user folds, global ridge
lambda=1000 fitted on outer-training rows only. Bank v2 already holds A
(s3-tx300r), so (ii) uses bank v2 without A and (iii) uses the full bank v2.
Known side only; never reads qualifying labels."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('RCB_BANK_DIR', 'artifacts/bank-v2')
import rcb_blend as b

ROOT = Path(__file__).resolve().parents[1]
L2 = ROOT / 'artifacts/rcb-l2'
OUT = ROOT / 'experiments/remote-context-blend/lane2'
NP3 = 1760494

def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def proj(vec_path, map_path):
    v = np.load(vec_path); m = np.load(map_path)
    assert v.shape == (NP3,) and v.dtype == np.float32 and np.isfinite(v).all(), vec_path
    return v[m]

def main():
    bank = json.loads((ROOT / 'experiments/remote-context-blend/bank-v2.json').read_text())
    names = [c['column'] for c in bank['columns']]
    X0 = np.load(ROOT / 'artifacts/bank-v2/known.npy'); y = np.load(ROOT / 'artifacts/bank-v2/known_y.npy')
    users = np.load(ROOT / 'artifacts/bank-v2/known_users.npy'); folds = b.splitmix_folds(users, 20261116, 5)
    src = dict(B=(L2 / 'source/preds_3b/s3-tx300r.probe_3.npy', ROOT / 'artifacts/rcb-r1/probex-row-map.npy'),
               C=(L2 / 'source/preds_3s/s3-tx300r.probe_3s20261124.npy', L2 / 'probex-row-map-3s.npy'))
    new = {k: proj(*v) for k, v in src.items()}
    a = names.index('s3-tx300r')
    X = np.column_stack([X0, new['B'], new['C']]).astype(np.float32)
    iB, iC = X.shape[1] - 2, X.shape[1] - 1
    bank_noA = [j for j in range(len(names)) if j != a]; bank_all = list(range(len(names)))
    arms = {'A': [a], 'B': [iB], 'C': [iC],
            'bank-A': bank_noA, 'bank-A+B': bank_noA + [iB], 'bank-A+C': bank_noA + [iC],
            'bank': bank_all, 'bank+B': bank_all + [iB], 'bank+C': bank_all + [iC], 'bank+B+C': bank_all + [iB, iC]}
    held = {k: np.full(len(y), np.nan) for k in arms}; per = {k: [] for k in arms}
    for f in range(5):
        tr = np.flatnonzero(folds != f); te = np.flatnonzero(folds == f)
        for k, cols in arms.items():
            d, w = b.fit_global(X, y, cols, tr); held[k][te] = d.predict(te, w); per[k].append(b.rmse(held[k][te], y[te]))
        print(f, {k: round(v[-1], 6) for k, v in per.items()}, flush=True)
    pooled = {k: b.rmse(v, y) for k, v in held.items()}
    raw = {k: float(np.sqrt(np.mean((np.clip(new[k], 1.05, 4.95) - y) ** 2))) for k in 'BC'}
    raw['A'] = float(np.sqrt(np.mean((np.clip(X0[:, a], 1.05, 4.95) - y) ** 2)))
    def cmp(x, z):  # z - x, negative = z better
        d = [per[z][f] - per[x][f] for f in range(5)]
        return dict(delta=pooled[z] - pooled[x], fold_deltas=d, folds_better=int(sum(v < 0 for v in d)))
    s = dict(rows=len(y), raw_rmse=raw, pooled=pooled, per_fold=per,
             comparisons={'(i) C vs B standalone ridge': cmp('B', 'C'),
                          '(ii) bank-A+C vs bank-A+B': cmp('bank-A+B', 'bank-A+C'),
                          '(iii) bank+C vs bank+B': cmp('bank+B', 'bank+C'),
                          'bank-A+B vs bank-A': cmp('bank-A', 'bank-A+B'), 'bank-A+C vs bank-A': cmp('bank-A', 'bank-A+C'),
                          'bank+B vs bank': cmp('bank', 'bank+B'), 'bank+C vs bank': cmp('bank', 'bank+C'),
                          'bank+B+C vs bank+B': cmp('bank+B', 'bank+B+C')},
             corr=dict(AB=float(np.corrcoef(X0[:, a], new['B'])[0, 1]), AC=float(np.corrcoef(X0[:, a], new['C'])[0, 1]),
                       BC=float(np.corrcoef(new['B'], new['C'])[0, 1])),
             inputs={k: dict(vector=str(v[0]), vector_sha256=sha(v[0]), map=str(v[1]), map_sha256=sha(v[1])) for k, v in src.items()},
             bank_known_sha256=bank['known_sha256'], qualifying_labels_read=False,
             note='bank v2 contains A; (ii) drops A, (iii) keeps it. Known side only; C qual = B qual by declaration.')
    (OUT / 'summary.json').write_text(json.dumps(s, indent=1) + '\n')
    print(json.dumps(dict(raw=raw, comparisons=s['comparisons'], corr=s['corr']), indent=1))

if __name__ == '__main__':
    main()
