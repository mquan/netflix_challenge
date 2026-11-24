#!/usr/bin/env python3
"""Own17x (X) and native bank-v2 (N) refits with CF-NADE residual columns (plan amendment).
usage: rcb_fullbank_cf.py OUT [ARM ...]   (default: all arms whose inputs exist)
Global ridge lambda 1000, 5 user folds (seed 20261116); final fit on all known rows -> frozen qual."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import rcb_blend as b
ROOT = b.ROOT; A = ROOT / 'artifacts'; P = A / 'rcb-r1/source/preds_new'; E = ROOT / 'experiments/remote-context-blend'
CH = ['knn3', 'knn3d', 'knn-25c', 'knn-25m']
G = {'cf48ch': [f'cfnade-48__{c}' for c in CH], 'cf96': ['cfnade-96'], 'cf96ch': [f'cfnade-96__{c}' for c in CH]}
ARMS = {'0': [], '1': ['cf48ch'], '2': ['cf96'], '3': ['cf96', 'cf96ch'], '4': ['cf96', 'cf96ch', 'cf48ch']}
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); want = sys.argv[2:]
y = np.load(A / 'bank-v2/known_y.npy'); users = np.load(A / 'bank-v2/known_users.npy'); folds = b.splitmix_folds(users, 20261116, 5)
def load_bank(name):
    meta = json.loads((E / f'bank-{name}.json').read_text())
    return [c['column'] for c in meta['columns']], np.load(A / f'bank-{name}/known.npy'), np.load(A / f'bank-{name}/qual.npy'), meta['known_sha256']
summary = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
for tag, bank in [('X', 'own17x'), ('N', 'v2')]:
    names, K0, Q0, bsha = load_bank(bank)
    for k, groups in ARMS.items():
        arm = tag + k
        if want and arm not in want: continue
        if arm in summary: print('kept', arm); continue
        cols = [c for g in groups for c in G[g]]
        if not all((P / f'{c}.probex.npy').exists() and (P / f'{c}.qual.npy').exists() for c in cols): print('skip (inputs missing)', arm); continue
        ek = [np.load(P / f'{c}.probex.npy') for c in cols]; eq = [np.load(P / f'{c}.qual.npy') for c in cols]
        dups = [c for c, a, q in zip(cols, ek, eq) for j in range(K0.shape[1]) if np.array_equal(a, K0[:, j]) and np.array_equal(q, Q0[:, j])]
        keep = [i for i, c in enumerate(cols) if c not in dups]
        Xk = np.column_stack([K0] + [ek[i] for i in keep]) if keep else K0; Xq = np.column_stack([Q0] + [eq[i] for i in keep]) if keep else Q0
        allc = list(range(Xk.shape[1])); held = np.full(len(y), np.nan); fr = []
        for f in range(5):
            tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            d, w = b.fit_global(Xk, y, allc, tr); held[te] = d.predict(te, w); fr.append(b.rmse(held[te], y[te]))
        d, w = b.fit_global(Xk, y, allc, np.arange(len(y)))
        q = b.Design(Xq, None, None, allc).predict(np.arange(Xq.shape[0]), w).astype('<f4')
        q.tofile(out / f'qual-{arm}.f32'); held.astype('<f4').tofile(out / f'known-{arm}.f32'); np.save(out / f'coef-{arm}.npy', w)
        summary[arm] = dict(bank=bank, bank_known_sha256=bsha, added=[cols[i] for i in keep], duplicates_dropped=dups,
                            n_columns=len(allc), pooled=b.rmse(held, y), folds=fr, qual_sha256=b.sha_arr(q),
                            known_sha256=b.sha_arr(held.astype('<f4')), coef_sha256=b.sha_arr(w),
                            added_sha256={c: hashlib.sha256(open(P / f'{c}.probex.npy', 'rb').read()).hexdigest() for c in cols})
        print(arm, len(allc), round(summary[arm]['pooled'], 6), [round(v, 6) for v in fr], summary[arm]['qual_sha256'][:12], flush=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
