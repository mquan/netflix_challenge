#!/usr/bin/env python3
"""Verify a fresh split-new package and measure it against the fixed native bank v2.

usage: rcb_pkg_eval.py PREDS_DIR OUT_DIR GROUP=col1,col2 [GROUP=...] [--extra NAME=G1+G3 ...]

Each column needs PREDS_DIR/<col>.probex.npy (1,128,164 f32) and <col>.qual.npy (2,817,131 f32),
native split-new row order (same files as bank v2's split-new columns). Arms are cumulative in
GROUP order: bank, bank+G1, bank+G1+G2, ... plus optional --extra combinations. Global ridge
lambda=1000, 5 original user folds (seed 20261116), outer-training rows only. Missing groups are
skipped and listed. Never reads qualifying labels (qual targets are zero in these trees).
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('RCB_BANK_DIR', 'artifacts/bank-v2')
import rcb_blend as b

ROOT = Path(__file__).resolve().parents[1]
NK, NQ = 1128164, 2817131

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(8 << 20), b''): h.update(c)
    return h.hexdigest()

def check(p, n):
    v = np.load(p)
    ok = v.shape == (n,) and v.dtype == np.float32 and bool(np.isfinite(v).all())
    return v, dict(path=str(p), sha256=sha(p), shape=list(v.shape), dtype=str(v.dtype), finite=bool(np.isfinite(v).all()),
                   ok=ok, mtime_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(p.stat().st_mtime)),
                   min=float(v.min()), max=float(v.max()))

def main():
    args = sys.argv[1:]; preds = Path(args[0]); out = Path(args[1]); out.mkdir(parents=True, exist_ok=True)
    groups, extras = [], {}
    i = 2
    while i < len(args):
        if args[i] == '--extra': k, v = args[i + 1].split('='); extras[k] = v.split('+'); i += 2
        else: k, v = args[i].split('='); groups.append((k, v.split(','))); i += 1
    X0 = np.load(ROOT / 'artifacts/bank-v2/known.npy'); y = np.load(ROOT / 'artifacts/bank-v2/known_y.npy')
    users = np.load(ROOT / 'artifacts/bank-v2/known_users.npy'); folds = b.splitmix_folds(users, 20261116, 5)
    nb = X0.shape[1]; cols, manifest, idx, missing = [], {}, {}, []
    for g, names in groups:
        have = all((preds / f'{c}.probex.npy').exists() and (preds / f'{c}.qual.npy').exists() for c in names)
        if not have: missing.append(g); continue
        idx[g] = []
        for c in names:
            vk, mk = check(preds / f'{c}.probex.npy', NK); vq, mq = check(preds / f'{c}.qual.npy', NQ)
            if not (mk['ok'] and mq['ok']): raise SystemExit(f'bad vector {c}: {mk} {mq}')
            manifest[c] = dict(group=g, probex=mk, qual=mq, probex_rmse_raw=float(np.sqrt(np.mean((np.clip(vk, 1.05, 4.95) - y) ** 2))))
            idx[g].append(nb + len(cols)); cols.append(vk)
    X = np.column_stack([X0] + cols).astype(np.float32) if cols else X0
    base = list(range(nb)); arms = {'bank': base}; acc = list(base); order = []
    for g, _ in groups:
        if g in idx: acc = acc + idx[g]; order.append(g); arms['bank+' + '+'.join(order)] = list(acc)
    for k, gs in extras.items():
        if all(g in idx for g in gs): arms[k] = base + sum((idx[g] for g in gs), [])
    held = {k: np.full(len(y), np.nan) for k in arms}; per = {k: [] for k in arms}
    for f in range(5):
        tr = np.flatnonzero(folds != f); te = np.flatnonzero(folds == f)
        for k, c in arms.items():
            d, w = b.fit_global(X, y, c, tr); held[k][te] = d.predict(te, w); per[k].append(b.rmse(held[k][te], y[te]))
        print(f, {k: round(v[-1], 6) for k, v in per.items()}, flush=True)
    pooled = {k: b.rmse(v, y) for k, v in held.items()}
    def cmp(x, z):
        d = [per[z][f] - per[x][f] for f in range(5)]
        return dict(delta=pooled[z] - pooled[x], fold_deltas=d, folds_better=int(sum(v < 0 for v in d)))
    names = list(arms); steps = {f'{names[j]} vs {names[j-1]}': cmp(names[j - 1], names[j]) for j in range(1, len(names)) if not names[j] in extras}
    vs_bank = {f'{k} vs bank': cmp('bank', k) for k in names[1:]}
    s = dict(preds_dir=str(preds), groups=dict(groups), missing_groups=missing, columns=manifest, pooled=pooled, per_fold=per,
             steps=steps, vs_bank=vs_bank, bank='bank v2 (65 native columns), known probex rows', rows=len(y),
             protocol='global ridge lambda=1000, unpenalized intercept, 5 user folds seed 20261116, outer-train fit only',
             qualifying_labels_read=False, generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (out / 'summary.json').write_text(json.dumps(s, indent=1) + '\n')
    lines = ['| arm | pooled | Δ vs bank | folds better |', '|---|---|---|---|']
    for k in names:
        c = vs_bank.get(f'{k} vs bank'); lines.append(f"| {k} | {pooled[k]:.6f} | {'' if c is None else f'{c["delta"]:+.6f}'} | {'' if c is None else str(c['folds_better'])+'/5'} |")
    (out / 'table.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))

if __name__ == '__main__':
    main()
