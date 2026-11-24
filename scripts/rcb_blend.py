#!/usr/bin/env python3
"""Global vs context-dependent ridge blend on the frozen bank (plan §3).

Known rows only; five original user folds (upstream_exchange.user_folds, seed
20261116). Context bin cutoffs are quantiles of the outer-training rows' context
values; inner lambda_ctx selection uses a 4-fold user split of the outer-training
users (splitmix seed 20261116+1+outer) with the same cutoffs. Writes fold
receipts, held-out known vectors and (with --final) the frozen qualifying
predictions of both candidates. Never reads qualifying labels.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / os.environ.get('RCB_BANK_DIR', 'artifacts/bank')
BANK_JSON = os.environ.get('RCB_BANK_JSON', 'bank.json')
GLOBAL_LAMBDA = 1000.0
import os
CTX_GRID = [float(v) for v in os.environ.get("RCB_CTX_GRID", "1e4,3e4,1e5,3e5,1e6").split(",")]
CTX_NAMES = ['log_user_support', 'log_item_support', 'days_since_last_train']
BINS = 4
CHUNK = 65536


def splitmix_folds(users, seed, k):
    x = np.asarray(users, dtype=np.uint64) + np.uint64(seed)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return (x % np.uint64(k)).astype(np.int8)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.clip(p, 1.05, 4.95) - y.astype(np.float64)) ** 2)))


class Design:
    """Column block X (clipped inputs + intercept) and optional context interactions."""

    def __init__(self, X, ctx, cuts, cols):
        self.X, self.ctx, self.cuts, self.cols = X, ctx, cuts, cols
        self.k = len(cols) + 1
        self.width = self.k * (1 + (len(CTX_NAMES) * BINS if cuts is not None else 0))

    def rows(self, idx):
        x = np.clip(np.asarray(self.X[idx][:, self.cols], dtype=np.float64), 0, 6)
        x = np.column_stack([x, np.ones(len(idx))])
        if self.cuts is None: return x
        blocks = [x]
        for c, name in enumerate(CTX_NAMES):
            b = np.searchsorted(self.cuts[c], self.ctx[name][idx], side='right')
            for j in range(BINS): blocks.append(x * (b == j)[:, None])
        return np.hstack(blocks)

    def gram(self, idx, y):
        G = np.zeros((self.width, self.width)); r = np.zeros(self.width)
        for s in range(0, len(idx), CHUNK):
            z = self.rows(idx[s:s + CHUNK]); G += z.T @ z; r += z.T @ y[idx[s:s + CHUNK]].astype(np.float64)
        return G, r

    def penalty(self, lam_ctx):
        p = np.zeros(self.width); p[:self.k] = GLOBAL_LAMBDA; p[self.k - 1] = 0.0  # unpenalized intercept
        p[self.k:] = lam_ctx
        return np.diag(p)

    def predict(self, idx, w):
        out = np.empty(len(idx))
        for s in range(0, len(idx), CHUNK): out[s:s + CHUNK] = self.rows(idx[s:s + CHUNK]) @ w
        return np.clip(out, 1.05, 4.95)


def cutoffs(ctx, idx):
    return [np.quantile(ctx[n][idx], [0.25, 0.5, 0.75]) for n in CTX_NAMES]


def fit_context(X, y, users, ctx, cols, train, inner_seed):
    """Choose lambda_ctx by inner 4-fold user CV inside `train`, then refit on all of `train`."""
    d = Design(X, ctx, cutoffs(ctx, train), cols)
    inner = splitmix_folds(users[train], inner_seed, 4)
    parts = [d.gram(train[inner == f], y) for f in range(4)]
    scores = {}
    for lam in CTX_GRID:
        se = 0.0
        for f in range(4):
            G = sum(p[0] for g, p in enumerate(parts) if g != f); r = sum(p[1] for g, p in enumerate(parts) if g != f)
            w = np.linalg.solve(G + d.penalty(lam), r); held = train[inner == f]
            se += float(np.sum((d.predict(held, w) - y[held]) ** 2))
        scores[lam] = float(np.sqrt(se / len(train)))
    best = min(scores.values())
    lam = max(l for l, s in scores.items() if s == best)  # tie -> larger lambda
    G = sum(p[0] for p in parts); r = sum(p[1] for p in parts)
    return d, np.linalg.solve(G + d.penalty(lam), r), lam, scores


def fit_global(X, y, cols, train):
    d = Design(X, None, None, cols); G, r = d.gram(train, y)
    return d, np.linalg.solve(G + d.penalty(0.0), r)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'experiments/remote-context-blend/blend')
    ap.add_argument('--final', action='store_true', help='also fit on all known rows and write qual predictions')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    bank = json.loads((ROOT / 'experiments/remote-context-blend' / BANK_JSON).read_text())
    names = [c['column'] for c in bank['columns']]
    X = np.load(BANK / 'known.npy', mmap_mode='r'); y = np.load(BANK / 'known_y.npy'); users = np.load(BANK / 'known_users.npy')
    ctx = dict(np.load(BANK / 'known_ctx.npz'))
    folds = splitmix_folds(users, 20261116, 5)
    new = [names.index(c) for c in ('s3-tx300r__knn3', 's3-tx300r__knn3d') if c in names]
    allc = list(range(len(names)))
    arms = {'global': allc}
    if len(new) == 2:
        arms.update({'global-no-new': [j for j in allc if j not in new],
                     'global-plus-knn3': [j for j in allc if j != new[1]],
                     'global-plus-knn3d': [j for j in allc if j != new[0]]})
    held = {k: np.full(len(y), np.nan) for k in list(arms) + ['context']}
    receipts = []
    for f in range(5):
        tr = np.flatnonzero(folds != f); te = np.flatnonzero(folds == f)
        rec = dict(fold=f, train_rows=len(tr), held_rows=len(te), held_users=int(len(np.unique(users[te]))))
        for arm, cols in arms.items():
            d, w = fit_global(X, y, cols, tr); held[arm][te] = d.predict(te, w)
            rec[arm] = dict(rmse=rmse(held[arm][te], y[te]))
        d, w, lam, sc = fit_context(X, y, users, ctx, allc, tr, 20261116 + 1 + f)
        held['context'][te] = d.predict(te, w)
        rec['context'] = dict(rmse=rmse(held['context'][te], y[te]), lambda_ctx=lam, inner_scores=sc,
                              cutoffs=[c.tolist() for c in d.cuts])
        receipts.append(rec); print(json.dumps(rec), flush=True)
    pooled = {k: rmse(v, y) for k, v in held.items()}
    fold_gain = [r['global']['rmse'] - r['context']['rmse'] for r in receipts]
    improve = pooled['global'] - pooled['context']
    primary = 'context' if (improve >= 0.001 and sum(g > 0 for g in fold_gain) >= 4) else 'global'
    for k, v in held.items():
        v.astype('<f4').tofile(a.out / f'known-{k}.f32')
    summary = dict(rows=len(y), columns=names, pooled=pooled, folds=receipts, context_minus_global=-improve,
                   context_fold_gains=fold_gain, folds_improved=int(sum(g > 0 for g in fold_gain)),
                   primary_rule='context iff pooled gain >= 0.001 and >= 4/5 folds improve', primary=primary,
                   known_vector_sha256={k: sha_arr(v.astype('<f4')) for k, v in held.items()},
                   bank_known_sha256=bank['known_sha256'], qualifying_labels_read=False)
    if a.final:
        Q = np.load(BANK / 'qual.npy', mmap_mode='r'); qctx = dict(np.load(BANK / 'qual_ctx.npz'))
        allrows = np.arange(len(y)); qrows = np.arange(Q.shape[0])
        dg, wg = fit_global(X, y, allc, allrows)
        dc, wc, lam, sc = fit_context(X, y, users, ctx, allc, allrows, 20261116 + 99)
        qg = Design(Q, None, None, allc).predict(qrows, wg)
        qc = Design(Q, qctx, dc.cuts, allc).predict(qrows, wc)
        for k, v, w in [('global', qg, wg), ('context', qc, wc)]:
            v.astype('<f4').tofile(a.out / f'qual-{k}.f32'); np.save(a.out / f'coef-{k}.npy', w)
        summary['final'] = dict(global_coef_sha256=sha_arr(wg), context_coef_sha256=sha_arr(wc), context_lambda=lam,
                                context_inner_scores=sc, context_cutoffs=[c.tolist() for c in dc.cuts],
                                qual_sha256=dict(glob=sha_arr(qg.astype('<f4')), context=sha_arr(qc.astype('<f4'))),
                                qual_rows=int(Q.shape[0]), qual_target_note='qual targets are synthetic zeros; never read')
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(dict(pooled=pooled, primary=primary, fold_gain=fold_gain)))


if __name__ == '__main__':
    main()
