#!/usr/bin/env python3
"""Local CatBoost tree-residual blend arm (declared in REMOTE_CONTEXT_BLEND_EXECUTION_PLAN.md).

One unit = (outer fold f | 'final', seed slot s). Features: clipped bank columns, the three
contexts, and the ridge baseline (last). Target: residual from the global ridge (λ=1000).
Inner split: user-index parity inside the outer-training users; the inner baseline is refit
on inner-train rows only. Early stopping on inner-held (max 2000, patience 100); if the best
model does not beat the zero correction, the slot keeps 0 iterations; otherwise refit fresh
on all outer-training rows with best_iteration+1. Saves the correction vector for the held
rows (or qual rows for 'final') and the model. Never reads qualifying labels.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, catboost
import rcb_blend as b

SEEDS = [2026, 7, 13]
PARAMS = dict(loss_function='RMSE', eval_metric='RMSE', depth=8, learning_rate=0.03, l2_leaf_reg=10,
              border_count=64, bootstrap_type='Bernoulli', subsample=0.8, rsm=0.9, thread_count=2,
              task_type='CPU', grow_policy='SymmetricTree', allow_writing_files=False, verbose=False)
MAX_IT, PATIENCE = 2000, 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fold', required=True); ap.add_argument('--slot', type=int, required=True)
    ap.add_argument('--max-it', type=int, default=MAX_IT)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.fold}-s{a.slot}'
    if (a.out / f'{tag}.json').exists(): raise SystemExit('unit done; preserved')
    X = np.load(a.bank / 'known.npy', mmap_mode='r'); y = np.load(a.bank / 'known_y.npy')
    users = np.load(a.bank / 'known_users.npy'); ctx = dict(np.load(a.bank / 'known_ctx.npz'))
    cols = list(range(X.shape[1])); folds = b.splitmix_folds(users, 20261116, 5)
    final = a.fold == 'final'
    tr = np.arange(len(y)) if final else np.flatnonzero(folds != int(a.fold))
    if final:
        Q = np.load(a.bank / 'qual.npy', mmap_mode='r'); qctx = dict(np.load(a.bank / 'qual_ctx.npz'))
        te, Xte, cte = np.arange(Q.shape[0]), Q, qctx
    else:
        te = np.flatnonzero(folds == int(a.fold)); Xte, cte = X, ctx
    seed = SEEDS[a.slot] + (99 if final else int(a.fold))

    def feats(M, c, idx, d, w):
        return np.column_stack([np.clip(M[idx], 0, 6), *[c[n][idx] for n in b.CTX_NAMES], d.predict(idx, w)]).astype(np.float32)

    t0 = time.time()
    odd = (users[tr] % 2).astype(bool); itr, iho = tr[~odd], tr[odd]
    di, wi = b.fit_global(X, y, cols, itr)
    Fi, Fh = feats(X, ctx, itr, di, wi), feats(X, ctx, iho, di, wi)
    m = catboost.CatBoostRegressor(**PARAMS, iterations=a.max_it, random_seed=seed, od_type='Iter', od_wait=PATIENCE,
                                   use_best_model=True)
    m.fit(Fi, y[itr] - Fi[:, -1], eval_set=(Fh, y[iho] - Fh[:, -1]))
    zero = float(np.sqrt(np.mean((y[iho] - Fh[:, -1]) ** 2)))
    best = float(m.get_best_score()['validation']['RMSE']); n_it = m.get_best_iteration() + 1 if best < zero else 0
    do, wo = b.fit_global(X, y, cols, tr)
    corr = np.zeros(len(te))
    if n_it:
        Ftr = feats(X, ctx, tr, do, wo)
        r = catboost.CatBoostRegressor(**PARAMS, iterations=n_it, random_seed=seed)
        r.fit(Ftr, y[tr] - Ftr[:, -1]); r.save_model(str(a.out / f'{tag}.cbm'))
        for s in range(0, len(te), 262144):
            idx = te[s:s + 262144]; corr[s:s + len(idx)] = r.predict(feats(Xte, cte, idx, do, wo))
    np.save(a.out / f'{tag}.corr.npy', corr)
    base = do.predict(te, wo); np.save(a.out / f'{a.fold}.base.npy', base) if a.slot == 0 else None
    rec = dict(unit=tag, seed=seed, inner_train=len(itr), inner_held=len(iho), inner_zero_rmse=zero, inner_best_rmse=best,
               selected_iterations=n_it, rows=len(te), seconds=round(time.time() - t0, 1),
               corr_sha256=b.sha_arr(corr), qualifying_labels_read=False)
    (a.out / f'{tag}.json').write_text(json.dumps(rec, indent=1) + '\n'); print(json.dumps(rec))


if __name__ == '__main__':
    main()
