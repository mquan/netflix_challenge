"""Throughput-only pilot: CatBoost (full-context plan parameters) on fold-0 training rows of
the local bank + 3 contexts, fitting residuals of the fold's global ridge. Reports seconds per
iteration and peak RSS; the eval RMSE is a sanity value, not a selection input."""
import json, resource, sys, time
from pathlib import Path
import numpy as np, catboost
sys.path.insert(0, str(Path(__file__).parent)); import rcb_blend as b
ROOT = b.ROOT; X = np.load(b.BANK / 'known.npy', mmap_mode='r'); y = np.load(b.BANK / 'known_y.npy')
u = np.load(b.BANK / 'known_users.npy'); ctx = dict(np.load(b.BANK / 'known_ctx.npz'))
f = b.splitmix_folds(u, 20261116, 5); tr = np.flatnonzero(f != 0); te = np.flatnonzero(f == 0)
cols = list(range(X.shape[1])); d, w = b.fit_global(X, y, cols, tr)
def feats(idx): return np.column_stack([np.clip(X[idx], 0, 6), *[ctx[n][idx] for n in b.CTX_NAMES], d.predict(idx, w)]).astype(np.float32)
Ftr, Fte = feats(tr), feats(te); rtr = y[tr] - Ftr[:, -1]; rte = y[te] - Fte[:, -1]
out = []
for threads in (2, 8):
    m = catboost.CatBoostRegressor(loss_function='RMSE', eval_metric='RMSE', depth=8, learning_rate=0.03, l2_leaf_reg=10,
        border_count=64, bootstrap_type='Bernoulli', subsample=0.8, rsm=0.9, thread_count=threads, task_type='CPU',
        grow_policy='SymmetricTree', iterations=200, random_seed=2026, allow_writing_files=False, verbose=False)
    t = time.time(); m.fit(Ftr, rtr, eval_set=(Fte, rte)); dt = time.time() - t
    out.append(dict(threads=threads, iterations=200, seconds=round(dt, 1), sec_per_iter=round(dt / 200, 3),
                    eval_rmse_after_200=float(m.get_best_score()['validation']['RMSE'])))
    print(out[-1], flush=True)
rec = dict(train_rows=len(tr), features=Ftr.shape[1], runs=out, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           catboost=catboost.__version__, note='throughput pilot only; local 43-col bank, not the full-context closure',
           full_arm_estimate_note='2 arms x 5 folds x 3 seeds x (inner fit + refit) at <=2000 iters')
(ROOT / 'experiments/remote-context-blend/catboost/pilot.json').write_text(json.dumps(rec, indent=1) + '\n')
