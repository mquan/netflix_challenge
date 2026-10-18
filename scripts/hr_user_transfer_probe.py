"""Diagnostic: does a user's out-of-sample P2 residual predict their probe residual
under the promoted R9-LAT-A2h blend?  P2 columns come from models trained on T-P2."""
import sys, numpy as np
sys.path.insert(0, '/Users/quan/workspace/netflix_challenge/scripts')
from gbdt_blend import load, blend_fold
R = '/Users/quan/workspace/netflix_challenge/artifacts/rehearsal/'
cols = ['A-MH3-categorical-mse.p2.f32','A-MT0-select.p2.f32','A-MT1-select.p2.f32','A-MT2-select.p2.f32',
        'A-MT3-select.p2.f32','A-K1.p2.f32','A-knn256td.train-matched-20260927.f32','A-knn64.train-matched-20260927.f32',
        'A-rbm100c.p2.f32','r12/A-B4-union.p2.f32','r12/A-A2h-select.p2.f32']
p2 = load('/Users/quan/workspace/netflix_challenge/artifacts/validation/train-matched-20260927.bin')
full = load('/Users/quan/workspace/netflix_challenge/data/processed/netflix.bin')
assert p2['users'] == full['users']
P = p2['probe']; y2 = P['rating'].astype(np.float64); u2 = P['user'].astype(np.int64)
X = np.column_stack([np.fromfile(R+c, dtype='<f4') for c in cols] + [np.ones(len(P))]).astype(np.float64)
# P2 blend residuals out-of-fold across the same user folds (ridge)
f2 = blend_fold(u2); r2 = np.empty(len(P))
for k in range(5):
    tr, te = f2 != k, f2 == k
    w = np.linalg.solve(X[tr].T @ X[tr] + 1e-3*np.eye(X.shape[1]), X[tr].T @ y2[tr])
    r2[te] = y2[te] - X[te] @ w
print('P2 OOF ridge RMSE', np.sqrt(np.mean(r2**2)))
U = full['users']
cnt = np.bincount(u2, minlength=U).astype(np.float64)
s = np.bincount(u2, weights=r2, minlength=U)
Q = full['probe']; uq = Q['user'].astype(np.int64); yq = Q['rating'].astype(np.float64)
pred = np.fromfile('/Users/quan/workspace/netflix_round9/artifacts/r9/R9-LAT-A2h-20261010-NN2.avg.f32', dtype='<f4').astype(np.float64)
assert len(pred) == len(Q)
rq = yq - pred
print('R9-LAT-A2h RMSE', np.sqrt(np.mean(rq**2)), 'probe rows with P2 history', np.mean(cnt[uq] > 0))
fq = blend_fold(uq)
for lam in (2, 5, 10, 25):
    feat = s[uq] / (cnt[uq] + lam)
    out = np.empty(len(Q))
    for k in range(5):
        tr, te = fq != k, fq == k
        b = (feat[tr] @ rq[tr]) / (feat[tr] @ feat[tr])
        out[te] = rq[te] - b*feat[te]
    folds = [np.sqrt(np.mean(out[fq==k]**2)) - np.sqrt(np.mean(rq[fq==k]**2)) for k in range(5)]
    print(f'lam {lam}: corr {np.corrcoef(feat, rq)[0,1]:.4f}  delta {np.sqrt(np.mean(out**2)) - np.sqrt(np.mean(rq**2)):+.6f}  folds', ' '.join(f'{d:+.6f}' for d in folds))
