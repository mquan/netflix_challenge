"""Fit item-specific ridge regressions on the histories of locally trained neighbors."""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.linalg import solve, blas
from own17x_cf96_followup import ARMS, KNOWN_ROWS, QUAL_ROWS, clipped, load_vectors, rmse, sha, user_folds, audit

ROOT = Path('artifacts')
RANK = ROOT / 'rank1000-knnf-20261125'
BASE = ROOT / 'own17x-cf96-20261124'
OUT = ROOT / 'own-neighbor-20261126'
LABELS = ROOT / 'private-audit/public619-meta-ridge-20261116/qual_ratings.csv.gz'
STEM = 'tsvdx4p-1000lm'
RIDGES = (64., 384.)
K = 40


def history(split, key):
    return np.load(RANK / 'data' / split / (key + '.npy'), mmap_mode='r')


def verified_rank(split, factors=False):
    name = STEM + ('.ifeat.' if factors else '.') + split + '.npy'
    receipt = json.loads((RANK / 'inputs.json').read_text())
    path = RANK / 'preds_new' / name
    assert receipt['rank_parent_training_verified']
    assert sha(path) == receipt['inputs']['artifacts/upstream-parallel-mapped/source/preds_new/' + name]
    values = np.load(path, mmap_mode='r')
    assert np.isfinite(values).all()
    return values


def baseline():
    freeze = BASE / 'followup/freeze.json'
    receipt = json.loads(Path('experiments/OWN17X-cf96-context-tree-20261124.json').read_text())
    assert sha(freeze) == receipt['freeze_sha256']
    prior = json.loads(freeze.read_text())
    name = prior['selected']
    for key, field in [('ratings', 'ratings'), ('user_idxs', 'users')]:
        assert sha(RANK / 'data/probex' / (key + '.npy')) == prior[f'known_{field}_sha256']
    y = history('probex', 'ratings').astype(float)
    folds = user_folds(history('probex', 'user_idxs'))
    _, cf96, terms = load_vectors(BASE, 'known', KNOWN_ROWS)
    x = np.column_stack([terms[key] for key in ARMS[name]])
    p = np.empty_like(y)
    for i, w in enumerate(prior['known'][name]['fold_weights']):
        m = folds == i
        p[m] = clipped(cf96[m] + x[m] @ w)
    assert abs(rmse(p,y) - prior['known'][name]['pooled']) < 1e-12
    qp = BASE / 'followup' / (name + '.f32')
    assert sha(qp) == prior['qualifying_hashes'][name]
    return y, folds, p, np.fromfile(qp, dtype='<f4').astype(float)


def find_neighbors(v, counts):
    v = np.array(v)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    chosen = np.empty((len(v), K), np.int32)
    # Penalize imprecise factors of infrequently rated neighbor movies.
    reliability = np.sqrt(counts / (counts + 1000)).astype('f')
    for lo in range(0, len(v), 192):
        hi = min(lo + 192, len(v))
        sim = (v[lo:hi] @ v.T) * reliability
        sim[np.arange(hi-lo),np.arange(lo,hi)] = -np.inf
        chosen[lo:hi] = np.argpartition(sim, -K, axis=1)[:, -K:]
    return chosen


def generate(train, query, workers):
    lib = ctypes.CDLL(str((OUT / 'features.dylib').resolve()))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.group_items.argtypes = [integer, ptr, ptr, integer, ptr]
    lib.design.argtypes = [integer, ptr, ptr, integer] + [ptr]*7
    lib.group_items.restype = lib.design.restype = None
    items = history(train, 'item_idxs')
    counts = history(train, 'item_cnts')
    starts = np.r_[0, np.cumsum(counts)].astype('i')
    assert starts[-1] == len(items)
    order = np.empty(len(items), 'i')
    lib.group_items(len(items), items.ctypes.data, starts.ctypes.data, len(counts), order.ctypes.data)
    users = history(train, 'user_idxs')[order]
    dates = history(train, 'dates')[order]
    ratings = history(train, 'ratings')[order]
    residuals = ratings - verified_rank(train)[order]
    del order
    qu = history(query, 'user_idxs')
    qi = history(query, 'item_idxs')
    qorder = np.lexsort((qu, qi))
    qstarts = np.r_[0, np.bincount(qi, minlength=len(counts)).cumsum()]
    qusers = qu[qorder]
    qdates = history(query,'dates')[qorder]
    assert len(qi) == (KNOWN_ROWS if query == 'probex' else QUAL_ROWS)
    neighbors = find_neighbors(verified_rank(train, factors=True), counts)
    result = np.zeros((len(qi), len(RIDGES)), 'f')
    common = [a.ctypes.data for a in (starts, users, dates, ratings, residuals)]

    def matrix(u, d, related):
        x = np.zeros((len(u), 5*K + 1), 'f', order='F')
        lib.design(len(u), u.ctypes.data, d.ctypes.data, K, related.ctypes.data, *common, x.ctypes.data)
        x[:,-1] = 1
        return x

    def predict(item):
        sl = slice(starts[item], starts[item+1])
        qs = slice(qstarts[item], qstarts[item+1])
        if qs.start == qs.stop or sl.start == sl.stop:
            return
        related = neighbors[item]
        x = matrix(users[sl], dates[sl], related)
        gram = blas.ssyrk(1., x, trans=1, lower=1)
        rhs = x.T @ residuals[sl]
        del x
        qx = matrix(qusers[qs], qdates[qs], related)
        for j, penalty in enumerate(RIDGES):
            a = gram.copy(order='F')
            a.flat[::len(a)+1] += penalty
            w = solve(a, rhs, assume_a='pos', lower=True, check_finite=False, overwrite_a=True)
            result[qorder[qs], j] = qx @ w

    start = time.monotonic()
    with ThreadPoolExecutor(workers) as executor:
        for j,_ in enumerate(executor.map(predict, range(len(counts)))):
            if j % 1000 == 0:
                print(query, j, round(time.monotonic()-start), flush=True)
    assert np.isfinite(result).all()
    return result


def select(signal, y, folds, p):
    def metrics(pred):
        return dict(pooled=rmse(pred, y), folds=[rmse(pred[folds==i], y[folds==i]) for i in range(5)])
    stats = {'control':metrics(p)}
    for j, column in enumerate(signal.T):
        s = column.astype(float)
        def weight(m):
            return float(np.clip((s[m] @ (y[m]-p[m])) / max(s[m] @ s[m], 1e-16), 0, 2))
        held = p.copy()
        for i in range(5):
            m = folds == i
            held[m] = clipped(p[m] + weight(~m)*s[m])
        stats[str(j)] = metrics(held) | {'weight':weight(slice(None))}
    good = [k for k in stats if k!='control' and stats[k]['pooled'] < stats['control']['pooled']
            and sum(a<b for a,b in zip(stats[k]['folds'], stats['control']['folds'])) >= 4]
    selected = min(good, key=lambda k:stats[k]['pooled']) if good else 'control'
    return selected, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_neighbor_features.cpp','-o',str(OUT/'features.dylib')], check=True)
    y, folds, p, q = baseline()
    known = generate('trainx','probex',args.workers)
    np.save(OUT/'known.npy', known)
    selected, stats = select(known, y, folds, p)
    report = dict(selected=selected, known=stats, penalties=RIDGES, neighbors=K,
                  qualifying_hashes={}, known_comparison_is_nested=False,
                  qualifying_labels_used_for_fitting=False)
    (OUT/'selection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if selected!='control':
        signal = generate('fulltrain','qual',args.workers)
        np.save(OUT/'qual.npy',signal)
        qp = OUT/'selected.f32'
        clipped(q + stats[selected]['weight']*signal[:,int(selected)].astype(float)).astype('<f4').tofile(qp)
        report['qualifying_hashes']['selected'] = sha(qp)
    (OUT/'freeze.json').write_text(json.dumps(report, indent=2)+'\n')
    if report['qualifying_hashes']:
        audit(OUT,LABELS)


if __name__ == '__main__':
    main()
