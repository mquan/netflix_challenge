#!/usr/bin/env python3
"""Build the frozen admitted prediction bank (probex + qual matrices) for the
remote context-blend study. Reads predictions, known probex labels, and native
indices/dates only; never opens qualifying labels (qual ratings are zeros and
are checked to be zero).

Outputs (artifacts/bank/): known.f32 [N_KNOWN, K] row-major, qual.f32 [N_QUAL, K],
context arrays, and experiments/remote-context-blend/bank.json.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

HOME = Path.home() / 'workspace'
ROOT = Path(__file__).resolve().parents[1]
import os
VERSION = os.environ.get('RCB_BANK', 'v1')
OUT = ROOT / ('artifacts/bank' if VERSION == 'v1' else f'artifacts/bank-{VERSION}')
FRESH = HOME / 'netflix_upstream_repro/artifacts/upstream-repro/fresh'
NK, NQ = 1128164, 2817131

# split "new" producers (trainx -> probex), terminal vectors only, fixed list
FRESH_COLS = ['als8-8', 'als8-8__knn3', 'bknbrx-100', 'bknbrx-50', 'dnn-24', 'dnn-24-base', 'dnn-24-net',
              'dnn-24__knn3', 'dnn-24__asym-16p75', 'dnn-24__asym-16p75__knn3', 'dnn-24__dnn-16',
              'dnn-24__dnn-16__knn3', 'dnn-24__dnn-16p50', 'dnn-24__dnn-16p50__knn3',
              'dnn-24__dnn-16p50__asym-16p50', 'dnn-24__dnn-16p50__asym-16p75',
              'dnn-24__dnn-16p50__asym-16p75__knn3b', 'dnn-24__dnn-16p50__asym-16p75__knn3c',
              'dnn-24__dnn-24p50', 'mf-60', 'mf-60__knn3', 'rbmx2-40', 'rbmx2-40__knn3', 'tsvdx4-60',
              'tsvdx4-60__knns', 'tsvdx5-150o', 'tsvdx5-150o-bias', 'tsvdx5-150o-mf', 'tsvdx5-150o-nbr',
              'tsvdx5-150o-nsvd1']
R15 = HOME / 'netflix_remote15h/artifacts'
WK = HOME / 'netflix_remote_weekend/artifacts'
RCB = ROOT / 'artifacts/rcb-r1'
S3_COLS = [  # (column, tree, native stem)
    ('s3-attn32', R15 / 'upstream-remote15h-r2', 's3-attn32'),
    ('s3-dnn24', R15 / 'upstream-remote15h-r2', 's3-dnn24'),
    ('s3-rbm200e', R15 / 'upstream-remote15h-r2', 's3-rbm200e'),
    ('s3-tx50n', R15 / 'upstream-remote15h-ext', 's3-tx50n'),
    ('s3-tx60o', R15 / 'upstream-remote15h-ext', 's3-tx60o'),
    ('s3-mfrbmx-60h', R15 / 'upstream-remote15h-ext', 's3-mfrbmx-60h'),
    ('s3-bknbrx-100w', R15 / 'upstream-remote15h-ext', 's3-bknbrx-100w'),
    ('s3-tx300r', WK / 'upstream-remote-weekend-r10', 's3-tx300r'),
    ('s3-rbm800p', WK / 'upstream-remote-weekend-r9', 's3-rbm800p'),
    ('s3-rbm400pf-lr015', WK / 'upstream-remote-weekend-r11', 's3-rbm400pf'),
]
NEW_COLS = [('s3-tx300r__knn3', RCB, 's3-tx300r__knn3'), ('s3-tx300r__knn3d', RCB, 's3-tx300r__knn3d')]
WAVE_S3 = ['s3-tx50n', 's3-tx60o', 's3-rbm800p', 's3-dnn24', 's3-attn32', 's3-rbm200e', 's3-bknbrx-100w', 's3-mfrbmx-60h', 's3-rbm400pf']
WAVE_NEW = ['tsvdx4-60', 'tsvdx5-150o']
CF48 = HOME / 'netflix_remote9h/artifacts/upstream-remote9h/cfnade48-package/native-source/preds_preset48'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def vec(p, n):
    v = np.load(p, allow_pickle=False)
    if v.shape != (n,) or v.dtype != np.float32 or not np.isfinite(v).all(): raise SystemExit(f'bad {p}')
    return v


def contexts(train_dir, query_dir):
    u = np.load(train_dir / 'user_idxs.npy'); i = np.load(train_dir / 'item_idxs.npy'); d = np.load(train_dir / 'dates.npy')
    nu, ni = int(u.max()) + 1, int(i.max()) + 1
    ucnt = np.bincount(u, minlength=nu); icnt = np.bincount(i, minlength=ni)
    last = np.full(nu, -1, np.int32); np.maximum.at(last, u, d.astype(np.int32))
    qu = np.load(query_dir / 'user_idxs.npy'); qi = np.load(query_dir / 'item_idxs.npy'); qd = np.load(query_dir / 'dates.npy').astype(np.int32)
    lu = last[qu]
    return dict(log_user_support=np.log1p(ucnt[qu]).astype(np.float32),
                log_item_support=np.log1p(icnt[qi]).astype(np.float32),
                days_since_last_train=np.where(lu >= 0, qd - lu, 9999).astype(np.float32))


def main(include_new=True):
    OUT.mkdir(parents=True, exist_ok=True)
    D = FRESH / 'data'
    if np.any(np.load(D / 'qual/ratings.npy', mmap_mode='r')): raise SystemExit('qual targets must be zero')
    cols = []
    for c in FRESH_COLS:
        cols.append(dict(column=c, kind='split-new trainx->probex', known=FRESH / f'preds_new/{c}.probex.npy',
                         qual=FRESH / f'preds_new/{c}.qual.npy', map=None))
    cols.append(dict(column='cfnade-48', kind='split-new trainx->probex (remote9h)', known=CF48 / 'cfnade-48.probex.npy',
                     qual=CF48 / 'cfnade-48.qual.npy', map=None))
    if VERSION == 'v2':
        for base in WAVE_NEW:
            for st in ('knn3', 'knn3d'):
                n = f'{base}__{st}'
                cols.append(dict(column=n, kind='split-new trainx->probex chain (rcb wave)', known=RCB / f'source/preds_new/{n}.probex.npy',
                                 qual=RCB / f'source/preds_new/{n}.qual.npy', map=None))
    wave = [(f'{b}__{st}' if b != 's3-rbm400pf' else f's3-rbm400pf-lr015__{st}', RCB, f'{b}__{st}')
            for b in WAVE_S3 for st in ('knn3', 'knn3d')] if VERSION == 'v2' else []
    for c, tree, stem in S3_COLS + (NEW_COLS if include_new else []) + wave:
        cols.append(dict(column=c, kind='split-3 train_3->probe_3 projected', known=tree / f'source/preds_3/{stem}.probe_3.npy',
                         qual=tree / f'source/preds_3/{stem}.qual.npy', map=tree / 'probex-row-map.npy'))
    K = len(cols)
    km = np.lib.format.open_memmap(OUT / 'known.npy', 'w+', np.float32, (NK, K))
    qm = np.lib.format.open_memmap(OUT / 'qual.npy', 'w+', np.float32, (NQ, K))
    maps = {}
    for j, c in enumerate(cols):
        if c['map'] is None:
            km[:, j] = vec(c['known'], NK)
        else:
            mp = str(c['map']); maps.setdefault(mp, np.load(mp)); km[:, j] = vec(c['known'], 1760494)[maps[mp]]
        qm[:, j] = vec(c['qual'], NQ)
        c.update(known_sha256=sha(c['known']), qual_sha256=sha(c['qual']),
                 map_sha256=sha(c['map']) if c['map'] else None)
        for k in ('known', 'qual', 'map'): c[k] = str(c[k]) if c[k] else None
        print(j, c['column'], flush=True)
    km.flush(); qm.flush()
    y = np.load(D / 'probex/ratings.npy').astype(np.float32); u = np.load(D / 'probex/user_idxs.npy')
    np.save(OUT / 'known_y.npy', y); np.save(OUT / 'known_users.npy', u)
    for name, ctx in [('known', contexts(D / 'trainx', D / 'probex')), ('qual', contexts(D / 'fulltrain', D / 'qual'))]:
        np.savez(OUT / f'{name}_ctx.npz', **ctx)
    rec = dict(version=VERSION, columns=cols, n_known=NK, n_qual=NQ, include_new=include_new,
               known_sha256=sha(OUT / 'known.npy'), qual_sha256=sha(OUT / 'qual.npy'),
               known_ctx_sha256=sha(OUT / 'known_ctx.npz'), qual_ctx_sha256=sha(OUT / 'qual_ctx.npz'),
               context_sources=dict(known='fresh/data/trainx -> probex', qual='fresh/data/fulltrain -> qual'),
               excluded='epoch snapshots, rbm400pf LR0.02 partial, all reference/downloaded vectors, probex-fitted blends',
               qualifying_labels_read=False)
    (ROOT / ('experiments/remote-context-blend/bank.json' if VERSION == 'v1' else f'experiments/remote-context-blend/bank-{VERSION}.json')).write_text(json.dumps(rec, indent=1) + '\n')
    print('bank', K, 'columns')


if __name__ == '__main__':
    main()
