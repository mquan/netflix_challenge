#!/usr/bin/env python3
"""Round 9 Bet 1, arm A: label-free latent factor + bias features from a joint timeSVD++ model.

Reads a `build/netflix joint` model (src/joint.hpp save_joint layout), re-derives
the C++ probe prediction term by term, and writes one float32 column per feature
in probe order (data/processed/netflix.bin), the format `round4_boost.py
--extra-support` expects (np.fromfile(path, '<f4'), length = probe rows).

Per probe row (u, i, t), with s = the user-day session of (u, t):
  pred = mean + bu + bu_dev + but + bi_scaled + bif + dot             (C++ eval_joint)
  bu        b_u
  bu_dev    alpha_u * dev_u(t)
  but       b_{u,t}                    (session day bias)
  bi        b_i + b_{i,Bin(t)}
  cu        c_u + c_{u,t}              (user + session scale)
  bi_scaled bi * cu                    (the term the model actually sums)
  bif       b_{i,f(u,t)}               (frequency bias)
  dot       q_i . (p_u + a_u dev + |N(u)|^-1/2 sum_{j in N(u)} y_j) [+ q_i[:d] . day_factors_s]
  u0..u7    effective user vector (same vector as in `dot`) projected on V
  i0..i7    q_i projected on V
V = top-8 right singular vectors of the item-factor matrix Q (items only,
unweighted, uncentered, so sum_k u_k*i_k is the best rank-8 approximation of
`dot` in item space). Signs fixed so each direction's largest |component| is positive.

N(u) is the model's implicit set: all train+probe+qualifying items of u (no
ratings). Nothing here reads probe ratings except the reproduction check, which
compares against the model's own saved prediction column.

    python scripts/round9_latent_features.py                 # check + write artifacts/r9/lat
    python scripts/round9_latent_features.py --check         # reproduction check only
"""
import argparse
import ctypes
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gbdt_blend import load  # noqa: E402

TIME_BINS, FREQ_BINS = 30, 8
MAGIC_V1, MAGIC_V2 = 0x4E46504A4F494E31, 0x4E46504A4F494E32
ARRAYS = ['user_bias', 'user_alpha', 'user_scale', 'item_bias', 'item_bin_bias',
          'item_frequency_bias', 'day_bias', 'day_scale', 'user_factors', 'item_factors',
          'implicit_factors', 'drift']
BIAS_COLUMNS = ['bu', 'bu_dev', 'but', 'bi', 'cu', 'bi_scaled', 'bif', 'dot']

FNV_C = r'''
#include <stdint.h>
#include <stddef.h>
static uint64_t P = 1099511628211ULL;
uint64_t fnv_bytes(uint64_t h, const unsigned char *b, size_t n) {
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= P; }
  return h;
}
/* ProbeRec fields hashed individually, each widened to uint64 (joint_fingerprint). */
uint64_t fnv_probe(uint64_t h, const unsigned char *rec, size_t n, size_t stride) {
  for (size_t r = 0; r < n; ++r, rec += stride) {
    uint32_t user = *(const uint32_t *)rec; uint16_t item = *(const uint16_t *)(rec+4);
    uint16_t day = *(const uint16_t *)(rec+6); uint8_t rating = rec[8];
    h ^= user; h *= P; h ^= item; h *= P; h ^= day; h *= P; h ^= rating; h *= P;
  }
  return h;
}
'''


def joint_fingerprint(d):
    """FNV-1a of the dataset exactly as src/joint.hpp joint_fingerprint (compiled C helper)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, lib = pathlib.Path(tmp)/'fnv.c', pathlib.Path(tmp)/'fnv.so'
        src.write_text(FNV_C)
        subprocess.run(['cc', '-O2', '-shared', '-fPIC', '-o', str(lib), str(src)], check=True)
        so = ctypes.CDLL(str(lib))
    so.fnv_bytes.restype = so.fnv_probe.restype = ctypes.c_uint64
    so.fnv_bytes.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    so.fnv_probe.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    h = 14695981039346656037
    for key in ('train_offsets', 'train_items', 'train_days', 'train_ratings',
                'implicit_offsets', 'implicit_items', 'implicit_days'):
        a = np.ascontiguousarray(d[key])
        h = so.fnv_bytes(h, a.ctypes.data, a.nbytes)
    p = np.ascontiguousarray(d['probe'])
    return so.fnv_probe(h, p.ctypes.data, len(p), p.dtype.itemsize)


def read_joint(path, d, sessions):
    """Parse save_joint output (memory-mapped); validate every size like load_joint."""
    raw = np.memmap(path, mode='r', dtype=np.uint8)
    magic, fingerprint = (int(x) for x in raw[:16].view('<u8'))
    rank, epoch = (int(x) for x in raw[16:24].view('<u4'))
    if magic not in (MAGIC_V1, MAGIC_V2) or not 0 < rank <= 2048:
        sys.exit('not a joint model: ' + str(path))
    users, items = d['users'], d['items']
    expected = dict(user_bias=users, user_alpha=users, user_scale=users, item_bias=items,
                    item_bin_bias=items*TIME_BINS, item_frequency_bias=items*FREQ_BINS,
                    day_bias=sessions, day_scale=sessions, user_factors=users*rank,
                    item_factors=items*rank, implicit_factors=items*rank, drift=users*rank)
    m, pos = dict(magic=magic, fingerprint=fingerprint, rank=rank, epoch=epoch), 24
    for name in ARRAYS:
        n = int(raw[pos:pos+8].view('<u8')[0]); pos += 8
        if n != expected[name]:
            sys.exit(f'{name}: size {n} != expected {expected[name]}')
        m[name] = raw[pos:pos+4*n].view('<f4'); pos += 4*n
    m['day_rank'], m['day_factors'] = 0, np.zeros(0, np.float32)
    if magic == MAGIC_V2:
        m['day_rank'] = int(raw[pos:pos+4].view('<u4')[0]); pos += 4
        n = int(raw[pos:pos+8].view('<u8')[0]); pos += 8
        if m['day_rank'] > rank or n != sessions*m['day_rank']:
            sys.exit('invalid day-factor dimensions')
        m['day_factors'] = raw[pos:pos+4*n].view('<f4'); pos += 4*n
    if pos != raw.size:
        sys.exit('trailing joint model data')
    for name in ('user_factors', 'item_factors', 'implicit_factors', 'drift'):
        m[name] = m[name].reshape(-1, rank)
    if m['day_rank']:
        m['day_factors'] = m['day_factors'].reshape(-1, m['day_rank'])
    return m


def temporal(d):
    """build_temporal_index + joint_features for probe rows: session id, dev, freq bin, time bin."""
    off = d['implicit_offsets'].astype(np.int64); days = d['implicit_days']
    counts = np.diff(off)
    if (counts == 0).any():
        sys.exit('user without implicit rows (C++ mean_day=0 case not handled)')
    new = np.empty(len(days), bool); new[0] = True
    np.not_equal(days[1:], days[:-1], out=new[1:]); new[off[:-1]] = True
    start = np.flatnonzero(new)                                   # session start positions
    session_count = np.diff(np.append(start, len(days)))
    session_user = np.repeat(np.arange(d['users'], dtype=np.int64),
                             np.add.reduceat(new, off[:-1], dtype=np.int64))
    key = session_user*65536 + days[start]
    del new
    mean_day = (np.add.reduceat(days, off[:-1], dtype=np.uint64).astype(np.float64)
                / counts).astype(np.float32)
    p = d['probe']; u = p['user'].astype(np.int64); day = p['day']
    s = np.searchsorted(key, u*65536 + day)
    if (s >= len(key)).any() or (key[np.minimum(s, len(key)-1)] != u*65536 + day).any():
        sys.exit('rating day absent from implicit set')
    delta = day.astype(np.float32) - mean_day[u]
    dev = (np.sign(delta)*np.power(np.abs(delta), np.float32(.4))).astype(np.float32)
    c = session_count[s].astype(np.float64)
    freq = np.where(c <= 1, 0, np.minimum(FREQ_BINS-1, np.floor(np.log(np.maximum(c, 1))/np.log(6.76))))
    span = d['max_day'] - d['min_day'] + 1
    tbin = np.minimum(TIME_BINS-1, (day.astype(np.int64) - d['min_day'])*TIME_BINS//span)
    return dict(sessions=len(start), session=s, dev=dev, freq=freq.astype(np.int64), tbin=tbin)


def implicit_sums(d, m):
    """|N(u)|^-1/2 sum_{j in N(u)} y_j for every user (build_effective_users), float32 out."""
    import scipy.sparse as sp
    off = d['implicit_offsets'].astype(np.int64)
    norm = (1/np.sqrt(np.diff(off))).astype(np.float32)
    data = np.repeat(norm, np.diff(off))
    S = sp.csr_matrix((data, d['implicit_items'].astype(np.int32) - 1, off),
                      shape=(d['users'], d['items']))
    return np.asarray(S @ np.asarray(m['implicit_factors'], np.float32), np.float32)


def item_directions(Q, k):
    """Top-k right singular vectors of Q (uncentered, unweighted); deterministic signs."""
    Q = np.asarray(Q, np.float64)
    w, v = np.linalg.eigh(Q.T @ Q)
    V = v[:, np.argsort(w)[::-1][:k]]
    V *= np.sign(V[np.abs(V).argmax(0), np.arange(k)])
    explained = np.sort(w)[::-1]
    return V, (explained[:k]/explained.sum())


def features(d, m, t, H, V, chunk=50_000):
    """Every term of eval_joint per probe row, plus projections. Returns (dict of columns, pred)."""
    p = d['probe']; u = p['user'].astype(np.int64); i = p['item'].astype(np.int64) - 1
    s, dev = t['session'], t['dev'].astype(np.float64)
    bu = np.asarray(m['user_bias'], np.float64)[u]
    bu_dev = np.asarray(m['user_alpha'], np.float64)[u]*dev
    but = np.asarray(m['day_bias'], np.float64)[s]
    bi = (np.asarray(m['item_bias'], np.float64)[i]
          + np.asarray(m['item_bin_bias'], np.float64)[i*TIME_BINS + t['tbin']])
    cu = np.asarray(m['user_scale'], np.float64)[u] + np.asarray(m['day_scale'], np.float64)[s]
    bif = np.asarray(m['item_frequency_bias'], np.float64)[i*FREQ_BINS + t['freq']]
    P, Q, A = m['user_factors'], m['item_factors'], m['drift']
    dot = np.empty(len(u)); ucol = np.empty((len(u), V.shape[1])); icol = np.empty((len(u), V.shape[1]))
    dr = m['day_rank']
    for a in range(0, len(u), chunk):
        b = slice(a, a+chunk); uu, ii = u[b], i[b]
        eff = (np.asarray(P[uu], np.float64) + np.asarray(H[uu], np.float64)
               + np.asarray(A[uu], np.float64)*dev[b, None])
        q = np.asarray(Q[ii], np.float64)
        dot[b] = np.einsum('rk,rk->r', q, eff)
        if dr:
            dot[b] += np.einsum('rk,rk->r', q[:, :dr], np.asarray(m['day_factors'][s[b]], np.float64))
            eff[:, :dr] += np.asarray(m['day_factors'][s[b]], np.float64)
        ucol[b] = eff @ V; icol[b] = q @ V
    cols = dict(bu=bu, bu_dev=bu_dev, but=but, bi=bi, cu=cu, bi_scaled=bi*cu, bif=bif, dot=dot)
    for k in range(V.shape[1]):
        cols[f'u{k}'] = ucol[:, k]
    for k in range(V.shape[1]):
        cols[f'i{k}'] = icol[:, k]
    pred = d['mean'] + bu + bu_dev + but + bi*cu + bif + dot
    return cols, pred


def reproduce(pred, reference_path, rmse_rows=None):
    ref = np.fromfile(reference_path, '<f4').astype(np.float64)
    if len(ref) != len(pred):
        sys.exit(f'{reference_path}: {len(ref)} rows != {len(pred)} probe rows')
    diff = np.abs(pred - ref)
    out = dict(reference=str(reference_path), rows=int(len(ref)), max_abs_diff=float(diff.max()),
               mean_abs_diff=float(diff.mean()), rows_over_1e4=int((diff > 1e-4).sum()))
    if rmse_rows is not None:
        y = rmse_rows.astype(np.float64)
        out['rmse_recomputed'] = float(np.sqrt(np.mean((np.clip(pred, 1, 5) - y)**2)))
        out['rmse_reference'] = float(np.sqrt(np.mean((np.clip(ref, 1, 5) - y)**2)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/processed/netflix.bin')
    ap.add_argument('--model', default='artifacts/K1-joint256.bin')
    ap.add_argument('--reference', default='artifacts/K1-joint256.f32',
                    help="the model's own probe predictions (build/netflix joint --predictions)")
    ap.add_argument('--output', default='artifacts/r9/lat')
    ap.add_argument('--prefix', default='lat-')
    ap.add_argument('--components', type=int, default=8)
    ap.add_argument('--tolerance', type=float, default=1e-4)
    ap.add_argument('--check', action='store_true', help='reproduction check only; write nothing')
    ap.add_argument('--skip-fingerprint', action='store_true')
    a = ap.parse_args(argv)
    t0 = time.time()
    d = load(a.data)
    t = temporal(d)
    m = read_joint(a.model, d, t['sessions'])
    print(f"model rank={m['rank']} epoch={m['epoch']} day_rank={m['day_rank']} sessions={t['sessions']}", flush=True)
    fp = None
    if not a.skip_fingerprint:
        fp = joint_fingerprint(d)
        if fp != m['fingerprint']:
            sys.exit(f"dataset fingerprint {fp} != model fingerprint {m['fingerprint']}")
        print(f'fingerprint ok {fp}', flush=True)
    H = implicit_sums(d, m)
    V, explained = item_directions(m['item_factors'], a.components)
    cols, pred = features(d, m, t, H, V)
    check = reproduce(pred, a.reference, d['probe']['rating'])
    check['fingerprint'] = fp
    print('reproduction', json.dumps(check), flush=True)
    if check['max_abs_diff'] > a.tolerance:
        sys.exit(f"reproduction failed: max |diff| {check['max_abs_diff']:.3g} > {a.tolerance}")
    if a.check:
        print(f'check ok in {time.time()-t0:.0f}s'); return check
    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    order = [f'u{k}' for k in range(a.components)] + [f'i{k}' for k in range(a.components)] + BIAS_COLUMNS
    paths = []
    for name in order:
        path = out/f'{a.prefix}{name}.f32'
        np.asarray(cols[name], '<f4').tofile(path); paths.append(str(path))
    manifest = dict(
        script='scripts/round9_latent_features.py', model=a.model, model_epoch=m['epoch'],
        model_rank=m['rank'], day_rank=m['day_rank'], data=a.data, rows=len(pred),
        pca=dict(kind='uncentered SVD of item factors, items only, unweighted',
                 components=a.components, explained_second_moment=explained.tolist()),
        columns=paths, reproduction=check, seconds=round(time.time()-t0, 1))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'wrote {len(paths)} columns + {out}/manifest.json in {time.time()-t0:.0f}s')
    print('--extra-support ' + ' '.join(paths))
    return manifest


if __name__ == '__main__':
    main()
