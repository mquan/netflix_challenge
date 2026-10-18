#!/usr/bin/env python3
"""Export frozen joint-model coordinates and bias terms without reading probe labels."""
import argparse
import hashlib
import json
import pathlib
import struct

import numpy as np
from gbdt_blend import load

NAMES = ('user_bias user_alpha user_scale item_bias item_bin_bias item_frequency_bias '
         'day_bias day_scale user_factors item_factors implicit_factors drift').split()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_model(path, users, items):
    arrays = {}
    with open(path, 'rb') as f:
        magic, fingerprint, rank, epoch = struct.unpack('<QQII', f.read(24))
        if magic not in (0x4E46504A4F494E31, 0x4E46504A4F494E32) or not 0 < rank <= 2048:
            raise ValueError('invalid joint model header')
        for name in NAMES:
            n, = struct.unpack('<Q', f.read(8))
            arrays[name] = np.memmap(path, mode='r', dtype='<f4', offset=f.tell(), shape=(n,))
            f.seek(n * 4, 1)
        if magic == 0x4E46504A4F494E32:
            day_rank, count = struct.unpack('<IQ', f.read(12))
            if day_rank or count:
                raise ValueError('day-factor models are not supported by this exporter')
        if f.read(1):
            raise ValueError('trailing model data')
    for name in ('user_factors', 'drift'):
        arrays[name] = arrays[name].reshape(users, rank)
    for name in ('item_factors', 'implicit_factors'):
        arrays[name] = arrays[name].reshape(items, rank)
    for name in ('user_bias', 'user_alpha', 'user_scale'):
        assert len(arrays[name]) == users
    assert len(arrays['item_bias']) == items
    return arrays, dict(rank=rank, epoch=epoch, fingerprint=fingerprint)


def features(d, m, dimensions):
    # Fit the shared coordinate system on frozen item vectors, never on labels.
    q = np.asarray(m['item_factors'], dtype=np.float64)
    centered = q - q.mean(0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    basis = vectors[:, -dimensions:][:, ::-1]
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(dimensions)])
    basis *= signs
    item = q @ basis
    effective = m['user_factors'] @ basis
    implicit = m['implicit_factors'] @ basis
    drift = m['drift'] @ basis
    probe = d['probe']; users = probe['user'].astype(np.int64)
    items = probe['item'].astype(np.int64) - 1
    days = probe['day'].astype(np.int64)
    order = np.argsort(users, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(users, minlength=d['users']))]
    sessions = np.empty(len(probe), dtype=np.int64)
    counts = np.empty(len(probe), dtype=np.int64)
    dev = np.empty(len(probe), dtype=np.float32)
    session_offset = 0
    for u in range(d['users']):
        lo, hi = map(int, d['implicit_offsets'][u:u+2])
        ids = d['implicit_items'][lo:hi].astype(np.int64) - 1
        history_days = d['implicit_days'][lo:hi]
        if len(ids):
            effective[u] += implicit[ids].sum(0) / np.sqrt(len(ids))
        unique, n = np.unique(history_days, return_counts=True)
        rows = order[offsets[u]:offsets[u+1]]
        if len(rows):
            pos = np.searchsorted(unique, days[rows])
            assert np.array_equal(unique[pos], days[rows])
            sessions[rows] = session_offset + pos
            counts[rows] = np.minimum(n[pos], 65535)
            delta = days[rows].astype(np.float32) - np.float32(history_days.mean())
            dev[rows] = np.copysign(np.abs(delta) ** np.float32(.4), delta)
        session_offset += len(unique)
    assert session_offset == len(m['day_bias']) == len(m['day_scale'])
    freq = np.minimum(7, np.floor(np.log(np.maximum(counts, 1))/np.log(6.76)).astype(np.int64))
    timebin = np.minimum(29, (days-d['min_day'])*30//(d['max_day']-d['min_day']+1))
    out = {}
    for k in range(dimensions):
        out[f'user_pc{k}'] = effective[users, k] + drift[users, k]*dev
        out[f'item_pc{k}'] = item[items, k]
    out.update(user_bias=m['user_bias'][users], user_drift=m['user_alpha'][users]*dev,
               day_bias=m['day_bias'][sessions],
               scale=m['user_scale'][users]+m['day_scale'][sessions],
               item_bias=m['item_bias'][items]+m['item_bin_bias'][items*30+timebin],
               frequency_bias=m['item_frequency_bias'][items*8+freq])
    return out, basis, dev


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--reference', required=True, help='raw joint prediction column for reconstruction check')
    ap.add_argument('--output', required=True); ap.add_argument('--dimensions', type=int, default=8)
    a = ap.parse_args(); dest = pathlib.Path(a.output); dest.mkdir(parents=True, exist_ok=False)
    d = load(a.data); m, info = read_model(a.model, d['users'], d['items'])
    if not 0 < a.dimensions <= info['rank']: raise ValueError('invalid dimensions')
    out, basis, dev = features(d, m, a.dimensions)
    # Independent full-rank reconstruction checks session alignment and factor semantics.
    rows = np.linspace(0, len(d['probe'])-1, min(2048, len(d['probe'])), dtype=int)
    pred = np.full(len(rows), np.float32(d['mean']), dtype=np.float32)
    for name in ('user_bias', 'user_drift', 'day_bias'):
        pred += out[name][rows]
    pred += out['item_bias'][rows]*out['scale'][rows]
    pred += out['frequency_bias'][rows]
    pred = pred.astype(np.float64)
    for j, row in enumerate(rows):
        u, i = int(d['probe']['user'][row]), int(d['probe']['item'][row])-1
        lo, hi = map(int, d['implicit_offsets'][u:u+2])
        ids = d['implicit_items'][lo:hi].astype(np.int64)-1
        # Match C++ sequential float32 accumulation.
        p = m['user_factors'][u].copy()
        norm = np.float32(1)/np.sqrt(np.float32(max(len(ids), 1)))
        for idx in ids: p += norm*m['implicit_factors'][idx]
        pred[j] += np.sum(m['item_factors'][i]*(p+m['drift'][u]*dev[row]), dtype=np.float64)
    expected = np.fromfile(a.reference, dtype='<f4')[rows]
    error = float(np.max(np.abs(pred-expected)))
    if error > 2e-5: raise ValueError(f'prediction reconstruction mismatch: {error}')
    paths = []
    for name, values in out.items():
        assert np.isfinite(values).all()
        path = dest/(name+'.f32'); np.asarray(values, dtype='<f4').tofile(path); paths.append(str(path))
    np.save(dest/'basis.npy', basis)
    report = dict(config=vars(a), model=info, max_reconstruction_error=error,
                  features=paths, inputs={p: digest(p) for p in (a.data, a.model, a.reference)},
                  outputs={p: digest(p) for p in paths}, source_sha256=digest(__file__))
    (dest/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(features=len(paths), max_reconstruction_error=error)), flush=True)


if __name__ == '__main__': main()
