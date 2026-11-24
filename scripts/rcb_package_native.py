#!/usr/bin/env python3
"""Package complete native producer outputs for coordinator admission.

usage: rcb_package_native.py JOB TREE PREFIX OUTDIR --lineage lineage.json [--extra FILE ...]

Collects every TREE/source/preds_3/PREFIX* file (all epoch/terminal query arrays,
training-prediction arrays, factor side outputs, trace) plus --extra files,
checks every .probe_3/.qual array is finite float32 of the native row count,
writes <=1 GiB-payload tar.gz parts, re-reads and rehashes every decoded member,
and writes OUTDIR/index.json binding each member and part by bytes + SHA-256.
Optionally adds the probex canonical projections (--canonical). Reads no labels.
"""
import argparse, datetime as dt, hashlib, json, sys, tarfile
from pathlib import Path
import numpy as np

N = {'probe_3': 1760494, 'qual': 2817131}
PART = 1024 * 2**20


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''): h.update(b)
    return h.hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('job'); a.add_argument('tree', type=Path); a.add_argument('prefix'); a.add_argument('out', type=Path)
    a.add_argument('--lineage', type=Path, required=True)
    a.add_argument('--extra', type=Path, nargs='*', default=[])
    a.add_argument('--canonical', action='store_true', help='also export probex projections via TREE/probex-row-map.npy')
    x = a.parse_args()
    if x.out.exists(): sys.exit('existing package preserved: ' + str(x.out))
    preds = x.tree / 'source/preds_3'
    files = sorted(p for p in preds.iterdir() if p.is_file() and (p.name.startswith(x.prefix + '.') or p.name.startswith(x.prefix + '_ep')))
    if not files: sys.exit('no native files')
    arrays = {}
    for p in files:
        for split, n in N.items():
            if p.name.endswith(f'.{split}.npy'):
                v = np.load(p, mmap_mode='r', allow_pickle=False)
                if v.shape != (n,) or v.dtype != np.float32 or not np.isfinite(v).all():
                    sys.exit(f'non-finite or wrong shape: {p.name} {v.shape} {v.dtype}')
                arrays[p.name] = dict(split=split, rows=n)
    x.out.mkdir(parents=True)
    members = [(f'native/{p.name}', p) for p in files] + [(f'extra/{p.name}', p) for p in x.extra]
    canon = []
    if x.canonical:
        m = np.load(x.tree / 'probex-row-map.npy', allow_pickle=False)
        cdir = x.out / 'canonical'; cdir.mkdir()
        for name in sorted(arrays):
            if not name.endswith('.probe_3.npy'): continue
            stem = name[:-len('.probe_3.npy')]
            q = preds / f'{stem}.qual.npy'
            if not q.exists(): continue
            t = cdir / f'{stem}.npz'
            known = np.load(preds / name, allow_pickle=False)[m]
            np.savez_compressed(t, probex=known, qual=np.load(q, allow_pickle=False))
            members.append((f'canonical/{t.name}', t)); canon.append(t.name)
    lineage = json.loads(x.lineage.read_text())
    expected = {rel: sha(p) for rel, p in members}
    buckets, cur, size = [], [], 0
    for rel, p in members:
        n = p.stat().st_size
        if cur and size + n > PART: buckets.append(cur); cur, size = [], 0
        cur.append((rel, p)); size += n
    if cur: buckets.append(cur)
    parts = []
    for i, b in enumerate(buckets, 1):
        t = x.out / f'{x.job}-part{i:02d}.tar.gz'
        with tarfile.open(t, 'w:gz', compresslevel=3, dereference=True) as tf:
            for rel, p in b: tf.add(p, arcname=f'{x.job}/{rel}', recursive=False)
        got = {}
        with tarfile.open(t, 'r:gz') as tf:
            for m_ in tf:
                rel = m_.name.split('/', 1)[1]; h = hashlib.sha256(); s = tf.extractfile(m_)
                for blk in iter(lambda: s.read(8 * 2**20), b''): h.update(blk)
                got[rel] = h.hexdigest()
        want = {rel: expected[rel] for rel, _ in b}
        if got != want: sys.exit(f'decoded members differ in {t.name}')
        parts.append(dict(file=t.name, bytes=t.stat().st_size, sha256=sha(t),
                          members={rel: dict(bytes=p.stat().st_size, sha256=expected[rel]) for rel, p in b}))
    index = dict(job=x.job, observed=dt.datetime.now(dt.timezone.utc).isoformat(), tree=str(x.tree),
                 lineage=lineage, native_query_arrays=arrays, canonical_pairs=canon, parts=parts,
                 all_members_decoded_and_rehashed=True, qualifying_labels_read=False)
    (x.out / 'index.json').write_text(json.dumps(index, indent=1) + '\n')
    print(x.job, len(parts), 'parts', sum(p['bytes'] for p in parts), 'bytes', len(members), 'members')


if __name__ == '__main__':
    main()
