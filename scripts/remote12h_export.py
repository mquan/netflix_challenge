#!/usr/bin/env python3
"""Verify a completed fresh native job and export its ProbeX vectors (REMOTE_12H_PLAN).

Every native receipt output hash is re-verified. Terminal and kept-epoch ProbeX
vectors bound by the receipt are written as separate compressed NPZ files
(`prediction` key, raw float32, native order, no clipping). A transfer manifest
records source/NPZ/content/receipt hashes, alignment hashes, data receipt,
source pin, host and environment. Weights and fulltrain/qual outputs stay local.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/upstream-repro'
FRESH = BASE/'fresh'
OUT = ROOT/'experiments/upstream-remote-12h'
PIN = '49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c'
ROWS = 1128164


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def content_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f4').tobytes()).hexdigest()


def sysctl(key):
    return subprocess.run(['sysctl', '-n', key], capture_output=True, text=True).stdout.strip()


def export(job, kept):
    receipt_path = BASE/'fresh-receipts'/(job+'.json')
    receipt = json.loads(receipt_path.read_text())
    if receipt['source_commit'] != PIN or receipt['job'] != job:
        raise ValueError('Receipt does not belong to this pinned job')
    for path, digest in receipt['outputs'].items():
        if sha(FRESH/path) != digest:
            raise ValueError(f'Receipt output hash mismatch: {path}')
    probex = sorted(p for p in receipt['outputs'] if p.endswith('.probex.npy'))
    expected = [f'preds_new/{job}.probex.npy'] + [f'preds_new/{job}_ep{e}.probex.npy' for e in kept]
    missing = sorted(set(expected) - set(probex))
    if missing:
        raise ValueError(f'Receipt lacks declared ProbeX outputs: {missing}')
    y = np.load(FRESH/'data/probex/ratings.npy', allow_pickle=False).astype(np.float64)
    out_dir = OUT/'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    job_dir = OUT/job
    job_dir.mkdir(parents=True, exist_ok=True)
    receipt_sha = sha(receipt_path)
    shutil.copyfile(receipt_path, job_dir/'native-receipt.json')
    shutil.copyfile(BASE/'fresh-receipts'/(job+'.log'), job_dir/'native.log')
    if (FRESH/f'preds_new/{job}.out').exists():
        shutil.copyfile(FRESH/f'preds_new/{job}.out', job_dir/'native.out')
    records = []
    for rel in expected:
        array = np.load(FRESH/rel, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (ROWS,) or not np.all(np.isfinite(array)):
            raise ValueError(f'{rel}: expected finite float32 ({ROWS},), got {array.dtype} {array.shape}')
        stem = Path(rel).name[:-len('.npy')]
        npz = out_dir/(stem+'.npz')
        np.savez_compressed(npz, prediction=array)
        with np.load(npz, allow_pickle=False) as check:
            if not np.array_equal(check['prediction'], array) or check['prediction'].dtype != np.float32:
                raise ValueError(f'NPZ round-trip mismatch: {npz}')
        records.append(dict(kind='terminal' if rel == expected[0] else 'kept-epoch',
                            source=rel, source_npy_sha256=sha(FRESH/rel),
                            receipt_bound_sha256=receipt['outputs'][rel],
                            npz=str(npz.relative_to(ROOT)), npz_sha256=sha(npz),
                            prediction_content_sha256=content_sha(array),
                            owning_native_receipt_sha256=receipt_sha,
                            diagnostic_standalone_probex_rmse=float(np.sqrt(np.mean((array - y) ** 2))),
                            min=float(array.min()), max=float(array.max())))
    by_content = {}
    for r in records:
        by_content.setdefault(r['prediction_content_sha256'], []).append(r['source'])
    alignment = {name: sha(FRESH/f'data/probex/{name}') for name in
                 ['ratings.npy', 'user_idxs.npy', 'item_idxs.npy', 'dates.npy']}
    data_receipt = json.loads((BASE/'fresh-data-receipt.json').read_text())
    if data_receipt['ratings_sha256'] != alignment['ratings.npy']:
        raise ValueError('ProbeX ratings differ from the fresh data receipt')
    head = subprocess.check_output(['git', '-C', str(FRESH), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = dict(
        job=job, created=dt.datetime.now(dt.timezone.utc).isoformat(), source_commit=head,
        source_commit_matches_pin=head == PIN, native_receipt=str((job_dir/'native-receipt.json').relative_to(ROOT)),
        native_receipt_sha256=receipt_sha, all_receipt_output_hashes_verified=len(receipt['outputs']),
        native_command=receipt['command'], native_started=receipt['started'], native_finished=receipt['finished'],
        outputs=records, identical_content_groups=[v for v in by_content.values() if len(v) > 1],
        probex_rows=ROWS, alignment_sha256=alignment, fresh_data_receipt=data_receipt,
        host=dict(platform=platform.platform(), machine=platform.machine(), cpu=sysctl('machdep.cpu.brand_string'),
                  ncpu=int(sysctl('hw.ncpu')), physical_ram_bytes=int(sysctl('hw.memsize'))),
        environment=dict(UPSTREAM_REPRO_THREADS='8', UPSTREAM_REPRO_BLAS_THREADS='1', RAYON_NUM_THREADS='8',
                         VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                         wrapper=f'.venv/bin/python -u scripts/upstream_repro.py --run-job {job}',
                         rustc=subprocess.run(['rustc', '--version'], capture_output=True, text=True).stdout.strip()),
        notes=['Raw native float32 values, native ProbeX order, no export-time clipping',
               'Standalone ProbeX RMSE is a diagnostic only, not an ensemble result',
               'Model weights and fulltrain/qual predictions remain on the originating machine',
               'Labels not transferred; use alignment hashes'])
    (job_dir/'transfer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({r['source']: round(r['diagnostic_standalone_probex_rmse'], 6) for r in records}, indent=2))
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--kept', nargs='*', default=[], help='Kept epoch labels, e.g. 01 05')
    a = ap.parse_args()
    export(a.job, a.kept)
