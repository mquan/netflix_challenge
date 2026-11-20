#!/usr/bin/env python3
"""Full real-row input/scaler parity; no proxy model fit or qualifying outcomes."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import upstream_cf96_prefix2030_virtual_features as data
import upstream_nonlinear as original
import upstream_exact_neural_scaling as compact
import upstream_knn3pair2_virtual_queue as resource

u = data.u
RUN = u.ROOT / 'artifacts/upstream-cf96-prefix2030-scaling-parity'
REPORT = u.ROOT / 'experiments/upstream-cf96-prefix2030-scaling-parity'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha_bytes(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build():
    if RUN.exists() or (REPORT / 'plan.json').exists():
        raise ValueError('Existing real-data parity evidence preserved')
    if shutil.disk_usage(u.ROOT).free < 16 * 2**30:
        raise ValueError('Full-size input cache requires sixteen GiB disk headroom')
    REPORT.mkdir(parents=True, exist_ok=True)
    source_proof = data.REPORT / 'cache-column-verification.json'
    proof = u.read(source_proof)
    if proof['state'] != 'complete_cache_columns' or not proof['all_retained_and_new_columns_checked']:
        raise ValueError('Complete original virtual feature proof required')
    parent = u.read(data.REPORT / 'manifest.json')
    if parent['rows'] != 1128164 or parent['nonlinear_features'] != 399 or parent['design_columns'] != 185:
        raise ValueError('Unreduced complete original matrix schema required')
    paths = [Path(__file__), Path(compact.__file__), Path(original.__file__), source_proof,
             data.REPORT / 'manifest.json', data.RUN / 'groups.npy', data.RUN / 'baseline.npy']
    bindings = {str(p.relative_to(u.ROOT)): u.sha(p) for p in paths}
    u.write(REPORT / 'plan.json', dict(created=stamp(), bindings=bindings, rows=1128164,
            ridge_columns=185, neural_input_columns=400,
            rule='Every actual known row and every original virtual column. Exact float32 baseline assembly, original complete ordered inner/full eligible scaler sets of outer fold0, original reference statistics and every real-row normalized value checked bitwise. Array transformations only: no model fitting, architecture change, row subsampling or qualifying-label access.',
            models_fitted=0, original_model_or_budget_changed=False, qualifying_labels_read=False))
    RUN.mkdir()
    arrays = data.inputs()
    baseline = np.load(data.RUN / 'baseline.npy', mmap_mode='r')
    z = np.lib.format.open_memmap(RUN / 'design.npy', mode='w+', dtype=np.float64, shape=(1128164, 185))
    x = np.lib.format.open_memmap(RUN / 'inputs.npy', mode='w+', dtype=np.float32, shape=(1128164, 400))
    digests = [hashlib.sha256(), hashlib.sha256()]
    actual_input, reference_input = hashlib.sha256(), hashlib.sha256()
    for lo in range(0, 1128164, 8192):
        hi = min(lo + 8192, 1128164)
        zb, fb = data.block(arrays, lo, hi)
        expected = np.column_stack([fb, np.asarray(baseline[lo:hi], dtype=np.float64)]).astype(np.float32)
        actual = compact.append_baseline(fb, np.asarray(baseline[lo:hi], dtype=np.float64))
        if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
            raise ValueError('Original complete raw input assembly differs')
        z[lo:hi], x[lo:hi] = zb, actual
        for h, a in zip(digests, [zb, fb]):
            h.update(np.ascontiguousarray(a).tobytes())
        actual_input.update(actual.tobytes());reference_input.update(expected.tobytes())
    z.flush();x.flush()
    del arrays, baseline, z, x
    if [h.hexdigest() for h in digests] != proof['virtual_c_order_sha256']:
        raise ValueError('Every complete original matrix fingerprint must match')
    u.write(REPORT / 'cache-verification.json', dict(state='complete_actual_virtual_matrices_and_input_assembly',
            observed=stamp(), rows=1128164, full_virtual_c_order_sha256=[h.hexdigest() for h in digests],
            actual_input_sha256=actual_input.hexdigest(), original_input_sha256=reference_input.hexdigest(),
            every_float32_input_bit_equal=True, original_virtual_proof_sha256=u.sha(source_proof),
            files_sha256={p.name: u.sha(p) for p in RUN.iterdir()}, models_fitted=0,
            qualifying_labels_read=False))


def worker():
    plan = u.read(REPORT / 'plan.json')
    for name, digest in plan['bindings'].items():
        if u.sha(u.ROOT / name) != digest:
            raise ValueError('Original parity source changed')
    cache = u.read(REPORT / 'cache-verification.json')
    for name, digest in cache['files_sha256'].items():
        if u.sha(RUN / name) != digest:
            raise ValueError('Complete actual input cache changed')
    x = np.load(RUN / 'inputs.npy', mmap_mode='r')
    groups = np.load(data.RUN / 'groups.npy')
    training = groups // 2 != 0
    records = []
    for name, mask in [('inner', training & (groups % 2 == 0)), ('full', training)]:
        start = time.monotonic()
        # This is the entire original eligible fit set in its original order.
        # Using a slice avoids a redundant fancy-index copy inside the reference.
        selected = np.asarray(x[mask], dtype=np.float32, order='C')
        reference_values, reference = original.scaled(selected, slice(None))
        scaler = compact.fit_scaler(x, mask)
        for key in ['lo', 'hi', 'mean', 'std']:
            if not np.array_equal(reference[key].view(np.uint64), scaler[key].view(np.uint64)):
                raise ValueError('Original complete eligible statistics differ: ' + name + '/' + key)
        for lo in range(0, len(selected), 8192):
            hi = min(lo + 8192, len(selected))
            actual = compact.apply_scaler(selected[lo:hi], scaler)
            if not np.array_equal(actual.view(np.uint32), reference_values[lo:hi].view(np.uint32)):
                raise ValueError('Complete original eligible normalized values differ')
        del selected, reference_values
        expected_digest, actual_digest = hashlib.sha256(), hashlib.sha256()
        for lo in range(0, len(x), 8192):
            hi = min(lo + 8192, len(x))
            expected, _ = original.scaled(np.asarray(x[lo:hi]), None, scaler=reference)
            actual = compact.apply_scaler(x[lo:hi], scaler)
            if not np.array_equal(expected.view(np.uint32), actual.view(np.uint32)):
                raise ValueError('Every complete known normalized input must match')
            expected_digest.update(expected.tobytes());actual_digest.update(actual.tobytes())
        saved = RUN / (name + '-scaler.npz')
        np.savez(saved, **scaler)
        row = dict(partition=name, outer_fold=0, eligible_rows=int(mask.sum()),
                   scored_rows=len(x), columns=x.shape[1], scaler_sha256=u.sha(saved),
                   original_normalized_sha256=expected_digest.hexdigest(),
                   compact_normalized_sha256=actual_digest.hexdigest(),
                   original_complete_statistics_bitwise_equal=True,
                   every_actual_normalized_float32_bit_equal=True,
                   seconds=time.monotonic()-start)
        records.append(row)
        u.write(REPORT / (name + '-verification.json'), row)
        print(name, 'full actual statistics and every row match', flush=True)
    u.write(REPORT / 'verification.json', dict(state='complete_real_data_bitwise_input_and_scaler_parity',
            observed=stamp(), plan_sha256=u.sha(REPORT / 'plan.json'),
            cache_verification_sha256=u.sha(REPORT / 'cache-verification.json'), records=records,
            rows=1128164, columns=400, models_fitted=0,
            original_model_or_budget_changed=False, qualifying_labels_read=False,
            scope='Exact transformation proof on all real rows, original complete outer-fold0 inner/full fit sets and all400 input columns. This is not a reduced training study, a model score, a full model-trajectory parity claim, or a qualifying audit.'))


def run():
    build()
    u.write(REPORT / 'status.json', dict(state='waiting_for_complete_raw_gpu_epoch_before_full_reference_transform', observed=stamp()))
    with (u.ROOT / 'experiments/upstream-source-extension/run.lock').open('a') as slot:
        fcntl.flock(slot, fcntl.LOCK_EX)
        u.write(REPORT / 'held.json', dict(state='shared_epoch_slot_reserved_for_full_reference_transform_only', observed=stamp()))
        with (REPORT / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
                    cwd=u.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            high = 0
            try:
                while child.poll() is None:
                    own = resource.model.base.preceding.group_rss(child.pid)
                    other = resource.model.base.preceding.resources.other_rss({child.pid})
                    free = shutil.disk_usage(u.ROOT).free
                    high = high + 1 if own + other > 24 * 2**30 else 0
                    u.write(REPORT / 'status.json', dict(state='checking_complete_actual_input_and_scaler_parity',
                            observed=stamp(), pid=child.pid, own_rss_bytes=own, other_rss_bytes=other, disk_free_bytes=free))
                    if high >= 3 or free < 8 * 2**30:
                        raise RuntimeError('Original resource guard; parity artifacts preserved')
                    time.sleep(5)
            except BaseException:
                resource.model.base.preceding.stop(child)
                raise
            if child.returncode:
                raise RuntimeError('Actual-data parity failed; original reference/evidence preserved')
    u.write(REPORT / 'released.json', dict(state='shared_epoch_slot_released_after_full_reference_transform', observed=stamp()))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--worker', action='store_true');a = p.parse_args()
    try:
        worker() if a.worker else run()
    except BaseException as error:
        REPORT.mkdir(parents=True, exist_ok=True)
        u.write(REPORT / 'failure.json', dict(state='parity_incomplete_preserved', observed=stamp(), error=repr(error)))
        raise
