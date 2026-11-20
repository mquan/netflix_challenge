#!/usr/bin/env python3
"""Reserve an idle GPU epoch slot while full original CPU KNN3 chains run."""
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'experiments/upstream-knn3-source-slot'
NATIVE = ROOT / 'experiments/upstream-split3-knn3-chains'
RAW = ROOT / 'artifacts/upstream-mps-item-autorec'
JOBS = ['s3-dnn32nm__knn3', 's3-nade12s__knn3']


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def processes():
    rows = subprocess.check_output(['ps', '-axo', 'pid=,stat=,command='], text=True)
    native, producers, raw = {}, {}, {}
    for row in rows.splitlines():
        parts = row.strip().split(None, 2)
        if len(parts) != 3 or parts[1].startswith('Z'):
            continue
        pid, _, command = parts
        args = command.split()
        if len(args) >= 3 and Path(args[0]).name == 'fit' and args[1] == 'chain' and args[2] in JOBS:
            native[pid] = command
        names = {Path(arg).name for arg in args}
        if 'upstream_split3_knn3_chains.py' in names:
            producers[pid] = command
        if 'upstream_mps_item_autorec.py' in names and '--worker' in args:
            raw[pid] = command
    return native, producers, raw


def durable_raw_state():
    state = {}
    for phase in ['trainx', 'fulltrain']:
        p = RAW / phase / 'latest.json'
        if not p.exists():
            raise ValueError('Existing complete durable raw trajectory required')
        pointer = json.loads(p.read_text())
        checkpoint = RAW / phase / pointer['checkpoint']
        if not checkpoint.is_file():
            raise ValueError('Durable raw checkpoint absent')
        state[phase] = dict(pointer_sha256=sha(p), epoch=pointer['epoch'],
                            checkpoint=pointer['checkpoint'],
                            declared_checkpoint_sha256=pointer['checkpoint_sha256'])
    return state


def release_reason(now, last_native, native, producers, completed):
    if completed:
        return 'both_original_native_producers_independently_verified'
    if not native and not producers:
        return 'original_native_producer_ended_without_both_proofs'
    if not native and now - last_native >= 20 * 60:
        return 'no_native_admission_for_twenty_minutes_release_raw_slot'
    return None


def run():
    if REPORT.exists():
        raise ValueError('Existing slot reservation evidence preserved')
    native, producers, _ = processes()
    if not producers:
        raise ValueError('The actual original native KNN3 coordinator must be live')
    REPORT.mkdir()
    source = ROOT / 'scripts/upstream_mps_item_autorec.py'
    original_source = sha(source)
    write(REPORT / 'plan.json', dict(created=stamp(), script_sha256=sha(Path(__file__)),
          raw_source_sha256=original_source, native_plan_sha256=sha(NATIVE / 'plan.json'),
          jobs=JOBS, actual_native_producers=producers,
          policy='Acquire the shared GPU lock only at a complete existing raw-epoch boundary. Keep all raw parameters/checkpoints/optimizer/RNG/budgets unchanged. Hold its next epoch while both CPU-only original KNN3 producers use their unchanged native admission/configuration. Release after both independent native proofs, ended producer, or twenty minutes without a native admission. No native/worker kill or source edit.'))
    with (ROOT / 'experiments/upstream-source-extension/run.lock').open('a') as slot:
        write(REPORT / 'status.json', dict(state='waiting_for_complete_raw_epoch_boundary', observed=stamp()))
        fcntl.flock(slot, fcntl.LOCK_EX)
        held = durable_raw_state()
        if processes()[2]:
            raise ValueError('A raw worker is still active despite exclusive epoch slot')
        write(REPORT / 'held.json', dict(state='shared_epoch_slot_reserved_without_raw_worker',
              observed=stamp(), raw_durable_state=held, raw_source_sha256=original_source))
        last_native = time.monotonic()
        try:
            while True:
                native, producers, raw = processes()
                if raw:
                    raise ValueError('Unexpected raw worker while the shared slot is owned')
                if native:
                    last_native = time.monotonic()
                proof_paths = [NATIVE / (name + '-verification.json') for name in JOBS]
                complete = all(p.exists() for p in proof_paths)
                if complete:
                    for path, name in zip(proof_paths, JOBS):
                        proof = json.loads(path.read_text())
                        if (proof['state'] != 'complete_original_two_phase_knn3_chain'
                                or proof['job'] != name
                                or not proof['both_full_native_neighbor_builds_verified']
                                or not proof['original_default_configuration_verified']
                                or proof['qualifying_labels_read'] is not False):
                            raise ValueError('Complete original independent native proof required')
                reason = release_reason(time.monotonic(), last_native, native, producers, complete)
                write(REPORT / 'status.json', dict(state='reserving_raw_epoch_slot_for_original_cpu_knn3',
                      observed=stamp(), native_workers=native, original_producers=producers,
                      seconds_since_last_native=time.monotonic()-last_native,
                      raw_durable_state=held, release_reason=reason))
                if reason:
                    if durable_raw_state() != held or sha(source) != original_source:
                        raise ValueError('Reserved raw trajectory changed')
                    write(REPORT / 'released.json', dict(state='releasing_shared_epoch_slot', observed=stamp(),
                          reason=reason, raw_durable_state=held,
                          raw_checkpoint_pointers_and_source_unchanged=True,
                          raw_full_fifty_epoch_budget_unchanged=True,
                          native_proof_sha256={p.name: sha(p) for p in proof_paths if p.exists()}))
                    return
                time.sleep(15)
        finally:
            fcntl.flock(slot, fcntl.LOCK_UN)


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        if REPORT.exists():
            write(REPORT / 'failure.json', dict(state='slot_released_after_error', observed=stamp(), error=repr(error)))
        raise
