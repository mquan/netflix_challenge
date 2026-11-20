#!/usr/bin/env python3
"""Independently admit an exact compiled preset and later its complete outputs."""
import argparse
import re
import subprocess
from pathlib import Path
import numpy as np
import upstream_repro as u

REPORT = u.ROOT / 'experiments/upstream-high-capacity-original'
TREE = u.ROOT / 'artifacts/upstream-high-capacity-original/source'


def admit(name):
    plan = u.read(REPORT / 'execution-plan.json')
    if plan['source_pin'] != u.PIN or plan['state'] != 'complete_isolated_original_preparation_no_training_launched':
        raise ValueError('Complete exact original preparation required')
    for rel, digest in plan['bindings'].items():
        if u.sha(u.ROOT / rel) != digest:
            raise ValueError('Original complete source/data/binary changed: ' + rel)
    if subprocess.check_output(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], text=True).strip() != u.PIN:
        raise ValueError('Original isolated source pin differs')
    changes = subprocess.check_output(['git', '-C', str(TREE), 'diff', 'HEAD', '--name-only'], text=True).splitlines()
    if set(changes) - {'data/qual_ratings/qual_ratings.csv.gz'}:
        raise ValueError('Original model/config source changed')
    job = plan['jobs'][name]
    if job['original_registry_job'] != u.read(u.BASE / 'registry.json')['jobs'][name]:
        raise ValueError('Original registry recipe changed')
    build_path = REPORT / 'original-build-receipt.json'
    if not build_path.exists():
        raise ValueError('Fresh isolated source-pinned binary build required; earlier shared binary receipt does not match')
    build = u.read(build_path)
    binary = u.ROOT / 'artifacts/upstream-high-capacity-original/target/release/tsvdx4-new'
    if (build['source_commit'] != u.PIN or
        build['binaries'].get(binary.name) != u.sha(binary) or
        build['binary_path'] != str(binary) or
        build['preparation_sha256'] != u.sha(REPORT / 'execution-plan.json') or
        build['builder_sha256'] != u.sha(u.ROOT / 'scripts/build_upstream_high_capacity_original.py') or
        not build['original_source_and_complete_data_unchanged'] or
        build['training_launched'] is not False):
        raise ValueError('Exact source-pinned compiled binary lacks matching original build receipt')
    if name != 'tsvdx4-1600lm' or job['factors'] != 1600 or job['epochs_per_phase'] != 2:
        raise ValueError('This first execution admits only the exact original1600/2 preset')
    if np.any(np.load(TREE / 'data/qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying fitting targets contain outcomes')
    fresh = u.read(u.BASE / 'fresh-training-plan.json')
    if u.sha(TREE / 'data/qual_ratings/qual_ratings.csv.gz') != fresh['synthetic_qualifying_payload_sha256']:
        raise ValueError('Synthetic qualifying CSV differs')
    expected = [('trainx', 'probex', 99352343, 1128164),
                ('fulltrain', 'qual', 100480507, 2817131)]
    if len(plan['full_identity_disjointness']) != len(expected):
        raise ValueError('Both complete original identity scans required')
    for receipt, (train, query, nt, nq) in zip(plan['full_identity_disjointness'], expected):
        if receipt != dict(training_rows=nt, query_rows=nq, all_training_identities_scanned=True, overlap=0):
            raise ValueError('Complete original identity-disjointness receipt differs')
        for split, count in [(train, nt), (query, nq)]:
            for key in ['user_idxs', 'item_idxs', 'ratings', 'dates']:
                if np.load(TREE / f'data/{split}/{key}.npy', mmap_mode='r').shape != (count,):
                    raise ValueError('Full original dataset shape differs')
    result = dict(state='complete_original1600_source_data_and_matching_compiled_binary_admission',
                  preparation_sha256=u.sha(REPORT / 'execution-plan.json'),
                  original_build_receipt_sha256=u.sha(build_path),
                  verifier_sha256=u.sha(Path(__file__)), binary_sha256=u.sha(binary),
                  native_command=[str(binary), name], cwd=str(TREE),
                  qualifying_targets_zero=True, qualifying_labels_read=False,
                  scope='Full bound inputs independently rehashed; source pin/config and matching original build receipt checked. All original identity-scan receipts bound to unchanged complete arrays. No fitting or score.')
    return result


def verify_complete(name):
    admission = admit(name)
    run = u.read(REPORT / (name + '-run.json'))
    plan = u.read(REPORT / 'execution-plan.json')
    job = plan['jobs'][name]
    if run['command'] != admission['native_command'] or run['cwd'] != str(TREE) or run['returncode'] != 0:
        raise ValueError('Complete original native execution required')
    text = (REPORT / (name + '.log')).read_text()
    trace = [int(v) for v in re.findall(r'^Epoch\s+(\d+)\s*\|', text, re.M)]
    if trace != [1, 2, 1, 2] or 'trainx => probex' not in text or 'fulltrain => qual' not in text:
        raise ValueError('Both complete original epoch traces required')
    out = TREE / 'preds_new'
    expected = {f'{name}.{split}.npy' for split in ['probex', 'qual']}
    expected |= {f'{name}_ep{ep:02}.{split}.npy' for ep in [1, 2] for split in ['probex', 'qual']}
    actual = {p.name for p in out.glob('*.npy')}
    if actual != expected:
        raise ValueError('Complete original query output population differs')
    outputs = {}
    for filename in sorted(expected):
        path = out / filename
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        rows = 1128164 if '.probex.' in filename else 2817131
        if value.shape != (rows,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Full finite original query vector required')
        outputs[str(path.relative_to(u.ROOT))] = u.sha(path)
    native_log = out / (name + '.out')
    if not set(job['original_registry_job']['outputs']).issubset({str(Path(p).relative_to(TREE.relative_to(u.ROOT))) for p in outputs} | {str(native_log.relative_to(TREE))}):
        raise ValueError('Original declared outputs missing')
    outputs[str(native_log.relative_to(u.ROOT))] = u.sha(native_log)
    result = dict(state='complete_original1600_both_full_phases_and_all_native_query_vectors',
                  job=name, epochs_per_phase=2, factors=1600,
                  admission=admission, original_run_sha256=u.sha(REPORT / (name + '-run.json')),
                  complete_epoch_log_sha256=u.sha(REPORT / (name + '.log')),
                  outputs=outputs, qualifying_labels_read=False,
                  scope='Fresh original source training, complete canonical/full qualifying vectors; no blend fitting, qualifying audit or score promotion.')
    u.write(REPORT / (name + '-verification.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    name = 'tsvdx4-1600lm'
    result = verify_complete(name) if args.complete else admit(name)
    if not args.complete:
        u.write(REPORT / 'independent-admission.json', result)
    print(result['state'], flush=True)
