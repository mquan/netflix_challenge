#!/usr/bin/env python3
"""Verify completed CFNADE48 resumed phases and export terminal vectors losslessly.

Usage: python -I verify_export_cf48.py --package P --data-root D --output O
Checks per phase: final latest.json -> epoch-00095/state.json md5, identity, all 84
tensor MD5s; all 95 epoch snapshots present, 1-D float32, finite, expected rows;
terminal vector byte-equal to ep95. Exports terminal as NPZ key `prediction`.
Refuses to export on any failure (exit 1) and writes the failure list.
"""
import argparse, hashlib, json, sys, datetime as dt
from pathlib import Path
import numpy as np

EXPECT = {'trainx': ('probex', 1128164), 'fulltrain': ('qual', 2817131)}
SOURCE = 'e14530b5ade496efb3980a64f98442d35c9a606960add79149ce67b057677d94'
DATA = '55ce479fc16447a826d013288b2ccb3e1eec39a07d7783f3d860a3f6e33e1fe0'


def h(path, algo='sha256'):
    d = hashlib.new(algo)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            d.update(chunk)
    return d.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    preds = a.package / 'native-source/preds_preset48'
    a.output.mkdir(parents=True, exist_ok=True)
    failures, report = [], {'observed': dt.datetime.now(dt.timezone.utc).isoformat(), 'phases': {}}
    for train, (query, rows) in EXPECT.items():
        ph = {'query': query, 'rows': rows}
        root = a.package / 'checkpoints' / train
        latest = json.loads((root / 'latest.json').read_text())
        ph['final_latest'] = latest
        if latest['epoch'] != 95:
            failures.append(f'{train}: latest epoch {latest["epoch"]} != 95')
        state_path = root / f'epoch-{latest["epoch"]:05}' / 'state.json'
        if h(state_path, 'md5') != latest['manifest_md5']:
            failures.append(f'{train}: state.json md5 mismatch')
        state = json.loads(state_path.read_text())
        ident = state['identity']
        if (ident['source'], ident['data'], ident['train'], ident['query'], ident['query_rows']) != (SOURCE, DATA, train, query, rows):
            failures.append(f'{train}: identity mismatch {ident}')
        files = state['files']
        if len(files) != 84:
            failures.append(f'{train}: {len(files)} tensors != 84')
        bad = [n for n, m in files.items() if h(state_path.parent / n, 'md5') != m]
        if bad:
            failures.append(f'{train}: tensor md5 mismatch {bad}')
        ph.update(checkpoint_state_md5=latest['manifest_md5'], tensors_verified=len(files) - len(bad),
                  identity=ident, timestep=state['timestep'], swa_count=state['swa_count'])
        snaps = {}
        for e in range(1, 96):
            p = preds / f'cfnade-48_ep{e:02}.{query}.npy'
            if not p.exists():
                failures.append(f'{train}: missing {p.name}'); continue
            v = np.load(p, allow_pickle=False)
            if v.dtype != np.float32 or v.shape != (rows,) or not np.isfinite(v).all():
                failures.append(f'{train}: bad snapshot {p.name} {v.dtype} {v.shape}')
            snaps[p.name] = h(p)
        ph['snapshot_sha256'] = snaps
        term = preds / f'cfnade-48.{query}.npy'
        if not term.exists() or not (preds / f'cfnade-48_ep95.{query}.npy').exists():
            failures.append(f'{train}: terminal or ep95 vector absent')
            report['phases'][train] = ph
            continue
        t = np.load(term, allow_pickle=False)
        ep95 = np.load(preds / f'cfnade-48_ep95.{query}.npy', allow_pickle=False)
        if t.dtype != np.float32 or t.shape != (rows,) or not np.isfinite(t).all():
            failures.append(f'{train}: bad terminal {t.dtype} {t.shape}')
        if t.tobytes() != ep95.tobytes():
            failures.append(f'{train}: terminal != ep95')
        ph['terminal_npy_sha256'] = h(term)
        ph['other_outputs_sha256'] = {p.name: h(p) for p in sorted(preds.glob('cfnade-48.*')) if p.name != term.name}
        ph['row_identity_sha256'] = {p.name: h(p) for p in sorted((a.data_root / query).glob('*.npy'))}
        if not failures:
            npz = a.output / f'cfnade-48.{query}.npz'
            np.savez_compressed(npz, prediction=t)
            back = np.load(npz)['prediction']
            assert back.tobytes() == t.tobytes()
            ph.update(npz=npz.name, npz_sha256=h(npz),
                      prediction_bytes_sha256=hashlib.sha256(t.tobytes()).hexdigest())
        report['phases'][train] = ph
    report['failures'] = failures
    report['state'] = 'verified_complete_resumed_cfnade48' if not failures else 'verification_failed'
    (a.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'state': report['state'], 'failures': failures}))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
