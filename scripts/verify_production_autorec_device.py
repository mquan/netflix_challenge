"""Verify AutoRec capture/reload parity on an explicitly selected device."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from production_rbm_audit import digest, verify, write_new

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_joint import dataset  # noqa: E402
from test_native_fixed_epochs import query_labels  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--device', required=True, choices=['cpu', 'mps'])
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS verification requested but unavailable; no CPU fallback')
    source, output, report = args.source_dir.resolve(), args.output_dir.resolve(), args.report.resolve()
    if output.exists() or report.exists():
        raise FileExistsError('Choose new verification paths')
    manifest = json.loads((source / 'source-manifest.json').read_text())
    source_files = {'autorec.py': manifest['source_sha256'],
                    'original-autorec.py': manifest['original_source_sha256'],
                    'gbdt_blend.py': manifest['loader_sha256'],
                    'checkpoint-and-reload.patch': manifest['patch_sha256']}
    verify({str(source / name): expected for name, expected in source_files.items()})
    inputs = {str(p): digest(p) for p in [*(source / name for name in source_files),
        source / 'source-manifest.json', Path(__file__),
        ROOT / 'scripts/production_autorec_source.py', ROOT / 'tests/test_joint.py', ROOT / 'tests/test_native_fixed_epochs.py']}
    output.mkdir(parents=True)
    data = {}
    for label in (0, 5):
        data[label] = output / f'data-{label}.bin'
        dataset(data[label], correlated_train=True)
        query_labels(data[label], label)
    commands = []

    def run(name, script, label=0, checkpoint=None):
        target = output / (name + '.f32')
        command = [sys.executable, str(ROOT / 'scripts/production_autorec_source.py'), 'run',
                   '--source', str(script), '--threads', '1', '--', '--data', str(data[label]),
                   '--predictions', str(target), '--hidden', '8', '--layers', '2', '--batch', '32',
                   '--epochs', '4', '--fixed-epoch', '--save-epoch', '2', '--device', args.device]
        if checkpoint is not None:
            command += ['--reload-checkpoint', str(checkpoint)]
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        commands.append(command)
        if target.stat().st_size != 200 * 4 or not np.isfinite(np.fromfile(target, '<f4')).all():
            raise ValueError('Invalid device-parity fixture predictions')
        return target

    original = run('original', source / 'original-autorec.py')
    captured = run('captured', source / 'autorec.py')
    for suffix in ('.f32', '.e2.f32'):
        if original.with_suffix(suffix).read_bytes() != captured.with_suffix(suffix).read_bytes():
            raise ValueError('Checkpoint instrumentation changed device predictions')
    if captured.read_bytes() == captured.with_suffix('.e2.f32').read_bytes():
        raise ValueError('Fixture did not distinguish final and snapshot epochs')
    final = run('reload-final', source / 'autorec.py', checkpoint=captured.with_suffix('.pt'))
    snapshot = run('reload-snapshot', source / 'autorec.py', checkpoint=captured.with_suffix('.e2.pt'))
    if final.read_bytes() != captured.read_bytes() or snapshot.read_bytes() != captured.with_suffix('.e2.f32').read_bytes():
        raise ValueError('Checkpoint reload changed device predictions')
    changed = run('changed-labels', source / 'autorec.py', label=5)
    if changed.read_bytes() != captured.read_bytes():
        raise ValueError('Query labels changed AutoRec device predictions')
    a, b = [torch.load(p.with_suffix('.pt'), map_location='cpu', weights_only=True) for p in (captured, changed)]
    if set(a['state']) != set(b['state']) or any(not torch.equal(a['state'][key], b['state'][key]) for key in a['state']):
        raise ValueError('Query labels changed AutoRec device weights')
    verify(inputs)
    result = dict(state='verified_autorec_device_parity', device=args.device, host_threads=1,
                  source_sha256=manifest['source_sha256'], torch_version=str(torch.__version__), numpy_version=np.__version__,
                  inputs=inputs, outputs={str(p): digest(p) for p in sorted(output.iterdir())}, commands=commands,
                  checks=dict(original_final_and_snapshot_parity=True, final_and_snapshot_reload_parity=True,
                              query_label_independent_predictions_and_weights=True, distinct_snapshot_and_final=True),
                  scope='Tiny device-specific engineering parity only. No production trajectory trained and no score gain claimed.')
    write_new(report, result)
    print(json.dumps(dict(state=result['state'], device=args.device, commands=len(commands))))


if __name__ == '__main__':
    main()
