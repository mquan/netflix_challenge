#!/usr/bin/env python3
"""Check a single early historical RBM build without running full-data training."""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

from prepare_production_rbm_r1 import digest, require

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_joint import dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    build_path, output, report_path = (p.resolve() for p in (
        args.builds, args.output_dir, args.report))
    require(not output.exists() and not report_path.exists(), 'Choose new outputs')
    builds = json.loads(build_path.read_text())
    require(len(builds['builds']) == 1, 'Expected one early historical source group')
    build = builds['builds'][0]
    frozen = {build_path: digest(build_path), ROOT / 'tests/test_joint.py': digest(ROOT / 'tests/test_joint.py'),
              ROOT / 'scripts/prepare_production_rbm_r1.py': digest(ROOT / 'scripts/prepare_production_rbm_r1.py'),
              ROOT / 'scripts/build_production_rbm_variants.py': digest(ROOT / 'scripts/build_production_rbm_variants.py'),
              ROOT / 'build/netflix': digest(ROOT / 'build/netflix')}
    for variant in build['variants'].values():
        binary = Path(variant['binary'])
        frozen[binary] = variant['binary_sha256']
        for path, expected in variant['sources'].items():
            frozen[binary.parent / path] = expected
    frozen[Path(build['patch'])] = build['patch_sha256']
    for path, expected in frozen.items():
        require(digest(path) == expected, 'Build input changed')
    output.mkdir(parents=True)
    queries = len(dataset(output / 'base.bin'))
    raw = bytearray((output / 'base.bin').read_bytes())
    users, _, train_count = struct.unpack_from('<IIQ', raw, 8)
    offset = 56 + (users + 1) * 8 + train_count * 5
    results, commands = {}, []
    cases = [('historical-first', 'original', 1, 5, False),
             ('fixed-first', 'fixed', 1, 5, True),
             ('historical-selected', 'original', 6, 5, False),
             ('disabled', 'fixed', 6, 5, False),
             ('unknown', 'fixed', 6, 0, True),
             ('changed', 'fixed', 6, 5, True)]
    for name, variant, epochs, label, fixed in cases:
        data = output / (name + '.bin')
        changed = bytearray(raw)
        for index in range(queries):
            changed[offset + index * 12 + 8] = label
        data.write_bytes(changed)
        model, pred = output / (name + '.model'), output / (name + '.f32')
        command = ['nice', '-n', '10', build['variants'][variant]['binary'], 'rbm',
                   '--data', str(data), '--model', str(model), '--predictions', str(pred),
                   '--hidden', '4', '--iterations', str(epochs), '--batch', '20',
                   '--threads', '2', '--patience', '1']
        if fixed:
            command.append('--fixed-epoch')
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        (output / (name + '.log')).write_text(result.stdout + result.stderr)
        commands.append(command)
        if fixed:
            require([int(x) for x in re.findall(r'^epoch=(\d+) ', result.stdout, re.M)] == list(range(1, epochs + 1)),
                    'Wrong training epoch sequence')
            require(struct.unpack_from('<I', model.read_bytes(), 20)[0] == epochs, 'Wrong saved epoch')
        predictions = struct.unpack('<' + str(queries) + 'f', pred.read_bytes())
        require(all(float('-inf') < x < float('inf') for x in predictions), 'Nonfinite predictions')
        results[name] = model.read_bytes(), pred.read_bytes()
    require(results['historical-first'] == results['fixed-first'], 'First-epoch training changed')
    require(results['historical-selected'] == results['disabled'], 'Disabled patch changed behavior')
    require(results['unknown'][0][16:] == results['changed'][0][16:] and
            results['unknown'][1] == results['changed'][1], 'Query labels changed training')
    reload_path = output / 'reload.f32'
    command = [build['variants']['fixed']['binary'], 'eval-rbm', '--data', str(output / 'unknown.bin'),
               '--model', str(output / 'unknown.model'), '--predictions', str(reload_path), '--threads', '2']
    subprocess.run(command, capture_output=True, text=True, check=True)
    commands.append(command)
    require(reload_path.read_bytes() == results['unknown'][1], 'Reload changed predictions')
    for path, expected in frozen.items():
        require(digest(path) == expected, 'Build input changed during verification')
    report = dict(state='early_RBM_fixture_contracts_verified', stages=build['stages'],
                  verifier_sha256=digest(Path(__file__)), frozen_inputs={str(p): h for p, h in frozen.items()},
                  checks=dict(first_epoch_parity=True, disabled_patch_parity=True, final_epoch=6,
                              query_labels=[0, 5], label_isolation=True, reload_parity=True,
                              shared_binary_unchanged=True), commands=commands,
                  artifacts={str(p): digest(p) for p in sorted(output.iterdir())},
                  limitations=['Small two-thread fixtures; full-data retraining and historical eight-thread runs remain unverified.'])
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(state=report['state'], commands=len(commands))))


if __name__ == '__main__':
    main()
