#!/usr/bin/env python3
"""Verify isolated RBM builds on small fixtures, without selecting any model."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_joint import dataset  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    build_path = args.builds.resolve()
    build_sha = digest(build_path)
    builds = json.loads(build_path.read_text())
    output = args.output_dir.resolve()
    report_path = args.report.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError('Choose new output paths')
    output.mkdir(parents=True)
    frozen = {build_path: build_sha, ROOT / 'tests/test_joint.py': digest(ROOT / 'tests/test_joint.py')}
    for build in builds['builds']:
        for variant in build['variants'].values():
            binary = Path(variant['binary'])
            frozen[binary] = variant['binary_sha256']
            for source, expected in variant['sources'].items():
                frozen[binary.parent / source] = expected
        frozen[Path(build['patch'])] = build['patch_sha256']
    for path, expected in frozen.items():
        require(digest(path) == expected, 'Build input changed: ' + str(path))
    query_count = len(dataset(output / 'base.bin'))
    data_paths = {}
    raw = bytearray((output / 'base.bin').read_bytes())
    # Header + user offsets + train item/day/rating arrays precede probe records.
    users, _, train_count = struct.unpack_from('<IIQ', raw, 8)
    probe_offset = 56 + (users + 1) * 8 + train_count * 5
    for label in (0, 5):
        changed = bytearray(raw)
        for index in range(query_count):
            changed[probe_offset + index * 12 + 8] = label
        path = output / ('labels-' + str(label) + '.bin')
        path.write_bytes(changed)
        data_paths[label] = path
    commands, checks = [], []

    def run(binary, name, data, epochs, fixed=True, conditional=False, init=None):
        prefix = output / name
        command = ['nice', '-n', '10', str(binary), 'rbm', '--data', str(data),
                   '--model', str(prefix.with_suffix('.model')),
                   '--predictions', str(prefix.with_suffix('.f32')),
                   '--hidden', '4', '--iterations', str(epochs), '--batch', '20',
                   '--threads', '2', '--patience', '1']
        if fixed:
            command.append('--fixed-epoch')
        if conditional:
            command.append('--conditional-biases')
        start_epoch = 1
        if init:
            start_epoch = struct.unpack_from('<I', init.read_bytes(), 20)[0] + 1
            command.extend(['--init-model', str(init), '--start-epoch', str(start_epoch)])
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        prefix.with_suffix('.log').write_text(result.stdout + result.stderr)
        commands.append(command)
        if fixed:
            expected_epochs = list(range(start_epoch, epochs + 1))
            require([int(x) for x in re.findall(r'^epoch=(\d+) ', result.stdout, re.M)] == expected_epochs,
                    'Fixed-budget epoch log mismatch')
            require(struct.unpack_from('<I', prefix.with_suffix('.model').read_bytes(), 20)[0] == epochs,
                    'Wrong final checkpoint epoch')
        payload = prefix.with_suffix('.f32').read_bytes()
        values = struct.unpack('<' + str(query_count) + 'f', payload)
        require(all(float('-inf') < x < float('inf') for x in values), 'Nonfinite predictions')
        return prefix

    def equal(a, b, ignore_fingerprint=False):
        offset = 16 if ignore_fingerprint else 0
        require(a.with_suffix('.model').read_bytes()[offset:] == b.with_suffix('.model').read_bytes()[offset:],
                'Model parameters differ: ' + a.name + ' / ' + b.name)
        require(a.with_suffix('.f32').read_bytes() == b.with_suffix('.f32').read_bytes(),
                'Predictions differ: ' + a.name + ' / ' + b.name)

    for build in builds['builds']:
        index = build['index']
        original = Path(build['variants']['original']['binary'])
        fixed = Path(build['variants']['fixed']['binary'])
        source = (fixed.parent / 'src/rbm.hpp').read_text()
        for conditional in (False, True) if '--conditional-biases' in source else (False,):
            name = f'v{index}-c{int(conditional)}'
            a = run(original, name + '-historical', data_paths[5], 1, fixed=False, conditional=conditional)
            b = run(fixed, name + '-first', data_paths[5], 1, conditional=conditional)
            equal(a, b)
            a = run(original, name + '-historical-selected', data_paths[5], 6, fixed=False, conditional=conditional)
            b = run(fixed, name + '-disabled', data_paths[5], 6, fixed=False, conditional=conditional)
            equal(a, b)
            results = [run(fixed, name + '-label' + str(label), data_paths[label], 6,
                           conditional=conditional) for label in (0, 5)]
            equal(*results, ignore_fingerprint=True)
            reload_path = output / (name + '-reload.f32')
            command = [str(fixed), 'eval-rbm', '--data', str(data_paths[5]),
                       '--model', str(results[1].with_suffix('.model')),
                       '--predictions', str(reload_path), '--threads', '2']
            subprocess.run(command, capture_output=True, text=True, check=True)
            commands.append(command)
            require(reload_path.read_bytes() == results[1].with_suffix('.f32').read_bytes(), 'Reload mismatch')
            checks.append(dict(build=index, conditional=conditional, first_epoch_parity=True,
                               disabled_patch_parity=True, final_epoch=6, label_isolation=True, reload_parity=True))
    # Exercise the historical standard -> resume -> conditional -> later-format chain.
    parents = {}
    chain = [0, 1, 3, 2]
    by_index = {b['index']: b for b in builds['builds']}
    require(set(by_index) == set(chain), 'Revalidate resume-chain mapping for changed build groups')
    for position, index in enumerate(chain):
        binary = Path(by_index[index]['variants']['fixed']['binary'])
        conditional = position >= 2
        results = []
        for label in (0, 5):
            prefix = run(binary, f'chain-{position}-label{label}', data_paths[label],
                         2 * (position + 1), conditional=conditional,
                         init=None if position == 0 else parents[label])
            results.append(prefix)
        equal(*results, ignore_fingerprint=True)
        parents = {label: prefix.with_suffix('.model') for label, prefix in zip((0, 5), results)}
        checks.append(dict(cross_version_build=index, parent_build=chain[position - 1] if position else None,
                           conditional=conditional, resume_label_isolation=True))
    for path, expected in frozen.items():
        require(digest(path) == expected, 'Build changed during verification: ' + str(path))
    report = dict(state='fixture_contracts_verified', builds=str(build_path), builds_sha256=build_sha,
                  verifier_sha256=digest(Path(__file__)), fixture_generator_sha256=frozen[ROOT / 'tests/test_joint.py'],
                  users=users, query_count=query_count, threads=2, checks=checks, commands=commands,
                  artifacts={str(p): digest(p) for p in sorted(output.iterdir())},
                  limitations=['Small fixtures verify save/stop, training parity, label isolation, reload and resume contracts.',
                               'Full-data retraining, historical production thread counts and score quality are not verified.',
                               'R1/R2 missing supporting sources remain unresolved.'])
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(state=report['state'], checks=len(checks), commands=len(commands))))


if __name__ == '__main__':
    main()
