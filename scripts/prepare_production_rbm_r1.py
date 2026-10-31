#!/usr/bin/env python3
"""Recover R1's exact uncommitted command source, then build and check it."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

from build_production_rbm_variants import fixed_source

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
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    output, source_archive, report_path = (p.resolve() for p in (
        args.output_dir, args.source_archive, args.report))
    require(not any(p.exists() for p in (output, source_archive, report_path)), 'Choose new outputs')
    recovery_path = ROOT / 'experiments/P0-rbm-source-recovery-20261031.json'
    recovery = json.loads(recovery_path.read_text())
    registry_path = Path(recovery['registry'])
    historical_archive = Path(recovery['archive'])
    frozen = {recovery_path: digest(recovery_path), registry_path: recovery['registry_sha256'],
              historical_archive: recovery['archive_sha256'],
              ROOT / 'scripts/build_production_rbm_variants.py': digest(ROOT / 'scripts/build_production_rbm_variants.py'),
              ROOT / 'tests/test_joint.py': digest(ROOT / 'tests/test_joint.py'),
              ROOT / 'build/netflix': digest(ROOT / 'build/netflix')}
    for path, expected in frozen.items():
        require(digest(path) == expected, 'Frozen input mismatch')
    registry = json.loads(registry_path.read_text())
    r1 = next(s for s in registry['stages'] if s['name'] == 'R1-rbm100')
    r3 = next(s for s in registry['stages'] if s['name'] == 'R3-rbm200-long')
    found = {(s['path'], s['sha256']): s for s in recovery['sources']}
    payloads = {}
    with zipfile.ZipFile(historical_archive) as archive:
        base_sha = r3['required_historical_sources']['src/netflix.cpp']
        base = archive.read(found['src/netflix.cpp', base_sha]['archive_member'])
        removed = [
            b'      << "  netflix knn --data FILE --graph FILE --time-model FILE "\n'
            b'         "--predictions FILE [--support FILE] [--neighbors N] [--power X] [--shrink X]\\n"\n',
            b'    } else if (command == "knn") {\n      command_knn(args);\n',
        ]
        reconstructed = base
        for block in removed:
            require(reconstructed.count(block) == 1, 'Unexpected R3 command source')
            reconstructed = reconstructed.replace(block, b'')
        for path, expected in r1['required_historical_sources'].items():
            payload = reconstructed if path == 'src/netflix.cpp' else archive.read(found[path, expected]['archive_member'])
            require(hashlib.sha256(payload).hexdigest() == expected, 'Reconstruction is not an exact historical match')
            payloads[path] = payload
    with zipfile.ZipFile(source_archive, 'x') as archive:
        for path, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    with zipfile.ZipFile(source_archive) as archive:
        require(all(archive.read(p) == v for p, v in payloads.items()), 'Archive round-trip failed')
    output.mkdir(parents=True)
    builds = {}
    for variant in ('original', 'fixed'):
        directory = output / variant
        (directory / 'src').mkdir(parents=True)
        for path, payload in payloads.items():
            if variant == 'fixed' and path == 'src/rbm.hpp':
                payload = fixed_source(payload.decode()).encode()
            (directory / path).write_bytes(payload)
        binary = directory / 'netflix'
        command = ['nice', '-n', '10', 'clang++', '-O3', '-DNDEBUG', '-std=c++17',
                   '-Wall', '-Wextra', '-Wpedantic', str(directory / 'src/netflix.cpp'),
                   '-framework', 'Accelerate', '-o', str(binary)]
        with (directory / 'build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        builds[variant] = dict(binary=str(binary), binary_sha256=digest(binary), command=command,
                               sources={p: digest(directory / p) for p in payloads})
        print(json.dumps(dict(stage='R1-rbm100', built=variant)), flush=True)
    fixtures = output / 'fixtures'
    fixtures.mkdir()
    queries = len(dataset(fixtures / 'base.bin'))
    raw = bytearray((fixtures / 'base.bin').read_bytes())
    users, _, train_count = struct.unpack_from('<IIQ', raw, 8)
    offset = 56 + (users + 1) * 8 + train_count * 5
    commands = []
    results = {}
    for name, variant, epochs, label, fixed in (
            ('historical-first', 'original', 1, 5, False),
            ('fixed-first', 'fixed', 1, 5, True),
            ('historical-selected', 'original', 6, 5, False),
            ('disabled', 'fixed', 6, 5, False),
            ('unknown', 'fixed', 6, 0, True),
            ('changed', 'fixed', 6, 5, True)):
        data = fixtures / (name + '.bin')
        changed = bytearray(raw)
        for index in range(queries):
            changed[offset + index * 12 + 8] = label
        data.write_bytes(changed)
        model, pred = fixtures / (name + '.model'), fixtures / (name + '.f32')
        command = ['nice', '-n', '10', builds[variant]['binary'], 'rbm', '--data', str(data),
                   '--model', str(model), '--predictions', str(pred), '--hidden', '4',
                   '--iterations', str(epochs), '--batch', '20', '--threads', '2', '--patience', '1']
        if fixed:
            command.append('--fixed-epoch')
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        (fixtures / (name + '.log')).write_text(result.stdout + result.stderr)
        commands.append(command)
        if fixed:
            require(struct.unpack_from('<I', model.read_bytes(), 20)[0] == epochs, 'Wrong saved epoch')
        predictions = struct.unpack('<' + str(queries) + 'f', pred.read_bytes())
        require(all(float('-inf') < x < float('inf') for x in predictions), 'Nonfinite predictions')
        results[name] = (model.read_bytes(), pred.read_bytes())
    require(results['historical-first'] == results['fixed-first'], 'First-epoch training changed')
    require(results['historical-selected'] == results['disabled'], 'Disabled patch changed historical behavior')
    require(results['unknown'][0][16:] == results['changed'][0][16:] and
            results['unknown'][1] == results['changed'][1], 'Query labels changed training')
    reload_path = fixtures / 'reload.f32'
    command = [builds['fixed']['binary'], 'eval-rbm', '--data', str(fixtures / 'unknown.bin'),
               '--model', str(fixtures / 'unknown.model'), '--predictions', str(reload_path), '--threads', '2']
    subprocess.run(command, capture_output=True, text=True, check=True)
    commands.append(command)
    require(reload_path.read_bytes() == results['unknown'][1], 'Reload changed predictions')
    for path, expected in frozen.items():
        require(digest(path) == expected, 'Frozen input changed during work')
    report = dict(state='exact_R1_source_reconstructed_built_fixture_verified',
                  generator_sha256=digest(Path(__file__)), frozen_inputs={str(p): h for p, h in frozen.items()},
                  source_archive=str(source_archive), source_archive_sha256=digest(source_archive),
                  historical_sources=r1['required_historical_sources'],
                  reconstruction=dict(base_cpp_sha256=base_sha, removed_blocks=[x.decode() for x in removed],
                                      recovered_cpp_sha256=hashlib.sha256(reconstructed).hexdigest(),
                                      expected_sha256=r1['required_historical_sources']['src/netflix.cpp']),
                  builds=builds, fixture_commands=commands,
                  checks=dict(first_epoch_parity=True, disabled_patch_parity=True, final_epoch=6,
                              query_labels=[0, 5], label_isolation=True, reload_parity=True,
                              shared_binary_unchanged=True),
                  artifacts={str(p): digest(p) for p in sorted(fixtures.iterdir())},
                  limitations=['Fixtures use two threads; production must preserve the historical eight threads.',
                               'No full production training performed. R2 neighborhood source remains missing.'])
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(state=report['state'], checks=report['checks'])))


if __name__ == '__main__':
    main()
