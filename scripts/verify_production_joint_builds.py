"""Check historical joint/integrated learning, label isolation and resume sidecars."""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np

from production_rbm_audit import digest, verify

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_joint import dataset  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    build_path, output, report_path = (p.resolve() for p in (args.builds, args.output_dir, args.report))
    require(not output.exists() and not report_path.exists(), 'Choose new outputs')
    builds = json.loads(build_path.read_text())
    registry = json.loads(Path(builds['registry']).read_text())
    frozen = dict(builds['frozen_inputs'])
    frozen.update({str(build_path): digest(build_path), str(ROOT / 'tests/test_joint.py'): digest(ROOT / 'tests/test_joint.py'),
                   builds['archive']: builds['archive_sha256']})
    legacy = builds['retained_legacy']
    frozen[legacy['binary']] = legacy['binary_sha256']
    for source, expected in legacy['sources'].items():
        frozen[str(Path(legacy['source_root']) / source)] = expected
    for build in builds['builds']:
        frozen.update(build['patches'])
        for variant in build['variants'].values():
            frozen[variant['binary']] = variant['binary_sha256']
            for source, expected in variant['sources'].items():
                frozen[str(Path(variant['binary']).parent / source)] = expected
    verify(frozen)
    # Check that the full-data legacy proofs refer to the actual retained recipes.
    j1 = json.loads((ROOT / 'experiments/PR5-J1-retraining-verification.json').read_text())
    require(j1['byte_identical'] and j1['model_byte_identical'] and
            j1['retrained_model_sha256'] == registry['stages'][0]['model_sha256'] and
            j1['retrained_prediction_sha256'] == registry['stages'][0]['prediction_sha256'], 'J1 retained proof differs')
    proofs = json.loads((ROOT / 'experiments/PR6-legacy-joint-final-verification.json').read_text())
    for stage in registry['stages'][1:4]:
        proof = next(p for p in proofs['stages'] if p['stage'] == stage['name'])
        require(proof['checkpoint_and_prediction_byte_identical'] and
                proof['checkpoint_sha256'] == stage['model_sha256'] and
                proof['prediction_sha256'] == stage['prediction_sha256'], 'Legacy retained proof differs')
    output.mkdir(parents=True)
    queries = len(dataset(output / 'base.bin', correlated_train=True))
    raw = bytearray((output / 'base.bin').read_bytes())
    users, _, ntrain = struct.unpack_from('<IIQ', raw, 8)
    probe_offset = 56 + (users + 1) * 8 + ntrain * 5
    data, graphs = {}, {}
    commands, checks = [], []
    for label in (0, 5):
        changed = bytearray(raw)
        for index in range(queries):
            changed[probe_offset + index * 12 + 8] = label
        data[label] = output / f'data-{label}.bin'
        data[label].write_bytes(changed)
        graphs[label] = output / f'graph-{label}.bin'
        command = [legacy['binary'], 'graph', '--data', str(data[label]), '--graph', str(graphs[label]),
                   '--neighbors', '2', '--history-cap', '256']
        subprocess.run(command, capture_output=True, text=True, check=True)
        commands.append(command)
    require(graphs[0].read_bytes()[16:] == graphs[5].read_bytes()[16:], 'Graph construction reads query labels')

    def run(binary, name, family, label, epochs, fixed=True, day_rank=0, init=None, start=None):
        prefix = output / name
        command = [binary, family, '--data', str(data[label]), '--model', str(prefix.with_suffix('.bin')),
                   '--predictions', str(prefix.with_suffix('.f32')), '--factors', '4', '--iterations', str(epochs),
                   '--learning-rate', '.012', '--regularization', '.015', '--decay', '.93', '--patience', '1', '--threads', '1']
        if fixed:
            command.append('--fixed-epoch')
        if family == 'integrated':
            command += ['--graph', str(graphs[label]), '--neighbor-rate', '.001', '--neighbor-regularization', '.015']
        elif day_rank:
            command += ['--day-factors', str(day_rank)]
        if init:
            command += ['--init-model', str(init.with_suffix('.bin'))]
            if family == 'integrated':
                require(init.with_suffix('.bin.nbr').is_file(), 'Missing production-style resume sidecar')
        if start is not None:
            command += ['--start-epoch', str(start)]
        result = subprocess.run(['nice', '-n', '10', *command], capture_output=True, text=True, check=True)
        prefix.with_suffix('.log').write_text(result.stdout + result.stderr)
        commands.append(command)
        header = struct.unpack('<QQII', prefix.with_suffix('.bin').read_bytes()[:24])
        require(header[2] == 4, 'Wrong factor rank')
        if fixed:
            require(header[3] == epochs, 'Wrong fixed checkpoint epoch')
            require([int(n) for n in re.findall(r'^epoch=(\d+) ', result.stdout, re.M)] == list(range(start or 1, epochs + 1)),
                    'Wrong fixed epoch sequence')
        values = np.fromfile(prefix.with_suffix('.f32'), '<f4')
        require(values.shape == (queries,) and np.isfinite(values).all(), 'Invalid fixture predictions')
        if family == 'integrated':
            require(prefix.with_suffix('.bin.nbr').is_file(), 'Integrated sidecar not saved')
        return prefix

    def equal(a, b, family, ignore_fingerprint=False):
        offset = 16 if ignore_fingerprint else 0
        require(a.with_suffix('.bin').read_bytes()[offset:] == b.with_suffix('.bin').read_bytes()[offset:], 'Checkpoint learning differs')
        require(a.with_suffix('.f32').read_bytes() == b.with_suffix('.f32').read_bytes(), 'Predictions differ')
        if family == 'integrated':
            require(a.with_suffix('.bin.nbr').read_bytes() == b.with_suffix('.bin.nbr').read_bytes(), 'Integrated sidecar differs')

    def reload(binary, prefix, family, label):
        command = [binary, 'eval-' + family, '--data', str(data[label]), '--model', str(prefix.with_suffix('.bin')),
                   '--predictions', str(prefix.with_suffix('.reload.f32')), '--threads', '1']
        if family == 'integrated':
            command += ['--graph', str(graphs[label])]
        subprocess.run(command, capture_output=True, text=True, check=True)
        commands.append(command)
        require(prefix.with_suffix('.f32').read_bytes() == prefix.with_suffix('.reload.f32').read_bytes(), 'Reload changed predictions')

    for build in builds['builds']:
        for family in ('joint', 'integrated') if 'src/integrated.hpp' in build['historical_sources'] else ('joint',):
            name = f"v{build['index']}-{family}"
            original, fixed = [build['variants'][v]['binary'] for v in ('original', 'fixed')]
            a = run(original, name + '-first-original', family, 5, 1, fixed=False)
            b = run(fixed, name + '-first-fixed', family, 5, 1)
            equal(a, b, family)
            a = run(original, name + '-selected-original', family, 5, 6, fixed=False)
            b = run(fixed, name + '-selected-disabled', family, 5, 6, fixed=False)
            equal(a, b, family)
            pair = [run(fixed, name + '-label' + str(label), family, label, 6) for label in (0, 5)]
            equal(*pair, family, ignore_fingerprint=True)
            reload(fixed, pair[0], family, 0)
            checks.append(dict(build=build['index'], family=family, first_epoch_parity=True,
                               disabled_patch_parity=True, final_epoch=6, query_label_isolation=True, reload_parity=True))
    # Cross the exact historical source boundary used by T6R and I2R.
    before, after = builds['builds'][2]['variants']['fixed']['binary'], builds['builds'][3]['variants']['fixed']['binary']
    for family in ('joint', 'integrated'):
        parents = [run(before, f'parent-{family}-{label}', family, label, 2) for label in (0, 5)]
        pair = [run(after, f'resume-{family}-{label}', family, label, 6, init=parent, start=3)
                for label, parent in zip((0, 5), parents)]
        equal(*pair, family, ignore_fingerprint=True)
        reload(after, pair[0], family, 0)
        if family == 'integrated':
            sidecar = np.frombuffer(parents[0].with_suffix('.bin.nbr').read_bytes(), '<f4', offset=16)
            require(np.any(sidecar != 0), 'Resume fixture sidecar is vacuous')
            zero_parent = output / 'zero-sidecar-parent'
            zero_parent.with_suffix('.bin').write_bytes(parents[0].with_suffix('.bin').read_bytes())
            original_sidecar = parents[0].with_suffix('.bin.nbr').read_bytes()
            zero_parent.with_suffix('.bin.nbr').write_bytes(original_sidecar[:16] + bytes(len(original_sidecar) - 16))
            zero_resume = run(after, 'zero-sidecar-resume', family, 0, 6, init=zero_parent, start=3)
            require(zero_resume.with_suffix('.bin').read_bytes() != pair[0].with_suffix('.bin').read_bytes(),
                    'Resume did not consume the nonzero parent sidecar')
        checks.append(dict(cross_version_resume=family, final_epoch=6, query_label_isolation=True,
                           sidecar_required=family == 'integrated', reload_parity=True))
    for day_rank in (0, 2):
        pair = [run(legacy['binary'], f'legacy-day{day_rank}-label{label}', 'joint', label, 2, day_rank=day_rank) for label in (0, 5)]
        equal(*pair, 'joint', ignore_fingerprint=True)
        reload(legacy['binary'], pair[0], 'joint', 0)
        checks.append(dict(retained_legacy=True, day_rank=day_rank, query_label_isolation=True, reload_parity=True))
    parents = [output / f'legacy-day0-label{label}' for label in (0, 5)]
    pair = [run(legacy['binary'], f'legacy-upgrade-label{label}', 'joint', label, 1, day_rank=2, init=parent)
            for label, parent in zip((0, 5), parents)]
    equal(*pair, 'joint', ignore_fingerprint=True)
    checks.append(dict(retained_J4_epoch_reset_and_day_upgrade=True, final_epoch=1, query_label_isolation=True))
    verify(frozen)
    report = dict(state='joint_and_integrated_fixture_contracts_verified', builds=str(build_path), builds_sha256=digest(build_path),
                  verifier_sha256=digest(Path(__file__)), frozen_inputs=frozen, checks=checks, commands=commands,
                  artifacts={str(p): digest(p) for p in sorted(output.iterdir())},
                  limitations=['Small one-thread fixtures and retained legacy replay provenance; no full production joint/integrated retraining.',
                               'Fixture ranks, graph size and budgets are reduced; production recipes must preserve their recorded values.',
                               'Production integrated resumes must require and bind the parent model.bin.nbr sidecar.'])
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(state=report['state'], check_groups=len(checks), commands=len(commands))))


if __name__ == '__main__':
    main()
