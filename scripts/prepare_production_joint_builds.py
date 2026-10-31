"""Recover and build the retained joint recipes without launching full training."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from production_rbm_audit import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def fixed_source(source, patience):
    replacements = [
        (f'  const uint32_t patience = uint_after(args, "--patience", {patience});',
         '  const bool fixed_epoch = std::find(args.begin(), args.end(), "--fixed-epoch") != args.end();\n'
         f'  const uint32_t patience = uint_after(args, "--patience", {patience});'),
        ('    if (score < best) {', '    if (fixed_epoch || score < best) {'),
        ('      best = score; stale = 0;', '      best = std::min(best, score); stale = 0;'),
        ('    if (patience && stale >= patience)', '    if (!fixed_epoch && patience && stale >= patience)'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Unexpected historical joint save/stop implementation')
        source = source.replace(before, after)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    output, archive_path, report_path = (p.resolve() for p in (args.output_dir, args.archive, args.report))
    if any(p.exists() for p in (output, archive_path, report_path)):
        raise FileExistsError('Choose new build and evidence paths')
    registry_path = ROOT / 'experiments/production-joint-recipes.json'
    registry = json.loads(registry_path.read_text())
    legacy_evidence = [ROOT / 'experiments/PR5-J1-retraining-verification.json',
                       ROOT / 'experiments/PR6-legacy-joint-final-verification.json',
                       ROOT / 'experiments/PR6-legacy-joint.plan.json']
    recovery_path = ROOT / 'experiments/P0-dependency-source-recovery-20261031.json'
    recovery = json.loads(recovery_path.read_text())
    legacy_binary = ROOT / 'build/netflix'
    expected_legacy_sha = '686631036b7f1c2f4e0c0d17d5bf7e8bc15e03834d3abc346d5fb7361db5234c'
    frozen = {str(p): digest(p) for p in [registry_path, recovery_path, *legacy_evidence]}
    frozen[str(legacy_binary)] = expected_legacy_sha
    for relative, expected in recovery['sources'].items():
        frozen[str(Path(recovery['source_root']) / relative)] = expected
    verify(frozen)
    # The first four retained artifacts differ from their earliest producer hashes;
    # PR5/PR6 prove these later exact retained recipes, not the obsolete producers.
    legacy_names = [s['name'] for s in registry['stages'][:4]]
    if legacy_names != ['J1-joint64', 'J2-joint128-reg02', 'J3-joint128-reg005', 'J4-joint128-day128']:
        raise ValueError('Revalidate retained legacy mapping')
    wanted = {}
    for stage in registry['stages'][4:]:
        for path, expected in stage['required_historical_sources'].items():
            wanted.setdefault(path, set()).add(expected)
    contents, sources = {}, {}
    for path, hashes in sorted(wanted.items()):
        commits = subprocess.check_output(['git', 'log', '--all', '--format=%H', '--', path], cwd=ROOT, text=True).splitlines()
        for commit in commits:
            result = subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT, capture_output=True)
            if result.returncode:
                continue
            expected = hashlib.sha256(result.stdout).hexdigest()
            key = path, expected
            if expected in hashes and key not in sources:
                member = expected + '/' + path
                contents[member] = result.stdout
                sources[key] = dict(path=path, sha256=expected, commit=commit, archive_member=member)
    if any((p, h) not in sources for p, hashes in wanted.items() for h in hashes):
        raise ValueError('Historical joint source recovery incomplete')
    with zipfile.ZipFile(archive_path, 'x') as archive:
        for name, payload in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    with zipfile.ZipFile(archive_path) as archive:
        for source in sources.values():
            if hashlib.sha256(archive.read(source['archive_member'])).hexdigest() != source['sha256']:
                raise ValueError('Source archive round-trip failed')
    output.mkdir(parents=True)
    legacy_dir = output / 'retained-legacy'
    legacy_dir.mkdir()
    legacy_copy = legacy_dir / 'netflix'
    shutil.copy2(legacy_binary, legacy_copy)
    if digest(legacy_copy) != expected_legacy_sha:
        raise ValueError('Isolated legacy binary copy differs')
    groups = {}
    for stage in registry['stages'][4:]:
        key = json.dumps(stage['required_historical_sources'], sort_keys=True)
        groups.setdefault(key, []).append(stage['name'])
    builds = []
    for index, (key, stages) in enumerate(groups.items()):
        required = json.loads(key)
        directory = output / ('historical-' + str(index))
        variants = {}
        patches = {}
        for variant in ('original', 'fixed'):
            target = directory / variant
            (target / 'src').mkdir(parents=True)
            for path, expected in required.items():
                payload = contents[sources[path, expected]['archive_member']]
                if variant == 'fixed' and path in ('src/joint.hpp', 'src/integrated.hpp'):
                    old = payload.decode()
                    new = fixed_source(old, 6 if path == 'src/joint.hpp' else 4)
                    patch = directory / (Path(path).name + '.fixed-epoch.patch')
                    patch.write_text(''.join(difflib.unified_diff(old.splitlines(keepends=True),
                        new.splitlines(keepends=True), fromfile='original/' + path, tofile='fixed/' + path)))
                    patches[str(patch)] = digest(patch)
                    payload = new.encode()
                (target / path).write_bytes(payload)
            binary = target / 'netflix'
            command = ['nice', '-n', '10', 'clang++', '-O3', '-DNDEBUG', '-std=c++17',
                       '-Wall', '-Wextra', '-Wpedantic', str(target / 'src/netflix.cpp'),
                       '-framework', 'Accelerate', '-o', str(binary)]
            print(json.dumps(dict(build=index, stages=stages, variant=variant)), flush=True)
            with (target / 'build.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            actual = {p: digest(target / p) for p in required}
            if any(actual[p] != h for p, h in required.items()
                   if not (variant == 'fixed' and p in ('src/joint.hpp', 'src/integrated.hpp'))):
                raise ValueError('Unexpected historical source edit')
            variants[variant] = dict(binary=str(binary), binary_sha256=digest(binary),
                                     sources=actual, command=command,
                                     build_log=str(target / 'build.log'), build_log_sha256=digest(target / 'build.log'))
        builds.append(dict(index=index, stages=stages, historical_sources=required, variants=variants, patches=patches))
    verify(frozen)
    report = dict(state='isolated_joint_builds_complete_not_fixture_verified',
                  registry=str(registry_path), registry_sha256=digest(registry_path),
                  generator_sha256=digest(Path(__file__)), frozen_inputs=frozen,
                  archive=str(archive_path), archive_sha256=digest(archive_path), sources=list(sources.values()),
                  retained_legacy=dict(stages=legacy_names, binary=str(legacy_copy), binary_sha256=digest(legacy_copy),
                      source_root=recovery['source_root'], sources=recovery['sources'],
                      evidence={str(p): digest(p) for p in legacy_evidence},
                      reason='Full original-data retraining reproduced the current retained J1/J2/J3/J4 checkpoints and predictions exactly; their earliest historical producer artifacts differ.'),
                  builds=builds,
                  limitations=['Only checkpoint saving and stopping are patched in historical learner headers.',
                               'Historical fixed-budget fixture contracts and resume sidecars remain to be verified.',
                               'No full production joint or integrated training performed.'])
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(state=report['state'], historical_source_versions=len(sources),
                          historical_groups=len(builds), retained_legacy_stages=len(legacy_names))))


if __name__ == '__main__':
    main()
