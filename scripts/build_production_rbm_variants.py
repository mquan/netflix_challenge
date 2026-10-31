#!/usr/bin/env python3
"""Build isolated historical RBMs with a checkpoint-only fixed-epoch patch."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_source(source):
    replacements = [
        ('  const uint32_t patience = uint_after(args, "--patience", 6);',
         '  const bool fixed_epoch = std::find(args.begin(), args.end(), "--fixed-epoch") != args.end();\n'
         '  const uint32_t patience = uint_after(args, "--patience", 6);'),
        ('    if (score < best) { best = score; stale = 0; save_rbm(model_path, m, fingerprint); export_predictions(); }',
         '    if (fixed_epoch || score < best) { best = std::min(best, score); stale = 0; save_rbm(model_path, m, fingerprint); export_predictions(); }'),
        ('    if (patience && stale >= patience)',
         '    if (!fixed_epoch && patience && stale >= patience)'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Unexpected historical save/stop implementation')
        source = source.replace(before, after)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    recovery_path = args.recovery.resolve()
    recovery_sha = digest(recovery_path)
    recovery = json.loads(recovery_path.read_text())
    archive_path = Path(recovery['archive'])
    registry_path = Path(recovery['registry'])
    if digest(archive_path) != recovery['archive_sha256'] or digest(registry_path) != recovery['registry_sha256']:
        raise ValueError('Frozen recovery input changed')
    registry = json.loads(registry_path.read_text())
    output = args.output_dir.resolve()
    report_path = args.report.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError('Choose new output paths')
    output.mkdir(parents=True)
    sources = {(s['path'], s['sha256']): s for s in recovery['sources']}
    groups, excluded = {}, []
    for stage in registry['stages']:
        required = stage['required_historical_sources']
        if any((p, h) not in sources for p, h in required.items()):
            excluded.append(stage['name'])
            continue
        key = json.dumps(required, sort_keys=True)
        groups.setdefault(key, []).append(stage['name'])
    builds = []
    with zipfile.ZipFile(archive_path) as archive:
        for index, (key, stages) in enumerate(groups.items()):
            required = json.loads(key)
            directory = output / ('variant-' + str(index))
            original = directory / 'original'
            fixed = directory / 'fixed'
            for target in (original, fixed):
                (target / 'src').mkdir(parents=True)
                for path, expected in required.items():
                    payload = archive.read(sources[path, expected]['archive_member'])
                    if hashlib.sha256(payload).hexdigest() != expected:
                        raise ValueError('Historical source mismatch')
                    (target / path).write_bytes(payload)
            old = (original / 'src/rbm.hpp').read_text()
            new = fixed_source(old)
            (fixed / 'src/rbm.hpp').write_text(new)
            patch = directory / 'fixed-epoch.patch'
            patch.write_text(''.join(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile='original/src/rbm.hpp', tofile='fixed/src/rbm.hpp')))
            variants = {}
            for name, target in (('original', original), ('fixed', fixed)):
                binary = target / 'netflix'
                command = ['nice', '-n', '10', 'clang++', '-O3', '-DNDEBUG',
                           '-std=c++17', '-Wall', '-Wextra', '-Wpedantic',
                           str(target / 'src/netflix.cpp'), '-framework', 'Accelerate',
                           '-o', str(binary)]
                print(json.dumps(dict(build=index, stages=stages, variant=name)), flush=True)
                log = target / 'build.log'
                with log.open('w') as stream:
                    subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                actual_sources = {p: digest(target / p) for p in required}
                for path, expected in required.items():
                    if name == 'fixed' and path == 'src/rbm.hpp':
                        continue
                    if actual_sources[path] != expected:
                        raise ValueError('Unexpected source edit')
                variants[name] = dict(binary=str(binary), binary_sha256=digest(binary),
                                      sources=actual_sources, command=command,
                                      log=str(log), log_sha256=digest(log))
            builds.append(dict(index=index, stages=stages, required_sources=required,
                               variants=variants, patch=str(patch), patch_sha256=digest(patch)))
    if digest(recovery_path) != recovery_sha or digest(archive_path) != recovery['archive_sha256'] or digest(registry_path) != recovery['registry_sha256']:
        raise ValueError('Input changed during compilation')
    report = dict(state='isolated_builds_complete_not_training_verified',
                  generator_sha256=digest(Path(__file__)), recovery=str(recovery_path),
                  recovery_sha256=recovery_sha, archive_sha256=recovery['archive_sha256'],
                  registry_sha256=recovery['registry_sha256'], builds=builds,
                  excluded_stages=excluded,
                  limitations=['Checkpoint selection and stopping are the only patched logic.',
                               'Fixture training contracts and cross-version resumes remain to be verified.',
                               'No full production training performed; incomplete source sets are excluded.'])
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
