#!/usr/bin/env python3
"""Archive exact historical RBM sources from Git, explicitly recording gaps."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def recover(registry_path, archive_path, report_path):
    if archive_path.exists() or report_path.exists():
        raise FileExistsError('Choose new archive and report paths')
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    wanted = {}
    for stage in registry['stages']:
        for path, digest in stage['required_historical_sources'].items():
            wanted.setdefault(path, set()).add(digest)
    contents, recovered = {}, {}
    for path, digests in sorted(wanted.items()):
        commits = subprocess.check_output(
            ['git', 'log', '--all', '--format=%H', '--', path], cwd=ROOT,
            text=True).splitlines()
        for commit in commits:
            result = subprocess.run(['git', 'show', commit + ':' + path],
                                    cwd=ROOT, capture_output=True, check=False)
            if result.returncode:
                continue
            digest = sha256(result.stdout)
            key = path + ':' + digest
            if digest in digests and key not in recovered:
                member = digest + '/' + path
                contents[member] = result.stdout
                recovered[key] = dict(path=path, sha256=digest, commit=commit,
                                      git_object=commit + ':' + path,
                                      archive_member=member)
    missing = [dict(path=path, sha256=digest)
               for path, digests in sorted(wanted.items())
               for digest in sorted(digests)
               if path + ':' + digest not in recovered]
    # Fixed metadata makes the source archive reproducible across runs.
    with zipfile.ZipFile(archive_path, 'x') as archive:
        for member, payload in sorted(contents.items()):
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    with zipfile.ZipFile(archive_path) as archive:
        for source in recovered.values():
            if sha256(archive.read(source['archive_member'])) != source['sha256']:
                raise ValueError('Archive source round-trip failed')
    stages = []
    for stage in registry['stages']:
        gaps = [dict(path=path, sha256=digest)
                for path, digest in stage['required_historical_sources'].items()
                if path + ':' + digest not in recovered]
        stages.append(dict(name=stage['name'], complete_source_set=not gaps,
                           missing_sources=gaps,
                           learner_sha256=stage['required_historical_sources']['src/rbm.hpp']))
    if registry_path.read_bytes() != registry_bytes:
        raise ValueError('Registry changed during recovery')
    report = dict(
        state='partial_exact_source_recovery' if missing else 'exact_source_recovery',
        registry=str(registry_path), registry_sha256=sha256(registry_bytes),
        generator_sha256=sha256(Path(__file__).read_bytes()),
        archive=str(archive_path), archive_sha256=sha256(archive_path.read_bytes()),
        sources=list(recovered.values()), missing_sources=missing, stages=stages,
        limitations=[
            'Only exact SHA-256 matches are archived; no source substitution.',
            'Historical sources are unmodified and may lack fixed-epoch support.',
            'No binaries built, training compatibility proved, or production retraining performed.',
            'Incomplete source sets require additional recovery or separately evidenced compatible implementations.',
        ])
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = recover(args.registry.resolve(), args.archive.resolve(), args.report.resolve())
    print(json.dumps(dict(state=report['state'], recovered=len(report['sources']),
                          missing=len(report['missing_sources']),
                          complete_stages=sum(s['complete_source_set'] for s in report['stages']))))


if __name__ == '__main__':
    main()
