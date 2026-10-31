#!/usr/bin/env python3
"""Complete the historical RBM archive using exact-hash source reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--r2-build-input', type=Path, required=True)
    args = parser.parse_args()
    destinations = [p.resolve() for p in (args.archive, args.report, args.r2_build_input)]
    if any(p.exists() for p in destinations):
        raise FileExistsError('Choose new outputs')
    archive_path, report_path, build_input_path = destinations
    old_report_path = ROOT / 'experiments/P0-rbm-source-recovery-20261031.json'
    r1_report_path = ROOT / 'experiments/P0-rbm-R1-build-verification-20261031.json'
    old = json.loads(old_report_path.read_text())
    r1 = json.loads(r1_report_path.read_text())
    registry_path = Path(old['registry'])
    frozen = {old_report_path: digest(old_report_path), r1_report_path: digest(r1_report_path),
              Path(old['archive']): old['archive_sha256'],
              Path(r1['source_archive']): r1['source_archive_sha256'], registry_path: old['registry_sha256']}
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise ValueError('Recovery input changed')
    registry = json.loads(registry_path.read_text())
    stages = {s['name']: s for s in registry['stages']}
    payloads, metadata = {}, {}
    with zipfile.ZipFile(old['archive']) as archive:
        for source in old['sources']:
            key = source['path'], source['sha256']
            payloads[key] = archive.read(source['archive_member'])
            metadata[key] = dict(source)
    path = 'src/netflix.cpp'
    expected = stages['R1-rbm100']['required_historical_sources'][path]
    with zipfile.ZipFile(r1['source_archive']) as archive:
        payloads[path, expected] = archive.read(path)
    metadata[path, expected] = dict(path=path, sha256=expected, reconstruction_report=str(r1_report_path),
                                   reconstruction_report_sha256=frozen[r1_report_path])
    path = 'src/neighborhood.hpp'
    base_sha = stages['R3-rbm200-long']['required_historical_sources'][path]
    base = payloads[path, base_sha].decode()
    head, body = base.split('void command_knn', 1)
    start, end = body.index('  const auto f ='), body.index('  const auto g =')
    replacement = ('  const auto f = joint_features(d, t);\n'
                   '  TimeBaselineHeader header{};\n'
                   '  JointModel base;\n'
                   '  base.base = load_time_baseline(value_after(args, "--time-model"), d, t, header);\n')
    reconstructed = head + 'void command_knn' + body[:start] + replacement + body[end:]
    replacements = [
        ('predict(u, i, d.train_days[p], f.train_session[p])',
         'joint_base(d, f, base, u, i, d.train_days[p], f.train_session[p])'),
        ('predict(r.user, i, r.day, f.probe_session[p])',
         'joint_base(d, f, base, r.user, i, r.day, f.probe_session[p])'),
        ('"knn base=" << (joint_path.empty() ? "time" : joint_path) << " neighbors="', '"knn neighbors="'),
    ]
    for before, after in replacements:
        if reconstructed.count(before) != 1:
            raise ValueError('Unexpected R3 neighborhood source')
        reconstructed = reconstructed.replace(before, after)
    expected = stages['R2-rbm100-fast']['required_historical_sources'][path]
    payloads[path, expected] = reconstructed.encode()
    metadata[path, expected] = dict(path=path, sha256=expected,
                                   reconstruction_base_sha256=base_sha,
                                   method='Remove later joint-model kNN support; require original recorded SHA-256')
    wanted = {(p, h) for s in registry['stages'] for p, h in s['required_historical_sources'].items()}
    if set(payloads) != wanted:
        raise ValueError('Historical source coverage mismatch')
    with zipfile.ZipFile(archive_path, 'x') as archive:
        for (path, expected), payload in sorted(payloads.items()):
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError('Reconstructed source does not match historical hash: ' + path)
            member = expected + '/' + path
            metadata[path, expected]['archive_member'] = member
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    with zipfile.ZipFile(archive_path) as archive:
        for source in metadata.values():
            if hashlib.sha256(archive.read(source['archive_member'])).hexdigest() != source['sha256']:
                raise ValueError('Archive round-trip failed')
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise ValueError('Recovery input changed during reconstruction')
    report = dict(state='all_historical_RBM_sources_recovered_exactly',
                  registry=str(registry_path), registry_sha256=old['registry_sha256'],
                  archive=str(archive_path), archive_sha256=digest(archive_path),
                  generator_sha256=digest(Path(__file__)),
                  frozen_inputs={str(p): h for p, h in frozen.items()},
                  sources=list(metadata.values()), missing_sources=[],
                  stages=[dict(name=s['name'], complete_source_set=True) for s in registry['stages']],
                  limitations=['Source identity proves neither full-data retraining nor hidden-test accuracy.',
                               'R2 isolated build and fixture verification remain separate steps.'])
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    # The existing builder selects stages by source availability. Restrict this
    # input to R2's five sources to avoid rebuilding five already verified groups.
    selected = dict(report)
    r2_sources = stages['R2-rbm100-fast']['required_historical_sources']
    selected['state'] = 'R2_only_build_source_selection'
    selected['complete_recovery_report'] = str(report_path)
    selected['complete_recovery_report_sha256'] = digest(report_path)
    selected['sources'] = [metadata[p, h] for p, h in r2_sources.items()]
    selected['stages'] = [dict(name='R2-rbm100-fast', complete_source_set=True)]
    with build_input_path.open('x') as stream:
        stream.write(json.dumps(selected, indent=2) + '\n')
    print(json.dumps(dict(state=report['state'], exact_versions=len(payloads), stages=len(stages))))


if __name__ == '__main__':
    main()
