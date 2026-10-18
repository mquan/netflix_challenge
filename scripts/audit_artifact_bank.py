#!/usr/bin/env python3
"""Verify an experiment's input hashes and locate renamed prediction columns."""
import argparse
import hashlib
import json
import pathlib


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=pathlib.Path)
    parser.add_argument('--search-root', type=pathlib.Path, default=pathlib.Path('artifacts'))
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    record = json.loads(args.manifest.read_text())
    inputs = {name: info['sha256'] if isinstance(info, dict) else info
              for name, info in record['inputs'].items()}
    exact, mismatch, missing = {}, {}, {}
    for name, expected in inputs.items():
        path = pathlib.Path(name)
        if not path.is_file():
            missing[name] = expected
        else:
            actual = sha256(path)
            if actual == expected:
                exact[name] = str(path)
            else:
                mismatch[name] = {'expected': expected, 'actual': actual}
    wanted = set(missing.values()) | {value['expected'] for value in mismatch.values()}
    candidates = {}
    if wanted:
        for path in args.search_root.rglob('*.f32'):
            if path.is_file():
                digest = sha256(path)
                if digest in wanted:
                    candidates.setdefault(digest, []).append(str(path))
    relocated = {name: candidates[expected] for name, expected in missing.items()
                 if expected in candidates}
    relocated.update({name: candidates[value['expected']]
                      for name, value in mismatch.items()
                      if value['expected'] in candidates})
    report = {'manifest': str(args.manifest), 'required': len(inputs),
              'exact_count': len(exact), 'relocated_count': len(relocated),
              'unavailable_count': len(inputs) - len(exact) - len(relocated),
              'exact': exact, 'relocated': relocated,
              'mismatched_at_expected_path': mismatch,
              'unavailable': [name for name in inputs if name not in exact and name not in relocated]}
    body = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    print(body, end='')


if __name__ == '__main__':
    main()
