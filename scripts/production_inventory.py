#!/usr/bin/env python3
"""Read-only provenance inventory for the promoted bank; not a retrain-readiness claim."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

from local_context_chain import resolve
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
R9 = Path('/Users/quan/workspace/netflix_round9')


def output_digests(record):
    """Only explicit outputs qualify. An input hash does not prove production."""
    found = set()
    for key in ('predictions_sha256', 'prediction_sha256'):
        if isinstance(record.get(key), str):
            found.add(record[key])
    value = record.get('predictions_artifact')
    if isinstance(value, dict) and isinstance(value.get('sha256'), str):
        found.add(value['sha256'])
    for path, value in record.get('outputs', {}).items() if isinstance(record.get('outputs'), dict) else []:
        if path.endswith('.f32') and isinstance(value, str):
            found.add(value)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    destination = Path(a.output)
    if destination.exists():
        parser.error('choose a new output path')
    base_path = R9 / 'artifacts/r9/R9-LAT-A2h-20261010-s3.json'
    base = json.loads(base_path.read_text())
    config = base['config']
    bank_root = R9 / config['bank_root']
    union_path = bank_root / 'experiments/UNION-round-manifest.json'
    union = json.loads(union_path.read_text())['reports']['union']['inputs']
    rows = [(str(bank_root / name), expected) for name, expected in union.items()]
    rows += [(str(resolve(name, config, R9)), base['inputs'][name]) for name in config['extra']]
    wanted = {expected for _, expected in rows}
    locations = [ROOT / 'experiments', ROOT / 'artifacts/masked', R9 / 'experiments', R9 / 'artifacts/masked']
    worktrees = ROOT / '.claude/worktrees'
    if worktrees.exists():
        locations += [path for child in worktrees.iterdir() if child.is_dir()
                      for path in (child / 'experiments', child / 'artifacts/masked') if path.is_dir()]
    candidates = subprocess.check_output(['rg', '--files', '--hidden', '--no-ignore', '-g', '*.json',
                                          *map(str, locations)], text=True).splitlines()
    matches = {value: [] for value in wanted}
    unreadable = []
    for filename in sorted(set(candidates)):
        path = Path(filename)
        try:
            body = path.read_bytes()
            record = json.loads(body)
        except (OSError, ValueError) as error:
            unreadable.append(dict(path=filename, error=type(error).__name__))
            continue
        if not isinstance(record, dict):
            continue
        intersection = output_digests(record) & wanted
        if not intersection:
            continue
        command = record.get('command')
        evidence = dict(record=filename, record_sha256=digest(path),
                        kind='command_with_output_digest' if isinstance(command, list) else 'output_digest_only')
        if isinstance(command, list):
            evidence['command'] = command
            evidence['exit_code'] = record.get('exit_code')
        for value in intersection:
            matches[value].append(evidence)
    columns = []
    for filename, expected in rows:
        path = Path(filename)
        verified = path.is_file() and digest(path) == expected
        sources = matches[expected]
        columns.append(dict(path=str(path), sha256=expected, current_artifact_verified=verified,
                            command_record_found=any(s['kind'] == 'command_with_output_digest' for s in sources),
                            explicit_output_record_found=bool(sources), producer_records=sources))
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base_report=str(base_path), base_report_sha256=digest(base_path),
        union_manifest_sha256=digest(union_path), columns=columns,
        counts=dict(prediction_columns=len(columns), unique_hashes=len(wanted),
                    verified=sum(c['current_artifact_verified'] for c in columns),
                    with_command_record=sum(c['command_record_found'] for c in columns),
                    with_explicit_output_record=sum(c['explicit_output_record_found'] for c in columns)),
        scanned_json_files=len(set(candidates)), unreadable_records=unreadable,
        qualifying_file_available=(ROOT / 'data/raw/extracted/download/qualifying.txt').is_file(),
        limitations=['Only prediction columns are inventoried here; support features and blender deployment still need recipes.',
                     'A matching output digest is provenance evidence, not proof of complete dependencies, rebuildability or fixed-epoch qualifying inference.',
                     'The inventory does not retrain models, change promoted scores, or establish a hidden-test result.'])
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(counts=report['counts'], scanned_json_files=report['scanned_json_files'],
                         unresolved=[c['path'] for c in columns if not c['explicit_output_record_found']])), flush=True)


if __name__ == '__main__':
    main()
