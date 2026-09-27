#!/usr/bin/env python3
"""Run a reproducible experiment, preserving command, source hashes and stdout."""
import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def capture_output(command, log_path, events_path):
    """Timestamp stdout receipt, without changing the original model log format."""
    with log_path.open('x') as log, events_path.open('x') as events:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            observed = utc_now()
            log.write(line)
            log.flush()
            events.write(json.dumps(dict(observed_utc=observed, line=line.rstrip('\n'))) + '\n')
            events.flush()
            print(line, end='', flush=True)
        code = proc.wait()
        proc.stdout.close()
    return code


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--strategy', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('a command is required after --')
    if pathlib.Path(args.name).name != args.name:
        parser.error('name must be a filename')
    directory = pathlib.Path('experiments')
    directory.mkdir(exist_ok=True)
    log_path = directory / (args.name + '.log')
    events_path = directory / (args.name + '.events.jsonl')
    manifest_path = directory / (args.name + '.json')
    if log_path.exists() or manifest_path.exists() or events_path.exists():
        parser.error('experiment already exists; use a new name')
    record = dict(name=args.name, strategy=args.strategy, command=command,
                  started_utc=utc_now(),
                  platform=platform.platform(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  sources={str(p): digest(p) for p in pathlib.Path('src').glob('*') if p.is_file()},
                  binary_sha256=digest(command[0]),
                  metric='Full Netflix probe development RMSE; no independent test claim')
    # Hash dependencies before launch, so blends can be tied to exact columns
    # even if a later experiment overwrites a conventional artifact filename.
    record['inputs'] = {}
    for i, arg in enumerate(command[:-1]):
        if arg in ('--data', '--prediction', '--init-model', '--time-model', '--svd-model') or (arg == '--graph' and 'graph' != command[1]):
            path = pathlib.Path(command[i+1])
            record['inputs'][str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    record['launched_utc'] = utc_now()
    record['events_path'] = str(events_path)
    manifest_path.write_text(json.dumps(record, indent=2) + '\n')
    start = time.monotonic()
    code = capture_output(command, log_path, events_path)
    record.update(exit_code=code, elapsed_seconds=time.monotonic()-start,
                  finished_utc=utc_now(), log_sha256=digest(log_path),
                  events_sha256=digest(events_path))
    for flag in ('--model', '--predictions', '--graph', '--weights'):
        if flag in command:
            path = pathlib.Path(command[command.index(flag)+1])
            if path.exists():
                record[flag[2:] + '_artifact'] = dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(record, indent=2) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
