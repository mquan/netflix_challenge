#!/usr/bin/env python3
"""Export observed best-so-far scores with explicit timing provenance to CSV.

No dependencies. Old runs use frozen filesystem-time proxies, not fabricated
epoch timestamps. Future runs use timestamped stdout events from experiment.py.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import re


UTC = dt.timezone.utc
FIELDS = ['milestone', 'run', 'epoch', 'score', 'metric', 'timestamp_utc',
          'timestamp_local', 'hours_since_reference', 'reference_utc',
          'calendar_elapsed_seconds', 'recorded_work_seconds',
          'recorded_pause_seconds', 'unclassified_seconds',
          'elapsed_minus_recorded_pauses_seconds', 'activity_coverage',
          'timestamp_basis', 'earliest_utc', 'status', 'best_so_far_in_metric',
          'source', 'strategy']
METRICS = {'probe_rmse': 'single_model_probe',
           'blend_cv_rmse': 'blend_probe_user_cv', 'blend_fit_rmse': 'blend_probe_fit'}


def parse_time(value):
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timestamp must include timezone')
    return result.astimezone(UTC)


def scores(line):
    values = dict(re.findall(r'(\w+)=([^\s]+)', line))
    # Standalone columns in blend diagnostics are not new training milestones.
    if 'column' in values:
        return []
    result = []
    for key, metric in METRICS.items():
        if key not in values:
            continue
        value = float(values[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid score in log')
        result.append((metric, value, values.get('epoch', '')))
    return result


def experiment_rows(directory, proxies):
    rows = []
    for manifest in sorted(directory.glob('*.json')):
        record = json.loads(manifest.read_text())
        if 'command' not in record or 'started_utc' not in record:
            continue
        name = record['name']
        log = directory / (name + '.log')
        events = directory / (name + '.events.jsonl')
        status = ('completed' if record.get('exit_code') == 0 else
                  'stopped_or_failed' if 'exit_code' in record else 'running')
        if events.exists():
            entries = []
            for line in events.read_text().splitlines(keepends=True):
                if not line.endswith('\n'):  # writer may still be flushing
                    continue
                event = json.loads(line)
                entries.append((event['line'], event['observed_utc'], 'stdout_observed', str(events)))
        else:
            # A live legacy run has no per-score clock: defer it until complete.
            if 'exit_code' not in record or not log.exists():
                continue
            content = log.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()
            if record.get('log_sha256') != checksum:
                raise ValueError('completed log hash mismatch: ' + str(log))
            if name not in proxies:
                # Freeze this evidence now: checkout/copying later changes mtimes.
                proxies[name] = dict(log_sha256=checksum,
                                    timestamp_utc=dt.datetime.fromtimestamp(log.stat().st_mtime, UTC).isoformat(),
                                    captured_utc=dt.datetime.now(UTC).isoformat())
            proxy = proxies[name]
            if proxy['log_sha256'] != checksum:
                raise ValueError('frozen timing proxy log changed: ' + name)
            if parse_time(proxy['timestamp_utc']) < parse_time(record['started_utc']):
                raise ValueError('log proxy predates experiment: ' + name)
            entries = [(line, proxy['timestamp_utc'], 'log_mtime_proxy_upper_bound', str(log))
                       for line in content.decode().splitlines()]
            # Only the run's best final checkpoint is timed by an old final-log
            # proxy. Do not invent a time series of individual old epochs.
            best = {}
            for entry in entries:
                for metric, value, epoch in scores(entry[0]):
                    if metric not in best or value < best[metric][0]:
                        best[metric] = (value, epoch, entry)
            entries = list(dict.fromkeys(v[2] for v in best.values()))
        best = {}
        for line, timestamp, basis, source in entries:
            for metric, value, epoch in scores(line):
                if metric in best and value >= best[metric]:
                    continue
                best[metric] = value
                rows.append(dict(milestone=name.split('-')[0], run=name, epoch=epoch,
                                 score=value, metric=metric, timestamp_utc=timestamp,
                                 timestamp_basis=basis, earliest_utc=record.get('launched_utc', record['started_utc']),
                                 status=status, source=source, strategy=record['strategy']))
    return rows


def load_activity(path):
    """Only explicitly observed intervals count; absence of output is not idle."""
    if not path.exists():
        return []
    result = []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            begin, end = parse_time(row['start_utc']), parse_time(row['end_utc'])
            if end <= begin:
                raise ValueError('activity interval must have positive duration')
            if row['state'] not in ('working', 'idle', 'rate_limit', 'timeout'):
                raise ValueError('unknown activity state: ' + row['state'])
            if not row['source'].strip():
                raise ValueError('activity interval needs evidence/source')
            result.append((begin, end, row['state']))
    return result


def activity_totals(start, end, intervals):
    """Union overlapping intervals. Useful work overrides assistant-only pauses.

    Unknown time stays unknown, not silently relabeled as working or idle.
    """
    events = {}
    for begin, finish, state in intervals:
        begin, finish = max(start, begin), min(end, finish)
        if finish <= begin:
            continue
        kind = 0 if state == 'working' else 1
        events.setdefault(begin, [0, 0])[kind] += 1
        events.setdefault(finish, [0, 0])[kind] -= 1
    events.setdefault(end, [0, 0])
    counts, totals = [0, 0], [0.0, 0.0, 0.0]
    previous = start
    for moment, change in sorted(events.items()):
        kind = 0 if counts[0] else 1 if counts[1] else 2
        totals[kind] += (moment-previous).total_seconds()
        counts = [counts[k]+change[k] for k in (0, 1)]
        previous = moment
    return totals  # work, pause, unknown


def decorate(rows, timezone, intervals=()):
    if not rows:
        return rows
    # Stable sorting preserves historical sequence when several bounds coincide.
    rows.sort(key=lambda row: parse_time(row['timestamp_utc']))
    reference = parse_time(rows[0]['timestamp_utc'])
    best = {}
    for row in rows:
        timestamp = parse_time(row['timestamp_utc'])
        row['timestamp_local'] = timestamp.astimezone(timezone).isoformat()
        row['reference_utc'] = reference.isoformat()
        row['hours_since_reference'] = format((timestamp-reference).total_seconds()/3600, '.6f')
        elapsed = (timestamp-reference).total_seconds()
        work, pause, unknown = activity_totals(reference, timestamp, intervals)
        for key, value in [('calendar_elapsed_seconds', elapsed),
                           ('recorded_work_seconds', work), ('recorded_pause_seconds', pause),
                           ('unclassified_seconds', unknown),
                           ('elapsed_minus_recorded_pauses_seconds', elapsed-pause)]:
            row[key] = format(value, '.6f')
        row['activity_coverage'] = 'complete' if unknown < 1e-6 else 'partial'
        row['score'] = format(float(row['score']), '.10f')
        metric = row['metric']
        best[metric] = min(best.get(metric, math.inf), float(row['score']))
        row['best_so_far_in_metric'] = format(best[metric], '.10f')
    return rows


def main():
    from zoneinfo import ZoneInfo
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory', type=pathlib.Path, default=pathlib.Path('experiments'))
    parser.add_argument('--timezone', default='America/Los_Angeles')
    args = parser.parse_args()
    directory = args.directory
    proxy_path = directory / 'timing_backfill.json'
    proxies = json.loads(proxy_path.read_text()) if proxy_path.exists() else {}
    with (directory/'legacy_milestones.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(run=row['milestone'], epoch='', earliest_utc='', status='historical')
    rows += experiment_rows(directory, proxies)
    rows = decorate(rows, ZoneInfo(args.timezone), load_activity(directory/'activity_intervals.csv'))
    proxy_path.write_text(json.dumps(proxies, indent=2, sort_keys=True)+'\n')
    output = directory / 'score_timeline.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(str(output) + ': ' + str(len(rows)) + ' observations')


if __name__ == '__main__':
    main()
