#!/usr/bin/env python3
"""Execute a frozen multi-base export plan after its training inputs finish."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def ready(plan):
    for producer in plan['producers']:
        path = Path(producer) / 'report.json'
        report = json.loads(path.read_text()) if path.exists() else {}
        if report.get('state') != 'complete':
            return False
    if plan.get('snapshot_producer'):
        path = Path(plan['snapshot_producer']) / 'snapshots/report.json'
        captures = json.loads(path.read_text()).get('snapshots', {}) if path.exists() else {}
        if any(str(epoch) not in captures for epoch in plan['snapshot_epochs']):
            return False
    path = Path(plan['models_dir']) / 'report.json'
    return path.exists() and json.loads(path.read_text()).get('state') == 'models_ready'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'pipeline-report.json'
    if path.exists():
        raise ValueError('pipeline report exists')
    report = dict(state='waiting', plan=plan, stages=[],
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

    def status(state):
        report['state'] = state
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(path)
        print(json.dumps(dict(state=state)), flush=True)

    status('waiting')
    deadline = time.monotonic() + 12 * 3600
    while not ready(plan):
        if time.monotonic() > deadline:
            raise TimeoutError('required training inputs did not finish')
        time.sleep(15)
    for stage in plan['stages']:
        status(stage['name'])
        subprocess.run([sys.executable, 'scripts/experiment.py', '--name',
                        plan['name'] + '-' + stage['name'], '--strategy',
                        stage['strategy'], '--', sys.executable] + stage['command'], check=True)
        report['stages'].append(stage['name'])
    audit = json.loads((Path(plan['audit_output']) / 'report.json').read_text())
    report['audit'] = audit
    status('target_met' if audit['both_source_groups_below_target'] else 'complete_target_unmet')


if __name__ == '__main__':
    main()
