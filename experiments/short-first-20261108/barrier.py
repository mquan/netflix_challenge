"""Release held own controllers only after audited short runs finish."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import datetime

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compatible_cpu(actual, expected, plan_sha, index):
    return (actual.get('state') == 'verified_complete' and
            actual.get('stage') == index and actual.get('plan_sha256') == plan_sha and
            actual.get('column') == expected['column'] and
            actual.get('fixed_epochs') == expected['fixed_epochs'] and
            actual.get('qualifying_count') == 2817131 and
            'metric' in actual and actual['metric'] is None)


def ready(cpu_ok, arms_ok, summary_ok, short_runners_live, pressure):
    return (len(cpu_ok) == 7 and all(cpu_ok) and len(arms_ok) == 25 and
            all(arms_ok) and summary_ok and not short_runners_live and pressure == '1')


def identities():
    output = subprocess.check_output(['ps', '-ww', '-axo', 'pid=,command='], text=True)
    return {int(parts[0]): parts[1] for line in output.splitlines()
            if len(parts := line.strip().split(maxsplit=1)) == 2}


def check():
    cpu_path = ROOT / 'experiments/P0-CPU-neural-20261031-plan.json'
    cpu_sha = digest(cpu_path)
    cpu = json.loads(cpu_path.read_text())
    monitor = json.loads((ROOT / 'experiments/S0-arm-monitor-20261031-plan.json').read_text())
    cpu_ok, arms_ok, receipts = [], [], []
    for index, stage in enumerate(cpu['stages']):
        path = Path(stage['output_dir']) / 'audit.json'
        value = json.loads(path.read_text()) if path.exists() else {}
        cpu_ok.append(compatible_cpu(value, stage['expected'], cpu_sha, index))
        if value:
            receipts.append(value)
    for name, path in monitor['receipts'].items():
        path = Path(path)
        value = json.loads(path.read_text()) if path.exists() else {}
        arms_ok.append(value.get('state') == 'verified_complete_arm' and value.get('arm') == name and
                       value.get('execution_sha256') == monitor['control_bindings'][monitor['execution']] and
                       value.get('reporter_binding_sha256') == monitor['control_bindings'][monitor['reporter_binding']])
        if value:
            receipts.append(value)
    summary_path = ROOT / 'artifacts/structural/S0-scoping-20261031/final-summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary_ok = (summary.get('state') == 'complete_all_fixed_arms_audited' and
                  len(summary.get('arms', {})) == 26 and
                  summary.get('execution_sha256') == digest(monitor['execution']) and
                  summary.get('binding_sha256') == digest(monitor['reporter_binding']))
    processes = identities()
    patterns = ['S0-lane0-20261031-run.sh', 'S0-lane1-20261031-run.sh',
                'P0-CPU-neural-20261031-run.sh', 'S0-report-handoff-20261031-run.sh',
                'scripts/round4_boost.py', 'scripts/round8_nn_blend.py']
    live = any(pattern in command for command in processes.values() for pattern in patterns)
    pressure = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
    if not ready(cpu_ok, arms_ok, summary_ok, live, pressure):
        print(json.dumps(dict(state='waiting_short_first', cpu_audits=sum(cpu_ok),
                              s0_audits=sum(arms_ok), summary_complete=summary_ok,
                              short_runners_live=live, pressure=pressure)), flush=True)
        return False
    # Full hashes are checked only when all completion gates are satisfied.
    for receipt in receipts + [summary]:
        for path, expected in receipt['outputs'].items():
            if digest(path) != expected:
                raise ValueError('Completed short-run output changed: ' + path)
    print('READY_ALL_SHORT_RUNS_AUDITED_AND_EXITED', flush=True)
    return True


def release():
    hold = json.loads((HERE / 'hold.json').read_text())
    # Verify every identity before resuming any process; never signal a reused PID.
    for row in hold['queues']:
        command = subprocess.check_output(['ps', '-ww', '-p', str(row['pid']), '-o', 'command='], text=True).strip()
        started = subprocess.check_output(['ps', '-p', str(row['pid']), '-o', 'lstart='], text=True).strip()
        state = subprocess.check_output(['ps', '-p', str(row['pid']), '-o', 'state='], text=True).strip()
        if command != '/bin/bash ' + row['script'] or started != row['identity'] or 'T' not in state:
            raise ValueError('Held controller identity/state differs: ' + row['name'])
    path = HERE / 'release.json'
    result = dict(state='releasing_after_short_runs', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  released=[], scope='Original guards and commands resume; no new trainer launched by barrier.')
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
    for row in hold['queues']:
        os.kill(row['pid'], signal.SIGCONT)
        result['released'].append(row['pid'])
        path.write_text(json.dumps(result, indent=2) + '\n')
    result['state'] = 'all_short_runs_complete_original_guards_resumed'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(result['state'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    try:
        if not check():
            raise SystemExit(1)
        if args.release:
            release()
    except Exception as error:
        print('SHORT_FIRST_BARRIER_ERROR: ' + repr(error), flush=True)
        raise SystemExit(2)
