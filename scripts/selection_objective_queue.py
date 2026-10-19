#!/usr/bin/env python3
"""Wait for N1 and other Python jobs, then execute the frozen SO1 pair and gate."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def python_jobs(snapshot, own_pid):
    """Inspect executable names, not shell text containing Python commands."""
    jobs = []
    for line in snapshot.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            raise ValueError('unrecognized process snapshot row')
        pid, executable = int(fields[0]), fields[1]
        if pid == own_pid:
            continue
        if Path(executable).name.lower().startswith('python'):
            jobs.append(dict(pid=pid, executable=executable))
    return jobs


def inspect_idle(chain_status, own_pid):
    snapshot = subprocess.check_output(['ps', '-ww', '-axo', 'pid=,comm='], text=True)
    jobs = python_jobs(snapshot, own_pid)
    try:
        chain = json.loads(chain_status.read_text())
    except (OSError, ValueError):
        chain = {'state': 'unreadable'}
    ready = chain.get('state') in {'complete', 'stopped', 'failed'} and not jobs
    return ready, dict(n1_state=chain.get('state'), python_jobs=jobs)


def verify(frozen):
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise ValueError('frozen input changed: ' + str(path))


def run_pair(plan, run_stage, wait_idle, verify_inputs):
    for name in ('control', 'treatment'):
        wait_idle(name)
        verify_inputs()
        # Hashing large inputs takes time; require another live process check.
        wait_idle(name)
        run_stage(name, plan['arms'][name]['command'])
    verify_inputs()
    run_stage('gate', plan['gate_command'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'experiments/SO1-plan.json')
    parser.add_argument('--chain-status', type=Path, default=ROOT / '.claude/worktrees/honest-residual-chains/experiments/N1-chain.status.json')
    parser.add_argument('--name', default='SO1-queue')
    args = parser.parse_args()
    os.chdir(ROOT)
    prefix = ROOT / 'experiments' / args.name
    status_path = prefix.with_suffix('.status.json')
    launch_path = prefix.with_suffix('.launch.json')
    # Keep the descriptor open for the runner's whole lifetime.
    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if launch_path.exists() or status_path.exists():
            raise FileExistsError('existing queue record; inspect it before any manual recovery')
        plan = json.loads(args.plan.read_text())
        frozen = dict(plan['frozen'])
        for path in (args.plan, Path(__file__), ROOT / 'scripts/experiment.py'):
            frozen[str(path.resolve())] = digest(path)
        verify(frozen)
        for arm in plan['arms'].values():
            output = Path(arm['output'])
            if list(output.parent.glob(output.name + '.*')):
                raise FileExistsError('existing arm output: ' + str(output))
        gate_output = Path(plan['gate_command'][plan['gate_command'].index('--output') + 1])
        if gate_output.exists():
            raise FileExistsError(gate_output)
        for stage in ('control', 'treatment', 'gate'):
            if list((ROOT / 'experiments').glob(args.name + '-' + stage + '.*')):
                raise FileExistsError('existing experiment: ' + stage)
        launch_path.write_text(json.dumps(dict(
            pid=os.getpid(), frozen=frozen, plan=str(args.plan.resolve()),
            n1_status=str(args.chain_status.resolve()),
            condition='N1 terminal status and two idle process snapshots 30 seconds apart; conservatively wait for every other Python executable.',
            commands={**{k: v['command'] for k, v in plan['arms'].items()}, 'gate': plan['gate_command']},
            continuation='Stop after the fixed gate; a passing gate permits the planned full-data pair, not promotion.'
        ), indent=2) + '\n')

        def status(**values):
            payload = dict(pid=os.getpid(), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
            temporary = status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2) + '\n')
            temporary.replace(status_path)

        def wait_idle(stage):
            consecutive = 0
            while True:
                ready, evidence = inspect_idle(args.chain_status, os.getpid())
                consecutive = consecutive + 1 if ready else 0
                status(state='waiting', stage=stage, idle_samples=consecutive, **evidence)
                if consecutive >= 2:
                    return
                time.sleep(30)

        def run_stage(stage, command):
            status(state='running', stage=stage)
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', args.name + '-' + stage,
                            '--strategy', 'Frozen SO1 matched objective experiment; fixed epoch 2 and predeclared P2 gate',
                            '--', *command], check=True)

        try:
            run_pair(plan, run_stage, wait_idle, lambda: verify(frozen))
            verify(frozen)
            gate = json.loads(gate_output.read_text())
            status(state='complete', passes_full_refit_gate=gate['passes_full_refit_gate'],
                   primary=gate['primary'], gate_sha256=digest(gate_output))
        except BaseException as error:
            status(state='failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
