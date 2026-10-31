#!/usr/bin/env python3
"""Run the unchanged SO1 plans after the paused N1 checkpoint is evaluated.

The original all-Python idle guard predates Codex transcript workers. This
adapter excludes only their exact executable/module command; unknown Python
jobs still block. Original plans, trainers, validators and gates stay frozen.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import selection_objective_queue as queue

ROOT = Path(__file__).resolve().parents[1]
WORKER = str(Path.home() / '.local/share/uv/tools/runlayer/bin/python')


def classify(snapshot, own_pid):
    blocking, infrastructure = [], []
    for line in snapshot.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            raise ValueError('incomplete process snapshot')
        pid, executable, command = int(fields[0]), fields[1], fields[2]
        if pid == own_pid:
            continue
        if not Path(executable).name.lower().startswith('python'):
            continue
        entry = dict(pid=pid, executable=executable, command=command)
        if executable == WORKER and shlex.split(command) == [
                WORKER, '-m', 'runlayer_cli.hook._transcript_stream_worker']:
            infrastructure.append(entry)
        else:
            blocking.append(entry)
    return blocking, infrastructure


def process_snapshot():
    # On macOS comm is truncated when it precedes args, even with -ww.
    # Read each as the last column so the executable match is exact.
    executables = subprocess.check_output(['ps', '-ww', '-axo', 'pid=,comm='], text=True)
    commands = subprocess.check_output(['ps', '-ww', '-axo', 'pid=,args='], text=True)
    command_by_pid = dict(line.strip().split(None, 1) for line in commands.splitlines())
    rows = []
    for line in executables.splitlines():
        pid, executable = line.strip().split(None, 1)
        rows.append(f'{pid} {executable} {command_by_pid.get(pid, "<exited-or-unreadable>")}')
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--n1-evidence', type=Path, required=True)
    parser.add_argument('--n1-followup-plan', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    prefix = ROOT / 'experiments' / args.name
    status_path = prefix.with_suffix('.status.json')
    launch_path = prefix.with_suffix('.launch.json')
    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists() or launch_path.exists():
            raise FileExistsError('existing resume record')
        original = json.loads((ROOT / 'experiments/SO1-queue.launch.json').read_text())
        queue.verify(original['frozen'])
        plan_path = ROOT / 'experiments/SO1-plan.json'
        plan = json.loads(plan_path.read_text())
        full_path = ROOT / 'experiments/SO1-full-plan.json'
        full_plan = json.loads(full_path.read_text())
        queue.verify(full_plan['frozen'])
        evidence = json.loads(args.n1_evidence.read_text())
        if evidence['state'] != 'stopped':
            raise ValueError('N1 selection is not stopped')
        if evidence['gate_passed'] and not args.n1_followup_plan:
            raise ValueError('passing N1 gate requires its fixed follow-up plan')
        frozen = dict(original['frozen'])
        frozen.update({str(path.resolve()): queue.digest(path) for path in
                       (Path(__file__), args.n1_evidence, full_path)})
        frozen.update(evidence['frozen'])
        followup = None
        if args.n1_followup_plan:
            followup = json.loads(args.n1_followup_plan.read_text())
            frozen.update(followup['frozen'])
            frozen[str(args.n1_followup_plan.resolve())] = queue.digest(args.n1_followup_plan)
        gate_path = ROOT / 'experiments/SO1-gate.json'
        if gate_path.exists():
            raise FileExistsError(gate_path)
        for arm in plan['arms'].values():
            output = Path(arm['output'])
            if list(output.parent.glob(output.name + '.*')):
                raise FileExistsError(output)
        launch_path.write_text(json.dumps(dict(pid=os.getpid(), frozen=frozen,
            policy='Only exact Runlayer transcript workers are excluded; all other Python jobs block.',
            commands=original['commands'], n1_followup=followup,
            conditional_full_plan=str(full_path)), indent=2)+'\n')

        def status(**values):
            payload = dict(pid=os.getpid(), updated_utc=datetime.datetime.now(
                datetime.timezone.utc).isoformat(), **values)
            temporary = status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2)+'\n')
            temporary.replace(status_path)

        def inspect_idle(_chain_path, own_pid):
            snapshot = process_snapshot()
            jobs, workers = classify(snapshot, own_pid)
            queue.verify({str(args.n1_evidence.resolve()): frozen[str(args.n1_evidence.resolve())]})
            return not jobs, dict(n1_state='stopped', python_jobs=jobs,
                                  excluded_transcript_workers=workers)

        def wait_idle(stage):
            consecutive = 0
            while consecutive < 2:
                ready, details = inspect_idle(None, os.getpid())
                consecutive = consecutive + 1 if ready else 0
                status(state='waiting', stage=stage, idle_samples=consecutive, **details)
                if consecutive < 2:
                    time.sleep(30)

        def run_stage(stage, command):
            status(state='running', stage=stage)
            subprocess.run([sys.executable, 'scripts/experiment.py', '--name', args.name+'-'+stage,
                '--strategy', 'Unchanged frozen SO1 objective comparison after N1 saved-checkpoint gate',
                '--', *command], check=True)

        try:
            if followup:
                import selection_objective_n1_followup as n1
                replacement = n1.execute(followup, run_stage, wait_idle,
                                         lambda: queue.verify(frozen))
                if replacement:
                    full_path = replacement
                    full_plan = json.loads(full_path.read_text())
                    frozen[str(full_path.resolve())] = queue.digest(full_path)
            queue.run_pair(plan, run_stage, wait_idle, lambda: queue.verify(frozen))
            gate = json.loads(gate_path.read_text())
            queue.verify(frozen)
            if not gate['passes_full_refit_gate']:
                status(state='complete', decision='stop: frozen P2 gate failed', primary=gate['primary'])
                return
            # Adapt only process discovery. All full-stage scientific checks,
            # two-snapshot waits, input hashes and gate recomputation remain.
            import selection_objective_full as full
            full.inspect_idle = inspect_idle
            status(state='running', stage='conditional-full-comparison')
            full.run(full_path)
            queue.verify(frozen)
            status(state='complete', comparison=str(Path(full_plan['output'])/'comparison.json'))
        except BaseException as error:
            status(state='failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
