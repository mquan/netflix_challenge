"""Restart-safe stage drain. Resume is never automatic and requires --resume."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = '/Users/quan/.cache/netflix-venv/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def process(pid):
    result = subprocess.run(['ps', '-ww', '-p', str(pid), '-o', 'state=,command='],
                            text=True, capture_output=True)
    parts = result.stdout.strip().split(maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else ('', '')


def terminal(state):
    return not state or 'Z' in state


def tail_script(text, completed, total):
    """Keep original setup and exact remaining command suffix, never replay a stage."""
    lines = text.splitlines()
    ends = [i for i, line in enumerate(lines) if line.startswith('echo COMPLETE_AUDITED_')]
    if len(ends) != total or not 0 <= completed <= total:
        raise ValueError('Stage completion markers differ')
    if completed == total:
        return None
    if completed == 0:
        return text
    first_check = next(i for i, line in enumerate(lines) if line.startswith('shasum '))
    return '\n'.join(lines[:first_check] + lines[ends[completed-1]+1:]) + '\n'


def receipt_prefix(plan):
    states = [(Path(s['output_dir']) / 'audit.json').is_file() for s in plan['stages']]
    count = next((i for i, value in enumerate(states) if not value), len(states))
    if any(states[count:]):
        raise ValueError('Noncontiguous receipts require explicit review')
    return count


def verify_hashes(values):
    for path, expected in values.items():
        if sha(path) != expected:
            raise ValueError('Hash differs: ' + path)


def drain():
    config = read(HERE / 'drain-plan.json')
    verify_hashes(config['control_inputs'])
    if (HERE / 'ready.json').exists():
        return 0
    now = dt.datetime.now(dt.timezone.utc)
    pending = []
    for stage in config['active_stages']:
        state, command = process(stage['pid'])
        if not terminal(state):
            if command != stage['process_command']:
                raise ValueError('Active stage PID identity changed')
            pending.append(stage)
            continue
        receipt = Path(stage['audit'])
        if not receipt.exists():
            # Only original post-training reload/audit commands; no training commands.
            for index, command in enumerate(stage['completion_commands']):
                out = HERE / (stage['phase'] + '-completion-' + str(index) + '.log')
                with out.open('ab') as stream:
                    result = subprocess.run(['/bin/bash', '-c',
                        'set -eu\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1\n' + command],
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError('Original completion command failed: ' + stage['phase'])
        if read(receipt).get('state') != 'verified_complete':
            raise ValueError('Complete audited stage required')
    if pending:
        if now >= dt.datetime.fromisoformat(config['deadline_utc']):
            for stage in pending:
                os.kill(stage['pid'], signal.SIGSTOP)
            write(HERE / 'deadline-not-ready.json', dict(state='deadline_reached_with_incomplete_stages',
                  pending=pending, warning='Processes suspended, not safely disk-resumable. Review before restart.'))
            raise RuntimeError('Cutoff reached before complete audits')
        print(json.dumps(dict(state='draining_active_stages', pending=[s['phase'] for s in pending])), flush=True)
        return 1
    return finalize(config)


def finalize(config):
    final = read(ROOT / 'experiments/P0-FINAL-20261031-plan.json')
    bound, completed = {}, []
    for expected in final['resource_gate']['required_audits']:
        path = Path(expected['path'])
        if not path.exists():
            continue
        actual = read(path)
        if any(actual.get(k) != value for k, value in expected.items() if k != 'path'):
            raise ValueError('Production receipt differs: ' + str(path))
        bound[str(path)] = sha(path)
        bound.update(actual['outputs'])
        completed.append(str(path))
    verify_hashes(bound)
    runners, frozen = [], dict(config['control_inputs'])
    for phase in config['phases']:
        plan_path = ROOT / f'experiments/P0-{phase}-20261031-plan.json'
        source = ROOT / f'experiments/P0-{phase}-20261031-run.sh'
        frozen[str(plan_path)] = sha(plan_path)
        frozen[str(source)] = sha(source)
        plan = read(plan_path)
        frozen.update(plan['frozen'])
        if phase == 'FINAL':
            text, count = source.read_text(), 0
        else:
            count = receipt_prefix(plan)
            text = tail_script(source.read_text(), count, len(plan['stages']))
            if text is not None and Path(plan['stages'][count]['output_dir']).exists():
                raise ValueError('Next stage already has partial output: ' + phase)
        if text is None:
            continue
        dest = HERE / 'resume' / source.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(text)
        subprocess.run(['/bin/bash', '-n', str(dest)], check=True)
        frozen[str(dest)] = sha(dest)
        runners.append(dict(phase=phase, script=str(dest), completed_stages=count))
    verify_hashes(frozen)
    # Controllers are stopped; active child stages are now terminal and audited.
    # Never continue old controllers: terminate only after rechecking identity.
    hold = read(HERE / 'hold.json')
    for row in hold['controllers']:
        state, command = process(row['pid'])
        if terminal(state):
            continue
        started = subprocess.check_output(['ps', '-p', str(row['pid']), '-o', 'lstart='], text=True).strip()
        if command != row['command'] or started != row['started'] or 'T' not in state:
            raise ValueError('Held controller identity/state differs')
    for row in hold['controllers']:
        state, _ = process(row['pid'])
        if not terminal(state):
            os.kill(row['pid'], signal.SIGKILL)  # idle held controller only, never a trainer
    frozen.update(bound)
    # Preserve BDS as queued intent only. Its original script is not edited.
    bds = next(r for r in hold['controllers'] if r['name'] == 'BDS')
    frozen[str(ROOT / bds['script'])] = sha(ROOT / bds['script'])
    manifest = dict(state='paused_restart_safe', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    resume_policy='explicit_user_instruction_only', completed_audits=len(completed),
                    completed_receipts=completed, frozen=frozen, runners=runners, bds=bds,
                    note='All active stages completed/reloaded/audited; remaining scripts start at next untouched stages. No automatic wake/reboot launch installed.')
    subprocess.run(['/bin/sync'], check=True)
    write(HERE / 'ready.json', manifest)
    update_status('PAUSED — restart-safe',
                  f"All active stages completed and audited; {len(completed)}/82 production audits preserved. "
                  'Controllers exited; sleep/restart safe for these runs. Resume only on explicit user instruction. '
                  'Resume manifest: experiments/pause-20261109/ready.json.')
    commit_records()
    print('PAUSED_RESTART_SAFE_EXPLICIT_RESUME_ONLY', flush=True)
    return 0


def update_status(title, message):
    path = ROOT / 'STATUS.md'
    text = path.read_text()
    start = text.index('## Planned pause')
    end = text.index('## Recent results', start)
    text = text[:start] + '## Planned pause — ' + title + '\n\n' + message + '\n\n' + text[end:]
    path.write_text(text)


def commit_records():
    paths = ['STATUS.md', 'experiments/pause-20261109']
    subprocess.run(['git', 'add', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', '-c', 'user.name=Quan Nguyen', '-c', 'user.email=mquannie@gmail.com',
                    'commit', '--only', '-m', 'ops: record restart-safe paused production handoff', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'push'], cwd=ROOT, check=True)


def resume():
    """Call only after a new, explicit user instruction to resume."""
    ready = read(HERE / 'ready.json')
    if ready['state'] != 'paused_restart_safe' or (HERE / 'resumed.json').exists():
        raise ValueError('Not an unused paused handoff')
    verify_hashes(ready['frozen'])
    if subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip() != '1':
        raise ValueError('Normal memory required')
    processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
    if any('/bin/bash ' + row['script'] in processes for row in ready['runners']) or any(
            line.startswith(('/bin/bash ', '/Users/quan/', '/opt/homebrew/')) and
            ('scripts/production_neural.py --column ' in line or
             'artifacts/production/P0-20261031/' in line and 'netflix ' in line)
            for line in processes.splitlines()):
        raise ValueError('Resume runner already live')
    results = dict(state='resuming_explicitly', utc=dt.datetime.now(dt.timezone.utc).isoformat(), queues=[])
    with (HERE / 'resumed.json').open('x') as stream:
        json.dump(results, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for row in ready['runners']:
        log = ROOT / 'artifacts/production/P0-20261031' / ('resume-20261109-' + row['phase'] + '.log')
        with log.open('ab') as out:
            proc = subprocess.Popen(['/bin/bash', row['script']], cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        results['queues'].append(dict(row, pid=proc.pid, log=str(log)))
        write(HERE / 'resumed.json', results)
    # User's resume also restores the previously queued diagnostic, still behind production.
    with (ROOT / 'artifacts/blend-scaling/resume-20261109-queue.log').open('ab') as out:
        proc = subprocess.Popen(['/bin/bash', ready['bds']['script']], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    results['bds_pid'] = proc.pid
    final_pid = next(q['pid'] for q in results['queues'] if q['phase'] == 'FINAL')
    proc = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(final_pid)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    results.update(state='resumed_explicitly', sleep_assertion_pid=proc.pid)
    write(HERE / 'resumed.json', results)
    print('RESUMED_AFTER_EXPLICIT_USER_INSTRUCTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--drain', action='store_true')
    group.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        if args.resume:
            resume()
        else:
            raise SystemExit(drain())
    except Exception as error:
        write(HERE / 'error.json', dict(state='needs_review', error=repr(error), utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(repr(error), flush=True)
        raise SystemExit(2)
