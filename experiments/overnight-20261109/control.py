"""Continue until 07:00 Pacific, then pause; disk resume requires explicit instruction."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('disk_resume_helper', ROOT/'experiments/pause-20261109/control.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.HERE = HERE


def processes():
    text = subprocess.check_output(['ps', '-ww', '-axo', 'pid=,ppid=,state=,command='], text=True)
    result = {}
    for line in text.splitlines():
        a = line.strip().split(maxsplit=3)
        if len(a) == 4:
            result[int(a[0])] = dict(pid=int(a[0]), ppid=int(a[1]), state=a[2], command=a[3])
    return result


def descendants(table, roots):
    found = set(roots) & set(table)
    while True:
        expanded = found | {pid for pid, row in table.items() if row['ppid'] in found}
        if expanded == found:
            return found
        found = expanded


def matched_controller(row, current):
    if not current or helper.terminal(current['state']):
        return False
    started = subprocess.check_output(['ps','-p',str(row['pid']),'-o','lstart='],text=True).strip()
    if current['command'] != row['command'] or started != row['started']:
        return False  # completed controller PID was reused; never signal unrelated work
    return True


def verified_prefix(states):
    count = next((i for i, good in enumerate(states) if not good), len(states))
    if any(states[count:]):
        raise ValueError('Noncontiguous audited stages require manual review')
    return count


def pause():
    config = helper.read(HERE/'schedule.json')
    helper.verify_hashes(config['frozen'])
    if dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(config['cutoff_utc']):
        raise ValueError('Refusing to pause before requested cutoff')
    if (HERE/'ready.json').exists():
        return
    initial = processes()
    roots = [row['pid'] for row in config['controllers'] if matched_controller(row, initial.get(row['pid']))]
    helper.write(HERE/'cutoff-started.json', dict(state='pausing_at_requested_cutoff',
                 utc=dt.datetime.now(dt.timezone.utc).isoformat(), controllers=roots,
                 resume_policy='explicit_user_instruction_only'))
    # Freeze launchers first, then repeatedly collect their descendants to close fork races.
    for pid in roots:
        os.kill(pid, signal.SIGSTOP)
    frozen_processes = {}
    for _ in range(4):
        table = processes()
        for pid in descendants(table, roots):
            row = table[pid]
            if helper.terminal(row['state']):
                continue
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                continue
            row['started'] = subprocess.check_output(['ps','-p',str(pid),'-o','lstart='],text=True).strip()
            frozen_processes[pid] = row
    table = processes()
    for pid in descendants(table, roots):
        if not helper.terminal(table[pid]['state']) and 'T' not in table[pid]['state']:
            raise RuntimeError('Process not stopped: '+str(pid))
    helper.write(HERE/'stopped-processes.json', list(frozen_processes.values()))
    create_handoff(config)
    # Release only our original idle-sleep assertion. Explicit sleep/restart is now allowed.
    state, command = helper.process(71758)
    if command == '/usr/bin/caffeinate -i -w 71746':
        os.kill(71758, signal.SIGTERM)
    helper.update_status('PAUSED at 07:00 — explicit resume only',
        'Production and queued BDS were stopped at the requested cutoff. Completed audits are preserved. '
        'Any unfinished stage is listed in experiments/overnight-20261109/ready.json and may need '
        'retraining after restart; completed stages are never rerun. No automatic resume is configured. '
        'On explicit instruction use experiments/overnight-20261109/control.py --resume, not the cancelled immediate-drain helper.')
    helper.commit_records = lambda: None
    commit()
    print('PAUSED_AT_0700_EXPLICIT_RESUME_ONLY', flush=True)


def create_handoff(config):
    final = helper.read(ROOT/'experiments/P0-FINAL-20261031-plan.json')
    receipts, bound, valid = [], {}, set()
    # A receipt interrupted during its write is incomplete, never a completed checkpoint.
    for expected in final['resource_gate']['required_audits']:
        path = Path(expected['path'])
        if not path.exists():
            continue
        try:
            value = helper.read(path)
            if any(value.get(k) != v for k,v in expected.items() if k != 'path'):
                continue
            helper.verify_hashes(value['outputs'])
        except (ValueError, KeyError, OSError):
            continue
        valid.add(str(path))
        receipts.append(str(path))
        bound[str(path)] = helper.sha(path)
        bound.update(value['outputs'])
    runners, partial, frozen = [], [], dict(config['frozen'])
    for phase in config['phases']:
        path = ROOT/f'experiments/P0-{phase}-20261031-plan.json'
        source = ROOT/f'experiments/P0-{phase}-20261031-run.sh'
        plan = helper.read(path)
        frozen[str(path)], frozen[str(source)] = helper.sha(path), helper.sha(source)
        frozen.update(plan['frozen'])
        if phase == 'FINAL':
            # The production bank cannot finish overnight on the existing dependency chain.
            # If final work nevertheless started, require review rather than overwrite it.
            if Path(plan['final_model']).exists() or Path(plan['resolved_bank']).exists():
                raise ValueError('Final work unexpectedly started; preserve and review before resume')
            text, count = source.read_text(), 0
        else:
            states = [str(Path(s['output_dir'])/'audit.json') in valid for s in plan['stages']]
            count = verified_prefix(states)
            text = helper.tail_script(source.read_text(), count, len(plan['stages']))
            if text is not None:
                for stage in plan['stages'][count:]:
                    directory = Path(stage['output_dir'])
                    if directory.exists():
                        partial.append(dict(phase=phase, name=stage.get('name',stage.get('kind',str(count))),
                                            path=str(directory), reason='No complete verified audit at cutoff; preserve and restart this fixed stage on explicit resume.'))
        if text is None:
            continue
        dest = HERE/'resume'/source.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(text)
        subprocess.run(['/bin/bash','-n',str(dest)],check=True)
        frozen[str(dest)] = helper.sha(dest)
        runners.append(dict(phase=phase,script=str(dest),completed_stages=count))
    frozen.update(bound)
    helper.verify_hashes(frozen)
    helper.write(HERE/'ready.json', dict(state='paused_restart_safe',
        utc=dt.datetime.now(dt.timezone.utc).isoformat(), resume_policy='explicit_user_instruction_only',
        completed_audits=len(receipts), completed_receipts=receipts, partial_stages=partial,
        frozen=frozen, runners=runners, bds=next(r for r in config['controllers'] if r['name']=='BDS'),
        note='Disk resume preserves completed stages. Unfinished stages are archived and rerun at their original complete fixed budget; exact interrupted-optimizer continuation is not promised.'))
    subprocess.run(['/bin/sync'],check=True)


def resume():
    ready = helper.read(HERE/'ready.json')
    helper.verify_hashes(ready['frozen'])
    if (HERE/'resumed.json').exists():
        raise ValueError('Already resumed; inspect recorded handles')
    # Processes may still exist after sleep, or may be absent after reboot. Never touch reused PIDs.
    for row in helper.read(HERE/'stopped-processes.json'):
        state, command = helper.process(row['pid'])
        if helper.terminal(state):
            continue
        started = subprocess.check_output(['ps','-p',str(row['pid']),'-o','lstart='],text=True).strip()
        if command != row['command'] or started != row['started']:
            continue  # PID reused after restart: unrelated, never signal it.
        if 'T' not in state:
            raise ValueError('Recorded process resumed unexpectedly; review before launching')
        os.kill(row['pid'], signal.SIGKILL)
    archive = ROOT/'artifacts/production/P0-20261031/interrupted-at-0700-20261109'
    for index,row in enumerate(ready['partial_stages']):
        source = Path(row['path'])
        if source.exists():
            archive.mkdir(exist_ok=True)
            destination = archive/(str(index)+'-'+source.name)
            if destination.exists():
                raise ValueError('Partial-stage archive already exists; inspect before retry')
            source.rename(destination)
    helper.resume()
    helper.update_status('RESUMED by explicit instruction',
        'Resumed from experiments/overnight-20261109/ready.json. Completed stages skipped; unfinished '
        'stages archived and rerun at their original full budgets. New handles: experiments/overnight-20261109/resumed.json.')
    commit()


def commit():
    paths=['STATUS.md','experiments/overnight-20261109']
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','-c','user.name=Quan Nguyen','-c','user.email=mquannie@gmail.com',
        'commit','--only','-m','ops: record scheduled morning pause or explicit resume','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','push'],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--pause',action='store_true')
    modes.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:
        if args.pause:
            pause()
        else:
            resume()
    except Exception as error:
        helper.write(HERE/'error.json',dict(state='needs_review',error=repr(error),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise
