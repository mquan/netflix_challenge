#!/usr/bin/env python3
"""Pause an exact owned trainer after a frozen prefix, then resume after exports."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    plan_path=Path(sys.argv[1]);plan=json.loads(plan_path.read_text());output=Path(plan['output'])
    if output.exists():raise ValueError('priority state exists')
    report=dict(state='waiting',plan=plan,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
    def status(state):
        report.update(state=state,updated_unix=time.time());tmp=output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(output)
        print(json.dumps(dict(state=state)),flush=True)
    def verify():
        command=subprocess.check_output(['ps','-p',str(plan['pid']),'-o','command='],text=True).strip()
        if command!=plan['command']:raise ValueError('trainer PID command differs')
    def interrupted(number,frame):raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,interrupted)
    deadline=time.monotonic()+4*3600;paused=False;completed=False
    status('waiting')
    try:
        while time.monotonic()<deadline:
            verify();path=Path(plan['capture_report'])
            captures=json.loads(path.read_text()).get('snapshots',{}) if path.exists() else {}
            if str(plan['epoch']) in captures:
                entry=captures[str(plan['epoch'])]
                if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:
                    raise ValueError('frozen checkpoint differs')
                verify();os.kill(plan['pid'],signal.SIGSTOP);paused=True;status('paused_for_fixed_prefix_exports');break
            time.sleep(10)
        else:raise TimeoutError('checkpoint wait expired')
        while time.monotonic()<deadline:
            verify();path=Path(plan['completion_report'])
            if path.exists() and json.loads(path.read_text()).get('state')=='complete':
                completed=True;break
            execution=Path(plan['completion_experiment'])
            if execution.exists() and json.loads(execution.read_text()).get('exit_code') not in [None,0]:
                raise ValueError('export coordinator failed')
            time.sleep(10)
        if not completed:raise TimeoutError('export wait expired')
    finally:
        if paused:
            verify();os.kill(plan['pid'],signal.SIGCONT)
            status('resumed_after_exports' if completed else 'resumed_after_interruption')


if __name__=='__main__':main()
