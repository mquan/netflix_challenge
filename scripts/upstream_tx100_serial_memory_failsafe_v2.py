#!/usr/bin/env python3
"""Suspend only the original temporal job if its resource monitor disappears."""
import datetime as dt
import os
from pathlib import Path
import signal
import sys
import time
import upstream_repro as u
import upstream_cf96_primary_lane as owners

REPORT=u.ROOT/'experiments/upstream-tx100-serial-memory-failsafe-v2'
PRIMARY=u.ROOT/'experiments/upstream-tx100-serial-memory-recovery-v2'
CAPTURED={}

def suspend_originals(reason):
    stopped=[]
    for key,expected in CAPTURED.items():
        actual=owners.process(expected['pid'])
        if owners.same_owner(expected,actual):os.kill(expected['pid'],signal.SIGSTOP);stopped.append(expected['pid'])
    if REPORT.exists():u.write(REPORT/'suspension.json',dict(reason=reason,identity_checked_pids_suspended=stopped,observed=dt.datetime.now(dt.timezone.utc).isoformat(),all_native_RAM_state_preserved=True))
    return stopped

def main():
    REPORT.mkdir(exist_ok=False);plan=u.read(PRIMARY/'plan.json')
    CAPTURED.update({key:plan['original_processes'][key] for key in ['tx','tx_owner']})
    monitor=owners.process(int(sys.argv[1]))
    if not monitor or 'upstream_tx100_serial_memory_recovery_v2.py' not in monitor['command']:
        suspend_originals('Original resource monitor absent before fail-safe admission');return
    plan_sha=u.sha(PRIMARY/'plan.json')
    u.write(REPORT/'plan.json',dict(primary=monitor,originals=CAPTURED,primary_plan_sha256=plan_sha,script_sha256=u.sha(Path(__file__)),
        rule='Independent monitor-loss protection. If original guard disappears or PID identity changes without its complete original22/22 proof, SIGSTOP only matching original native and controller. Own SIGTERM/interrupt also suspends them through exception handler. No model restart, state loss or unrelated signal.'))
    while True:
        completion=PRIMARY/'completion.json'
        if completion.exists():
            proof=u.read(completion)
            if proof['state']!='complete_original_TX22_22_then_resumed_same_CF_native_and_owner':raise ValueError('Actual complete original monitor result required')
            if owners.same_owner(CAPTURED['tx'],owners.process(CAPTURED['tx']['pid'])):raise ValueError('Completed native must have exited')
            u.write(REPORT/'completion.json',dict(state='complete_resource_monitor_finished_full_original_TX',primary_completion_sha256=u.sha(completion)));return
        if u.sha(PRIMARY/'plan.json')!=plan_sha:raise ValueError('Original resource monitor ownership plan changed')
        if not owners.same_owner(monitor,owners.process(monitor['pid'])):
            suspend_originals('Original resource monitor terminated or PID identity changed before full completion');return
        u.write(REPORT/'status.json',dict(state='watching_original_resource_monitor_identity',pid=monitor['pid'],observed=dt.datetime.now(dt.timezone.utc).isoformat()))
        time.sleep(5)

if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('Fail-safe signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:main()
    except BaseException as error:suspend_originals(repr(error));raise
