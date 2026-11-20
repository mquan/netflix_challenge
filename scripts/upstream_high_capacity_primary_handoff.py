#!/usr/bin/env python3
"""Retire only the restored idle scheduler after its current native receipt completes."""
import datetime as dt
from pathlib import Path
import time
import upstream_repro as u
import upstream_cf96_primary_lane as owners
import upstream_full_reproduce as primary

REPORT=u.ROOT/'experiments/upstream-high-capacity-primary-handoff'
PRIMARY=12394
HIGH=61480


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    if REPORT.exists():raise ValueError('Existing full/partial primary handoff retained')
    report=REPORT;report.mkdir();owner=owners.process(PRIMARY);higher=owners.process(HIGH)
    if not owner or str(owners.PRIMARY_SCRIPT) not in owner['command']:raise ValueError('Exact restored original scheduler required')
    if not higher or 'upstream_high_capacity_successor.py' not in higher['command']:raise ValueError('Exact reviewed original1600 queue required')
    restored=u.ROOT/'experiments/upstream-cf96-primary-lane/restored-primary.json'
    receipt=u.ROOT/'experiments/upstream-cf96-disk-resume/recovery-completion.json'
    if u.read(restored)['pid']!=PRIMARY or u.read(receipt)['state']!='complete_exact95_95_and_original_finalizer':
        raise ValueError('Completed original CF95 and its actual restored primary required')
    files=[Path(__file__),Path(owners.__file__),Path(primary.__file__),owners.PRIMARY_SCRIPT,restored,receipt]
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    u.write(report/'plan.json',dict(created=stamp(),primary_owner=owner,high_capacity_owner=higher,source_sha256=bindings,
        rule='Original CF95/95 and canonical finalizer complete. Restored primary reclaimed the lane. Let any native child finish; retire ONLY exact scheduler at a complete receipted idle boundary, rechecked while scheduler frozen. Existing stop-only-if-idle sends cleanup SIGTERM/CONT solely to childless scheduler. Preserve all numerical recipes/data/outputs. Reviewed1600 queue keeps its original22GiB/24GiB/disk/lane guards. Do not restart or interrupt native training. Original scheduler command retained for later safe continuation after higher-priority work.'))
    jobs=u.read(u.BASE/'registry.json')['jobs']
    while True:
        for relative,digest in bindings.items():
            if u.sha(u.ROOT/relative)!=digest:raise ValueError('Original handoff source identity changed')
        if not owners.same_owner(higher,owners.process(HIGH)):raise ValueError('Original high-capacity owner changed before handoff')
        if not owners.same_owner(owner,owners.process(PRIMARY)):
            u.write(report/'completion.json',dict(state='original_primary_already_terminal_no_signal_sent',observed=stamp()));return
        state=u.read(primary.REPORT/'status.json');job=state.get('current');children=owners.children(PRIMARY)
        u.write(report/'status.json',dict(state='waiting_completed_native_receipt_and_childless_scheduler',job=job,
            primary_pid=PRIMARY,live_children=children,observed=stamp()))
        if job in jobs and not children and primary.receipt_valid(job,jobs):
            frozen_status=u.sha(primary.REPORT/'status.json')
            captured=owners.stop_only_if_idle(PRIMARY,str(owners.PRIMARY_SCRIPT),
                lambda:owners.same_owner(owner,owners.process(PRIMARY)) and u.sha(primary.REPORT/'status.json')==frozen_status and primary.receipt_valid(job,jobs))
            if captured:
                proof=u.BASE/'fresh-receipts'/(job+'.json')
                u.write(report/'completion.json',dict(state='complete_verified_idle_primary_handoff_to_existing_original1600_queue',
                    observed=stamp(),retired_scheduler=captured,completed_native_job=job,
                    complete_native_receipt_sha256=u.sha(proof),no_live_child_at_stop=True,
                    original_high_capacity_owner=higher,all_native_outputs_and_states_preserved=True,
                    original_primary_restart_command=[str(u.ROOT/'.venv/bin/python'),'-u',str(owners.PRIMARY_SCRIPT)]))
                print('Original completed-job idle handoff:',job,'scheduler',PRIMARY,'retired; existing1600 queue retained',flush=True);return
        time.sleep(1)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if REPORT.exists():u.write(REPORT/'failure.json',dict(error=repr(exc),observed=stamp(),no_native_restart=True))
        raise
