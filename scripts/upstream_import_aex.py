#!/usr/bin/env python3
"""Reuse complete AEX128 only after the preceding primary transition finishes."""
import fcntl
from pathlib import Path
import time
from types import SimpleNamespace
import upstream_repro as u
import upstream_aex_backfill as aex
import upstream_import_backfill as transfer
import upstream_primary_handoff as handoff

REPORT=u.ROOT/'experiments/upstream-import-aex'


def prior_transition():
    previous=u.ROOT/'experiments/upstream-import-backfill'
    if not (previous/'result.json').exists() or not (previous/'status.json').exists():return None
    if u.read(previous/'status.json').get('state')!='complete':return None
    if not (handoff.REPORT/'result.json').exists():return None
    result=u.read(handoff.REPORT/'result.json')
    if (result['state']!='successor_launched' or
        result['auxiliary_import_sha256']!=u.sha(previous/'result.json') or
        result['launch']['pid']!=u.read(transfer.PRIMARY/'launch.json')['pid']):
        raise ValueError('Primary successor handoff identity differs')
    return result


def configure():
    # The unchanged transfer implementation retains its transactional hardlink,
    # native receipt/source checks and scheduler-only STOP/CONT recovery.
    transfer.REPORT=REPORT
    transfer.auxiliary=SimpleNamespace(BASE=aex.BASE,TREE=aex.TREE,REPORT=aex.REPORT)
    transfer.followon=SimpleNamespace(REPORT=aex.REPORT)
    transfer.PRIMARY_FRAGMENT='scripts/upstream_primary_guarded.py'


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Serialize with both existing mutators. They own their original locks
        # and must publish completed results before this adapter may proceed.
        while prior_transition() is None:
            u.write(REPORT/'queue-status.json',dict(state='waiting_for_prior_import_and_handoff'))
            time.sleep(10)
        old=handoff.REPORT/'result.json';result=u.read(old)
        plan=dict(adapter_sha256=u.sha(Path(__file__)),transfer_sha256=u.sha(Path(transfer.__file__)),
            source_runner_sha256=u.sha(Path(aex.__file__)),prior_handoff_sha256=u.sha(old),
            source_tree=str(aex.TREE),primary_launch=u.read(transfer.PRIMARY/'launch.json'),qualifying_labels_read=False)
        p=REPORT/'adapter-plan.json'
        if p.exists() and u.read(p)!=plan:raise ValueError('AEX import adapter plan changed')
        u.write(p,plan)
        configure();transfer.run()
        u.write(REPORT/'queue-status.json',dict(state='complete',adapter_plan_sha256=u.sha(p),goal_complete=False))


if __name__=='__main__':
    try:run()
    except BaseException as error:
        if not isinstance(error,BlockingIOError):u.write(REPORT/'queue-status.json',dict(state='incomplete_preserved',error=repr(error)))
        raise
