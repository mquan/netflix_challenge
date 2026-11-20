#!/usr/bin/env python3
"""Preserve the exact own330 worker and release its idle native-queue parking."""
import os
import signal
import time
import upstream_repro as u
import upstream_cf96_primary_lane as owners

REPORT=u.ROOT/'experiments/upstream-own330-queue'


def preserve_worker():
    path=REPORT/'active-child.json'
    if path.exists():
        expected=u.read(path)['process'];actual=owners.process(expected['pid'])
        if owners.same_owner(expected,actual):
            if expected['pgid']!=expected['pid']:raise ValueError('Dedicated original worker group required')
            os.killpg(expected['pid'],signal.SIGSTOP)


def release_idle_queue(plan):
    expected=plan.get('parked_highcap')
    if expected:
        actual=owners.process(expected['pid'])
        if owners.same_owner(expected,actual) and 'T' in actual['state']:
            os.kill(expected['pid'],signal.SIGCONT)


def main():
    plan=u.read(REPORT/'plan.json');owner=plan['monitor']
    try:
        while owners.same_owner(owner,owners.process(owner['pid'])):
            if (REPORT/'owner-release.json').exists():
                if u.read(REPORT/'owner-release.json')['monitor']!=owner:raise ValueError('Original owner release identity differs')
                u.write(REPORT/'watchdog-completion.json',dict(state='original_owner_released_worker_and_idle_queue'));return
            time.sleep(2)
        preserve_worker();release_idle_queue(plan)
        u.write(REPORT/'watchdog-recovery.json',dict(state='owner_lost_original_worker_suspended_and_idle_native_queue_released',monitor=owner))
    except BaseException:
        preserve_worker();release_idle_queue(plan);raise


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('Own330 watchdog signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    main()
