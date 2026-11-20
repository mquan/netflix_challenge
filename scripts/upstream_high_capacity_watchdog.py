#!/usr/bin/env python3
"""Preserve an exact native worker if its high-capacity resource owner exits."""
import os
from pathlib import Path
import signal
import sys
import time
import upstream_repro as u
import upstream_cf96_primary_lane as identities

REPORT = u.ROOT / 'experiments/upstream-high-capacity-original'
CAPTURED = None


def suspend(reason):
    if CAPTURED is not None and identities.same_owner(CAPTURED, identities.process(CAPTURED['pid'])):
        os.killpg(CAPTURED['pid'], signal.SIGSTOP)
        u.write(REPORT / 'watchdog-suspension.json', dict(reason=reason, original_native=CAPTURED,
                                                       all_native_RAM_state_preserved=True))


def main():
    global CAPTURED
    ownership = u.read(REPORT / 'native-ownership.json')
    owner = ownership['monitor']
    CAPTURED = ownership['native']
    if CAPTURED['pgid'] != CAPTURED['pid']:
        raise ValueError('Dedicated native process group required')
    while identities.same_owner(CAPTURED, identities.process(CAPTURED['pid'])):
        if not identities.same_owner(owner, identities.process(owner['pid'])):
            suspend('Exact resource-owner process disappeared');return
        time.sleep(2)
    u.write(REPORT / 'watchdog-completion.json', dict(state='original_native_process_terminal', native=CAPTURED))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Native watchdog signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as error:
        suspend(repr(error));raise
