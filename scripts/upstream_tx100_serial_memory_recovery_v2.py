#!/usr/bin/env python3
"""Resume preserved TX after completed CPU diagnostics, retaining original guards."""
import signal
from pathlib import Path
import upstream_repro as u
import upstream_tx100_serial_memory_recovery as original

PRIOR = original.REPORT
REPORT = u.ROOT / 'experiments/upstream-tx100-serial-memory-recovery-v2'


def interrupted(signum, frame):
    raise KeyboardInterrupt('Serial recovery signal ' + str(signum))


def main():
    # Preserve the prior suspension and every original source/owner binding.
    prior = u.read(PRIOR / 'plan.json')
    if not (PRIOR / 'suspended.json').exists():
        raise ValueError('Actual prior resource suspension required')
    for name, digest in prior['source_sha256'].items():
        if u.sha(u.ROOT / name) != digest:
            raise ValueError('Original binding changed: ' + name)
    for key, expected in prior['original_processes'].items():
        if not original.owners.same_owner(expected, original.owners.process(expected['pid'])):
            raise ValueError('Original process identity changed: ' + key)
    for pid, token in [(95094, 'upstream_tx100_serial_memory_recovery.py'),
                       (96146, 'upstream_tx100_serial_memory_failsafe.py')]:
        actual = original.owners.process(pid)
        if actual and token in actual['command']:
            raise ValueError('Previous resource owner is still live')
    diagnostic = u.ROOT / 'experiments/upstream-own181-recent15-data/independent-verification.json'
    if u.read(diagnostic)['state'] != 'complete_all_own181_retained_added_operator_bits_original_statistics_and_ridge_replay':
        raise ValueError('Full CPU diagnostic replay must already be complete')
    original.REPORT = REPORT
    original.main()  # Identical admission/continuous limits and full checkpoint checks.
    complete = REPORT / 'completion.json'
    if complete.exists():
        result = u.read(complete)
        # Existing frozen downstream queues use this canonical handoff path.
        # Publish it only after the original verifier and native-exit gates.
        result.update(successor_completion_sha256=u.sha(complete),
                      successor_script_sha256=u.sha(Path(__file__)),
                      prior_suspension_sha256=u.sha(PRIOR / 'suspended.json'),
                      completed_cpu_diagnostic_sha256=u.sha(diagnostic))
        u.write(PRIOR / 'completion.json', result)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as error:
        for key in ['tx', 'tx_owner']:
            actual = original.owners.process(original.PIDS[key])
            if key in original.CAPTURED and original.owners.same_owner(original.CAPTURED[key], actual):
                original.os.kill(original.PIDS[key], signal.SIGSTOP)
        if REPORT.exists():
            u.write(REPORT / 'failure.json', dict(error=repr(error),
                    original_native_state_preserved=True, observed=original.stamp()))
        raise
