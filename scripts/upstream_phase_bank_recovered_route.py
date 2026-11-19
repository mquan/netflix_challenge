#!/usr/bin/env python3
"""Resolve the unchanged phase-bank assembly with our completed remote source job."""
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import time
import upstream_phase_bank as bank
import upstream_remote_temporal_pair as temporal
import upstream_recovered_temporal_bank as verified

u=bank.u
REPORT=u.ROOT/'experiments/upstream-phase-bank-recovered-route'
DESTINATION=bank.REPORT


def watcher():
    row=subprocess.run(['ps','-p','46595','-o','stat=,command='],capture_output=True,text=True).stdout.strip()
    if not row or 'scripts/upstream_phase_bank.py --watch' not in row:raise ValueError('Original bank watcher identity differs')
    if subprocess.run(['pgrep','-P','46595'],capture_output=True,text=True).stdout.strip():raise ValueError('Original bank watcher has a child')
    return row


def run():
    if (DESTINATION/'manifest.json').exists() or bank.RUN.exists():raise ValueError('Existing original bank/cache preserved')
    if REPORT.exists():raise ValueError('Existing recovered-route attempt preserved')
    old=watcher();os.kill(46595,signal.SIGSTOP);published=False
    try:
        stopped=watcher()
        if 'T' not in stopped.split()[0]:raise ValueError('Original bank watcher is not stopped')
        if (DESTINATION/'manifest.json').exists() or bank.RUN.exists():raise ValueError('Original bank started before the idle hold')
        REPORT.mkdir(parents=True)
        reference=verified.build();receipt=temporal.verify_receipt()
        # Execute the ORIGINAL assembly function with an explicit, compatible
        # receipt provider. Its source hash remains accurate; no fake local
        # completion receipt is written. Stage metadata until all checks pass.
        bank.phase=temporal;bank.REPORT=REPORT/'staging';bank.PRIVATE=bank.REPORT/'assembly'
        result=bank.build()
        if result['files']!=reference['files']:
            raise ValueError('Original-schema cache differs from the independently verified recovered bank')
        result['source_sha256'][str(Path(__file__).relative_to(u.ROOT))]=u.sha(Path(__file__))
        result.update(temporal_provenance=verified.PROVENANCE,
            dependency_route='Unchanged original phase-bank assembly; completed own remote native receipt provider. The independent local native job remains active and is not reported complete.',
            dependency_adapter_sha256=u.sha(Path(__file__)),
            all_six_cache_files_identical_to_verified_recovered_bank=True,
            original_remote_transfer_commit=receipt['transfer_commit'])
        u.write(REPORT/'staging/manifest.json',result)
        inventory=bank.inventory(True);u.write(DESTINATION/'inventory.json',inventory)
        u.write(DESTINATION/'manifest.json',result);published=True
        os.kill(46595,signal.SIGTERM);os.kill(46595,signal.SIGCONT)
        for _ in range(100):
            if not subprocess.run(['ps','-p','46595','-o','pid='],capture_output=True,text=True).stdout.strip():break
            time.sleep(.1)
        else:raise RuntimeError('Old bank watcher still exiting; do not launch another producer')
        u.write(REPORT/'verification.json',dict(state='complete',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
            original_watcher=old,held_watcher=stopped,old_watcher_exited=True,
            original_bank_manifest_sha256=u.sha(DESTINATION/'manifest.json'),
            reference_manifest_sha256=u.sha(verified.REPORT/'manifest.json'),cache_files=result['files'],
            all_six_cache_files_bitwise_identical=True,native_processes_signalled=False,
            fake_local_completion_receipt_created=False,qualifying_labels_read=False))
        print('Original phase-bank schema complete; all six cache files identical to verified own recovered bank',flush=True)
    except BaseException:
        # Restore only the idle watcher if publication failed. Preserve all
        # partial cache/evidence files; native training is never controlled here.
        if not published:os.kill(46595,signal.SIGCONT)
        raise


if __name__=='__main__':run()
