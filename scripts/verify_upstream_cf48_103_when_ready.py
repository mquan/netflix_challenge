#!/usr/bin/env python3
"""Wait for the owned production queue; separately verify any completed audit."""
import datetime as dt
import fcntl
from pathlib import Path
import upstream_repro as u
import verify_upstream_frozen_mean as checker

REPORT=u.ROOT/'experiments/upstream-cf48-103-production'

def run():
    with (REPORT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        status=u.read(REPORT/'queue-status.json')
        if status['state']=='not_promoted_development_did_not_improve':
            result=dict(state='not_needed_no_qualifying_audit',development_rmse=status['rmse'])
        elif status['state']=='audit_complete_pending_separate_frozen_mean_verification':
            checker.verify('upstream_cf48_103_production')
            proof=REPORT/'frozen-mean-score-verification.json'
            result=dict(state='separately_verified',proof_sha256=u.sha(proof),
                both_groups_below_target=u.read(proof)['both_groups_below_target'])
        else:
            raise RuntimeError('Production queue ended without an eligible terminal state; outputs preserved: '+status['state'])
        result.update(observed=dt.datetime.now(dt.timezone.utc).isoformat(),watcher_sha256=u.sha(Path(__file__)))
        u.write(REPORT/'separate-verification-status.json',result);print(result,flush=True)

if __name__=='__main__':run()
