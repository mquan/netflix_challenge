#!/usr/bin/env python3
"""Check each predeclared paired raw snapshot as it arrives; preserve immutable proofs."""
import fcntl
from pathlib import Path
import time
import verify_upstream_mps_item_autorec_snapshot as checker
model,u=checker.model,checker.u
REPORT=u.ROOT/'experiments/upstream-itemae-snapshot-watch'

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists():raise ValueError('Existing snapshot watch preserved')
        model.verify_plan();epochs=model.CONFIG['snapshot_epochs']
        paths=[Path(__file__),Path(checker.__file__),model.REPORT/'plan.json']
        hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
        u.write(REPORT/'plan.json',dict(created=model.stamp(),epochs=epochs,source_sha256=hashes,
            rule='Each already predeclared actual saved-model replayed pair gets the unchanged full independent checks as soon as ready. Successful evidence is immutable; repeated checks must agree on everything except invocation timestamp. Known-only component diagnostics do not select checkpoints or change the fixed5/10 neural study. Raw50/50 fitting continues.',qualifying_labels_read=False,public_prediction_values_read=False))
        for epoch in epochs:
            while not (model.REPORT/f'pair-{epoch:05}.json').exists():
                if (model.REPORT/'completion.json').exists():raise ValueError('Raw completion declared without expected pair')
                u.write(REPORT/'status.json',dict(state='waiting_for_actual_paired_snapshot',epoch=epoch,observed=model.stamp()));time.sleep(20)
            for name,digest in hashes.items():
                if u.sha(u.ROOT/name)!=digest:raise ValueError('Declared snapshot checker/source changed')
            checker.verify(epoch)
            path=model.REPORT/f'snapshot-{epoch:05}-verification.json';proof=u.read(path)
            u.write(REPORT/f'checked-{epoch:05}.json',dict(state='complete_original_independent_snapshot_check',observed=model.stamp(),epoch=epoch,
                verification_sha256=u.sha(path),known_component_rmse=proof['known_component_rmse'],global_ensemble_accuracy_claimed=False,qualifying_labels_read=False))
        u.write(REPORT/'completion.json',dict(state='all_predeclared_paired_raw_snapshots_independently_checked',observed=model.stamp(),epochs=epochs,qualifying_labels_read=False,goal_complete=False))

if __name__=='__main__':
    try:run()
    except BaseException as error:
        u.write(REPORT/'failure.json',dict(state='incomplete_preserved',observed=model.stamp(),error=repr(error)));raise
