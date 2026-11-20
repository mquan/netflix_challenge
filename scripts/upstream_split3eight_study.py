#!/usr/bin/env python3
"""Declare and execute complete eight-column data proofs before full neural fitting."""
from pathlib import Path
import datetime as dt
import subprocess
import sys
import upstream_split3_eight_screen as screen
import upstream_split3eight_compact_queue as queue
u=screen.u
REPORT=u.ROOT/'experiments/upstream-split3eight-study'
STAGES=['verify_upstream_split3_eight_screen.py','upstream_split3eight_virtual_features.py',
        'upstream_split3eight_scaling_parity.py','upstream_split3eight_compact_queue.py']

def main():
    if REPORT.exists():raise ValueError('Existing study evidence preserved; no automatic restart')
    REPORT.mkdir()
    queue.REPORT.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),*[u.ROOT/'scripts'/s for s in STAGES],
        *[u.ROOT/'scripts'/s for s in ['upstream_split3_eight_screen.py',
          'upstream_split3eight_virtual_mps_context.py','upstream_split3eight_compact_mps_context.py',
          'upstream_rbm50_tree_split3eight_compact_mps.py','upstream_item154_tree_split3eight_compact_mps.py']]]
    declaration=dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
        code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths},
        fixed_addition=screen.NEW,reference_development_rmse=.8592012938198756,
        rows=1128164,inputs=406,hidden=128,context_rank=32,
        models_fitted=False,qualifying_labels_read=False,automatic_production=False)
    u.write(queue.REPORT/'declaration.json',declaration)
    for script in STAGES:
        for name,digest in declaration['code_sha256'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Declared implementation changed')
        u.write(REPORT/'status.json',dict(stage=script,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
        with (REPORT/(script+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    u.write(REPORT/'completion.json',dict(state='complete_all_eight_full_study',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        completion_sha256=u.sha(queue.REPORT/'completion.json'),qualifying_labels_read=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        REPORT.mkdir(parents=True,exist_ok=True)
        u.write(REPORT/'failure.json',dict(error=repr(error),artifacts_preserved=True))
        raise
