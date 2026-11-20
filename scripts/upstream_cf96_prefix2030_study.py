#!/usr/bin/env python3
"""Execute both fixed own CF96 prefix additions without altering native95/95 fits."""
from pathlib import Path
import subprocess
import sys
import time
import upstream_cf96_prefix2030_screen as screen
import upstream_cf96_prefix2030_compact_queue as queue
u=screen.u
REPORT=u.ROOT/'experiments/upstream-cf96-prefix2030-study'
STAGES=['upstream_cf96_prefix2030_screen.py','verify_upstream_cf96_prefix2030_screen.py',
    'upstream_cf96_prefix2030_virtual_features.py','upstream_cf96_prefix2030_scaling_parity.py',
    'upstream_cf96_prefix2030_compact_queue.py']

def main():
    if REPORT.exists():raise ValueError('Existing complete prefix study evidence preserved')
    REPORT.mkdir();queue.REPORT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),*[u.ROOT/'scripts'/s for s in STAGES],
        *[u.ROOT/'scripts'/s for s in ['upstream_cf96_prefix2030_source.py',
         'upstream_cf96_prefix2030_virtual_mps_context.py','upstream_cf96_prefix2030_compact_mps_context.py',
         'upstream_rbm50_tree_cf96_prefix2030_compact_mps.py','upstream_item154_tree_cf96_prefix2030_compact_mps.py']]]
    declaration=dict(created=queue.stamp(),code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in files},
        fixed_addition=screen.NEW,reference_development_rmse=.8592012938198756,
        rows=1128164,inputs=400,hidden=128,context_rank=32,
        original_native95_budget_unchanged_and_continuing=True,full95_model_complete=False,
        qualifying_labels_read=False,models_fitted=False,automatic_production=False)
    u.write(queue.REPORT/'declaration.json',declaration)
    while not (screen.prefix.REPORT/'paired-receipt.json').exists():
        u.write(REPORT/'status.json',dict(state='waiting_for_reverified_own_original_prefix_source',observed=queue.stamp()))
        time.sleep(20)
    for script in STAGES:
        for name,digest in declaration['code_sha256'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Declared prefix implementation changed')
        u.write(REPORT/'status.json',dict(stage=script,observed=queue.stamp()))
        with (REPORT/(script+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts'/script)],cwd=u.ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    u.write(REPORT/'completion.json',dict(state='complete_full_prefix_neural_study',observed=queue.stamp(),
        completion_sha256=u.sha(queue.REPORT/'completion.json'),qualifying_labels_read=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        REPORT.mkdir(parents=True,exist_ok=True)
        u.write(REPORT/'failure.json',dict(error=repr(error),artifacts_preserved=True));raise
