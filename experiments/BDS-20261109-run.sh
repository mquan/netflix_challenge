#!/bin/bash
set -u
cd /Users/quan/workspace/netflix_challenge
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
LOG=artifacts/blend-scaling/queue.log
# 2026-11-09: started by explicit user instruction while production is paused; no production guard.
echo "$(date -u +%FT%TZ) START" >> $LOG
for F in 1.0 0.75 0.5; do
    OUT=artifacts/blend-scaling/BDS-k$F-s3
    [ -e $OUT.json ] && { echo "$(date -u +%FT%TZ) SKIP $F exists" >> $LOG; continue; }
    echo "$(date -u +%FT%TZ) ARM $F" >> $LOG
    (cd scripts && nice -n 10 /Users/quan/.cache/netflix-venv/bin/python blend_data_scaling.py --keep-fraction $F \
        --data ../data/processed/netflix.bin --output ../$OUT \
        --bank-root ../artifacts/structural/S0-scoping-20261031/banks/all79 \
        --rounds 8000 --learning-rate 0.02 --leaves 63 --min-leaf 200 --processes 3 \
        --seeds 2026 7 13 --extra-meta) > $OUT.log 2>&1
    echo "$(date -u +%FT%TZ) ARM $F exit $?" >> $LOG
done
/Users/quan/.cache/netflix-venv/bin/python - <<'PY' >> $LOG 2>&1
import json, pathlib
ref = json.load(open('artifacts/structural/S0-scoping-20261031/results/all79-s3.json'))['refit_rmse']
r = {f: json.load(open(f'artifacts/blend-scaling/BDS-k{f}-s3.json'))['refit_rmse'] for f in ('1.0', '0.75', '0.5')
     if pathlib.Path(f'artifacts/blend-scaling/BDS-k{f}-s3.json').exists()}
out = dict(reference=ref, refit=r, parity_exact=r.get('1.0') == ref)
if '0.5' in r and '1.0' in r:
    out['delta50'] = r['0.5'] - r['1.0']
    out['followup_warranted'] = bool(out['parity_exact'] and out['delta50'] >= 0.0006)
if '0.75' in r and '1.0' in r:
    out['delta75'] = r['0.75'] - r['1.0']
pathlib.Path('artifacts/blend-scaling/BDS-summary.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
PY
echo "$(date -u +%FT%TZ) DONE" >> $LOG
