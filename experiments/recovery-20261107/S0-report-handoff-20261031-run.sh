#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "S0 report handoff exited with status $?"' EXIT
echo WAIT_S0_LANE_RUNNERS_39539_AND_39615
while pgrep -f '[S]0-lane0-20261031-run.sh' >/dev/null || pgrep -f '[S]0-lane1-20261031-run.sh' >/dev/null; do
    sleep 30
done
shasum -a 256 --check experiments/S0-report-handoff-20261031.sha256
echo START_S0_ALL_FIXED_ARMS_AUDIT_AND_REPORT
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python scripts/structural_scoping_report.py \
    --execution experiments/S0-execution-document-rebind-20261031.json \
    --parity experiments/S0-full-parity-verification-20261031.json \
    --binding experiments/S0-report-binding-20261031.json \
    --output artifacts/structural/S0-scoping-20261031/final-summary.json
echo COMPLETE_S0_REPORT_NO_AUTOMATIC_TRAINING_OR_PROMOTION
