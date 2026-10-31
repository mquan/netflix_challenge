#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "E3 gate handoff exited with status $?"' EXIT
echo WAIT_E3_TRAINER_38985_AND_RUNNER_38977
while kill -0 38985 2>/dev/null || kill -0 38977 2>/dev/null; do
    sleep 30
done
echo START_E3_COMPLETION_AUDIT_AND_BOTH_REGISTERED_GATES
shasum -a 256 --check experiments/E3-gate-handoff-20261031.sha256
test ! -e artifacts/ordinal/E3-gate-20261031
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python scripts/e3_candidate_gate.py \
    --plan experiments/E3-plan-20261031.json \
    --output-dir artifacts/ordinal/E3-gate-20261031
echo COMPLETE_E3_GATE_HANDOFF_NO_AUTOMATIC_FOLLOWUP
