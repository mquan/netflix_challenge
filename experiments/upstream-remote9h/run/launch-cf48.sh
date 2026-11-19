#!/bin/bash
# Launch one supervised CFNADE48 resume phase. Usage: launch-cf48.sh x|qual
set -euo pipefail
PHASE=$1
R=$HOME/workspace/netflix_remote9h
P=$R/artifacts/upstream-remote9h/cfnade48-package
DEADLINE=2026-11-19T14:33:28Z   # actual start 2026-11-19T06:03:28Z + 8h30m
export CF_CHECKPOINT_DIR=$P/checkpoints
export CF_CHECKPOINT_SOURCE=e14530b5ade496efb3980a64f98442d35c9a606960add79149ce67b057677d94
export CF_CHECKPOINT_DATA=55ce479fc16447a826d013288b2ccb3e1eec39a07d7783f3d860a3f6e33e1fe0
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
unset CF_EXIT_AFTER_CHECKPOINT CF_PROOF_OUTPUT CF_PROOF_INTERRUPT
exec caffeinate -i ~/workspace/netflix_upstream_repro/.venv/bin/python $R/scripts/remote12h_supervisor.py \
  --name remote9h-cf48-$PHASE --out-dir $R/experiments/upstream-remote9h/run/cf48-$PHASE \
  --deadline $DEADLINE --cwd $P/native-source -- $P/bin/cfnade48-stage $PHASE
