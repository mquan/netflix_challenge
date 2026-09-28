#!/usr/bin/env bash
set -euo pipefail
DATA=/Users/quan/workspace/netflix_challenge/data/processed/netflix.bin
# Equal active factor dimensions per rating; local model has more total parameters.
for bet in global local; do
  if [[ "$bet" == global ]]; then groups=1; rank=16; else groups=8; rank=8; fi
  python3 scripts/experiment.py --name "L1-$bet" \
    --strategy "Fixed six epochs of residual factorization: groups=$groups rank=$rank. Compare local routing with global residual capacity at equal active dimensions." -- \
    build/moonshot_local --data "$DATA" --model artifacts/snapshot/joint128-reg005.bin \
    --output "artifacts/local-$bet" --groups "$groups" --rank "$rank" --epochs 6
done
