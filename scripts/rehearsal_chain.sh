#!/usr/bin/env bash
# Phase 0 Stage C rehearsal: dependent steps after the (a) trainers finish.
# usage: scripts/rehearsal_chain.sh k1|rbm
#   k1   waits for P0-A-K1; scores it on the original probe, builds (a) kNN
#        columns, retrains K1 on all training ratings at the P2-selected
#        epoch (b) and builds (b) kNN columns.
#   rbm  waits for P0-A-rbm100; trains the conditional stage on train minus P2
#        (a) while retraining the plain stage on all training ratings (b) at
#        the P2-selected epoch, then the (b) conditional stage.
# Epochs are read from the (a) logs, i.e. chosen on P2; probe labels choose nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
D=artifacts/validation/train-matched-20260927.bin     # probe = P2
DP=artifacts/rehearsal/matched-train-original-probe.bin  # same training, original probe
FULL=data/processed/netflix.bin
R=artifacts/rehearsal
run() { local name=$1 strategy=$2; shift 2; python3 scripts/experiment.py --name "$name" --strategy "$strategy" -- "$@"; }
wait_for() { while pgrep -f -- "--name $1 " > /dev/null; do sleep 30; done; }
best_epoch() {  # last epoch whose probe_rmse equals the running best
  grep -E '^epoch=' "$1" | sed 's/.*epoch=\([0-9]*\).*probe_rmse=\([0-9.]*\) best=\([0-9.]*\).*/\1 \2 \3/' |
    awk '$2 == $3 {e = $1} END {print e}'
}
K1_ARGS=(--factors 256 --learning-rate .008 --regularization .02 --decay .97)
RBM_ARGS=(--hidden 100 --learning-rate .005 --visible-rate .004 --conditional-rate .003 --decay .98)
COND_ARGS=(--conditional-biases --user-bias-rate .01 --day-bias-rate .03 --frequency-bias-rate .012)

knn_columns() {  # side data model [fingerprint-data] -> two kNN columns (round-6 recipes)
  local side=$1 data=$2 model=$3 fp=${4:-} g256 g64 extra=()
  if [ "$side" = B ]; then g256=artifacts/r4/neighbors256.bin; g64=artifacts/neighbors64.bin
  else g256=$R/A-neighbors256.bin; g64=$R/A-neighbors64.bin; fi
  [ -n "$fp" ] && extra=(--fingerprint-data "$fp")
  local tag; tag=$(basename "$data" .bin)
  run "P0-$side-knn256td-$tag" "Phase 0 rehearsal ($side): time-decayed 256-neighbor kNN on K1 residuals" \
    build/netflix knn --data "$data" ${extra[@]+"${extra[@]}"} --graph $g256 --joint-model "$model" --neighbors 256 \
    --power 1 --shrink .5 --time-decay .003 --predictions $R/$side-knn256td.$tag.f32 \
    --support $R/$side-knn256td.$tag.support.f32
  run "P0-$side-knn64-$tag" "Phase 0 rehearsal ($side): 64-neighbor kNN on K1 residuals, power 1" \
    build/netflix knn --data "$data" ${extra[@]+"${extra[@]}"} --graph $g64 --joint-model "$model" --power 1 --shrink .5 \
    --predictions $R/$side-knn64.$tag.f32
}

case ${1:-} in
k1)
  wait_for P0-A-K1
  E=$(best_epoch $R/P0-A-K1.out); echo "K1 P2-selected epoch: $E"
  run P0-A-K1-probe "Phase 0 rehearsal (a): K1 trained on train minus P2, scored on original probe" \
    build/netflix eval-joint --data $DP --fingerprint-data $D --model $R/A-K1.bin --predictions $R/A-K1.probe.f32
  knn_columns A $D $R/A-K1.bin
  knn_columns A $DP $R/A-K1.bin $D
  run P0-B-K1 "Phase 0 rehearsal (b): K1 retrained on all training ratings, fixed at P2-selected epoch $E" \
    build/netflix joint --data $FULL --model $R/B-K1.bin --predictions $R/B-K1.f32 "${K1_ARGS[@]}" \
    --iterations "$E" --fixed-epoch --threads 5
  knn_columns B $FULL $R/B-K1.bin
  ;;
rbm)
  wait_for P0-A-rbm100
  E1=$(best_epoch $R/P0-A-rbm100.out); echo "plain RBM P2-selected epoch: $E1"
  run P0-B-rbm100 "Phase 0 rehearsal (b): plain RBM-100 on all training ratings, fixed at P2-selected epoch $E1" \
    build/netflix rbm --data $FULL --model $R/B-rbm100.bin --predictions $R/B-rbm100.f32 "${RBM_ARGS[@]}" \
    --iterations "$E1" --fixed-epoch --threads 4 > $R/P0-B-rbm100.out 2>&1 &
  run P0-A-rbm100c "Phase 0 rehearsal (a): conditional RBM-100 (R2C recipe) on train minus P2, epoch selected on P2" \
    build/netflix rbm --data $D --init-model $R/A-rbm100.bin --start-epoch $((E1+1)) --model $R/A-rbm100c.bin \
    --predictions $R/A-rbm100c.p2.f32 "${RBM_ARGS[@]}" "${COND_ARGS[@]}" --iterations 130 --patience 6 \
    --threads 5 > $R/P0-A-rbm100c.out 2>&1
  E2=$(best_epoch $R/P0-A-rbm100c.out); echo "conditional RBM P2-selected epoch: $E2"
  run P0-A-rbm100c-probe "Phase 0 rehearsal (a): conditional RBM-100 scored on original probe" \
    build/netflix eval-rbm --data $DP --fingerprint-data $D --model $R/A-rbm100c.bin --predictions $R/A-rbm100c.probe.f32
  wait
  run P0-B-rbm100c "Phase 0 rehearsal (b): conditional RBM-100 on all training ratings, fixed at P2-selected epoch $E2" \
    build/netflix rbm --data $FULL --init-model $R/B-rbm100.bin --start-epoch $((E1+1)) --model $R/B-rbm100c.bin \
    --predictions $R/B-rbm100c.f32 "${RBM_ARGS[@]}" "${COND_ARGS[@]}" --iterations "$E2" --fixed-epoch \
    --threads 5 > $R/P0-B-rbm100c.out 2>&1
  ;;
*) echo "usage: $0 k1|rbm" >&2; exit 2 ;;
esac
echo "chain $1 complete"
