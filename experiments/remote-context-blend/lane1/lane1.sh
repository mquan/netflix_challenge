#!/bin/bash
# Lane 1: CF48 residual chains. Stats + knn3/knn3d first (parallel), then knn-25c/25m.
set -u; R=~/workspace/netflix_remote_context_blend/artifacts/rcb-r1; L=$R/lane1; mkdir -p $L; cd $R/source
run(){ local name=$1 sub=$2 thr=$3; shift 3; local LOG=$L/$name.log
  [ -e $L/$name.end.json ] && { echo "skip $name"; return 0; }
  echo "{\"job\":\"$name\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"env\":\"$*\"}" > $L/$name.start.json
  /usr/bin/time -l env RAYON_NUM_THREADS=$thr "$@" ./target/release/fit $sub $name -p pipeline-new.toml --final --preds preds_new > $LOG 2>&1
  local rc=$?; echo "{\"job\":\"$name\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$rc}" > $L/$name.end.json; return $rc; }
run rtg_supp nbstats 3 X=1 & run cfnade-48_prod nbstats 3 X=1 & run cfnade-48_diff2 nbstats 3 X=1 &
run cfnade-48__knn3 chain 3 X=1 & run cfnade-48__knn3d chain 3 KD_FLOOR=0.5 KD_TAU=30 & wait
run cfnade-48__knn-25c chain 4 KN_K=25 KN_SCALE=2.5 KN_SIM=cosine KN_STAT0=rtg_supp KN_STAT2=cfnade-48_prod KN_TAU=0.05 &
run cfnade-48__knn-25m chain 4 KN_K=25 KN_SCALE=7.5 KN_SIM=mse KN_STAT0=rtg_supp KN_STAT2=cfnade-48_diff2 KN_TAU=0.05 & wait
echo "lane1 chains done $(date -u +%FT%TZ)" > $L/DONE
