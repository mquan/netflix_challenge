#!/bin/bash
# tsvdx4-4800lm package on split new (trainx->probex, fulltrain->qual), pinned upstream 49c9b367.
# Waits for TX300 arm C to exit, then: base -> knnf, knns (parallel). knn-d deferred (handoff).
# Sampler logs footprint/swap/free% every 30 s. Guard: if system memory free < 8%, kill ONLY the running 4800lm-package process.
set -u; W=~/workspace/netflix_remote_context_blend/artifacts/rcb-4800; L=$W/logs; mkdir -p $L; cd $W/source
until [ -e ~/workspace/netflix_remote_context_blend/artifacts/rcb-l2/C.end.json ]; do sleep 60; done
sampler(){ local pid=$1 name=$2; while kill -0 $pid 2>/dev/null; do
  fp=$(top -l 1 -pid $pid -stats mem,cmprs | tail -1 | tr -s ' '); sw=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M); fr=$(memory_pressure | awk '/free percentage/{print $5}' | tr -d %)
  echo "$(date -u +%FT%TZ) $name pid=$pid mem/cmprs=$fp swap_free_MB=$sw mem_free_pct=$fr du=$(du -sm preds_new | cut -f1)MB" >> $L/$name.resources
  if [ "${fr:-100}" -lt 8 ]; then echo "$(date -u +%FT%TZ) GUARD system memory free ${fr}%: killing $name ($pid)" | tee -a $L/$name.resources; pkill -TERM -P $pid; kill -TERM $pid; fi
  sleep 30; done; }
run(){ local name=$1; [ -e $L/$name.end.json ] && return 0
  echo "{\"job\":\"$name\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"binary_sha256\":\"$(shasum -a 256 target/release/tsvdx4-new | cut -c1-64)\"}" > $L/$name.start.json
  /usr/bin/time -l env RAYON_NUM_THREADS=${THR:-8} OPENBLAS_NUM_THREADS=${THR:-8} caffeinate -i ./target/release/tsvdx4-new $name > $L/$name.log 2>&1 & local p=$!
  sleep 5; sampler $(pgrep -f "release/tsvdx4-new $name$" | head -1) $name & wait $p; local rc=$?
  echo "{\"job\":\"$name\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$rc}" > $L/$name.end.json; return $rc; }
THR=8 run tsvdx4-4800lm || exit 1
THR=4 run tsvdx4-4800lm__knnf & THR=4 run tsvdx4-4800lm__knns & wait
echo "4800lm package (base+knnf+knns) done $(date -u +%FT%TZ)" > $L/DONE
