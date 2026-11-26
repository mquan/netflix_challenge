#!/bin/zsh
# tsvdx5-<F> NLPP chain, generalized from run_nlpp300.sh. usage: run_nlpp.sh tsvdx5-400
# Waits for the 4800lm package (base + knnf + knns) to finish and release memory.
R=~/workspace/netflix_remote_context_blend/artifacts
S=$R/rcb-nlpp300/source; L=$R/rcb-nlpp300/logs
B=$1; until [ -e $L/DONE ] && ! pgrep -f "release/tsvdx[45]-new" >/dev/null; do sleep 120; done
sleep 60
cd $S
sampler(){ local pid=$1 name=$2; while kill -0 $pid 2>/dev/null; do
  fp=$(top -l 1 -pid $pid -stats mem,cmprs,cpu | tail -1); fr=$(memory_pressure | awk '/free percentage/{print $5}' | tr -d %)
  echo "$(date -u +%FT%TZ) $name pid=$pid mem/cmprs/cpu=$fp mem_free_pct=$fr du=$(du -sm preds_new | cut -f1)MB" >> $L/$name.resources
  if [ "${fr:-100}" -lt 8 ]; then echo "$(date -u +%FT%TZ) GUARD free=$fr% -> TERM $pid" >> $L/$name.resources; kill -TERM $pid; fi
  sleep 30; done; }
run(){ local name=$1
  echo "{\"job\":\"$name\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"binary_sha256\":\"$(shasum -a 256 target/release/tsvdx5-new | cut -c1-64)\",\"threads\":${THR:-8},\"mem_free_pct\":$(memory_pressure | awk '/free percentage/{print $5}' | tr -d %),\"disk_free\":\"$(df -h . | tail -1 | awk '{print $4}')\"}" > $L/$name.start.json
  /usr/bin/time -l env RAYON_NUM_THREADS=${THR:-8} OPENBLAS_NUM_THREADS=${THR:-8} caffeinate -i ./target/release/tsvdx5-new $name > $L/$name.log 2>&1 & local p=$!
  sleep 5; local fp=$(pgrep -f "^./target/release/tsvdx5-new $name\$" | head -1)
  [ -n "$fp" ] && sampler $fp $name &
  wait $p; local rc=$?
  echo "{\"job\":\"$name\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$rc}" > $L/$name.end.json
  return $rc; }
run $B || exit 1
run ${B}__nlpp_paropt || exit 1
run ${B}__nlpp || exit 1
run ${B}__nlpp__knn3 || exit 1
echo "$(date -u +%FT%TZ) done" > $L/DONE-$B
