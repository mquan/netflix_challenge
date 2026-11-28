#!/bin/zsh
# Extension queue 2 (declared 2026-11-28T18:55Z in PLAN.md): E3 rbmx2-500 → __knnf → __knn3, then E4 rbmx2-600. Small-memory RBMs, run beside tsvdx6-300.
# Never starts work outside the declared queue; never touches other sessions' jobs or trees.
# Pause: touch $D/STOP (checked before each new launch). Every step logs to $D/driver.log and pushes.
set -u; setopt NULL_GLOB
W=~/workspace/netflix_remote_context_blend; A=$W/artifacts; E=$W/experiments/remote-context-blend; D=$E/driver4d
PY=~/workspace/netflix_upstream_repro/.venv/bin/python; REL=remote-context-blend-20261124
T48=$A/rcb-4800; L48=$T48/logs; P48=$T48/source/preds_new
TN=$A/rcb-nlpp300; LN=$TN/logs; PN=$TN/source/preds_new
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $D/driver.log; }
stopped(){ [ -e $D/STOP ] && { log "STOP file present: not launching $1"; return 0; }; return 1; }
waitfor(){ until [ -e "$1" ]; do sleep 120; done; }
exitof(){ $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['exit'])" "$1" 2>/dev/null || echo 99; }
push(){ cd $W; git add -f "$@" >/dev/null 2>&1; git -c user.email=mquannie@gmail.com commit -qm "driver4d: $(tail -1 $D/driver.log | cut -c22-200)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1
  for i in 1 2 3; do git fetch -q origin research/remote-context-blend-execution && git rebase -q --autostash FETCH_HEAD >/dev/null 2>&1 || git rebase --abort >/dev/null 2>&1
    git push -q origin HEAD:research/remote-context-blend-execution >/dev/null 2>&1 && return 0; sleep 60; done; log "push failed (kept local)"; }
# pkg NAME FILE...: APFS-clone files into a staging dir, hash members, tar.gz, split >1.9 GB, upload.
pkg(){ local name=$1; shift; local S=$A/driver4d-pkg/$name; rm -rf $S; mkdir -p $S/m
  for f in "$@"; do [ -e "$f" ] && cp -c -R "$f" $S/m/; done
  (cd $S/m && shasum -a 256 $(find . -type f | sort)) > $D/$name.members.sha256
  tar czf $S/$name.tar.gz -C $S/m . ; local sz=$(stat -f %z $S/$name.tar.gz)
  if [ $sz -gt 1900000000 ]; then (cd $S && split -b 1900m -a 2 $name.tar.gz $name.tar.gz.part- && rm $name.tar.gz); fi
  (cd $S && for f in $name.tar.gz*; do echo "$f $(stat -f %z $f) $(shasum -a 256 $f | cut -c1-64)"; done) > $D/$name.assets.txt
  local ok=1; for f in $S/$name.tar.gz*; do gh release upload $REL $f --clobber -R mquan/netflix_challenge >/dev/null 2>&1 || ok=0; done
  [ $ok = 1 ] && { log "uploaded $name: $(tr '\n' ';' < $D/$name.assets.txt)"; rm -rf $S/m; } || log "UPLOAD FAILED $name (staging kept at $S)"; }
evalpkg(){ local out=$1; shift; OPENBLAS_NUM_THREADS=3 nice -n 5 $PY $W/scripts/rcb_pkg_eval.py "$@" > $out.log 2>&1 || log "eval failed: $out"; }
TR=$A/rcb-rbm500bp; LR=$TR/logs; PR=$TR/source/preds_new
sampler(){ local pid=$1 name=$2; while kill -0 $pid 2>/dev/null; do
  fp=$(top -l 1 -pid $pid -stats mem,cmprs,cpu | tail -1); fr=$(memory_pressure | awk '/free percentage/{print $5}' | tr -d %)
  echo "$(date -u +%FT%TZ) $name pid=$pid mem/cmprs/cpu=$fp mem_free_pct=$fr du=$(du -sm $PR | cut -f1)MB" >> $LR/$name.resources
  if [ "${fr:-100}" -lt 8 ]; then echo "$(date -u +%FT%TZ) GUARD free=$fr% -> TERM $pid" >> $LR/$name.resources; kill -TERM $pid; fi
  sleep 30; done; }
runj(){ local bin=$1 name=$2 thr=$3; [ -e $LR/$name.end.json ] && return $(exitof $LR/$name.end.json); cd $TR/source
  echo "{\"job\":\"$name\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"binary_sha256\":\"$(shasum -a 256 target/release/$bin | cut -c1-64)\",\"threads\":$thr,\"mem_free_pct\":$(memory_pressure | awk '/free percentage/{print $5}' | tr -d %)}" > $LR/$name.start.json
  /usr/bin/time -l env RAYON_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr caffeinate -i ./target/release/$bin $name > $LR/$name.log 2>&1 & local p=$!
  sleep 5; local fp=$(pgrep -f "^./target/release/$bin $name\$" | head -1); [ -n "$fp" ] && sampler $fp $name &
  wait $p; local rc=$?; echo "{\"job\":\"$name\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$rc}" > $LR/$name.end.json; log "$name exit=$rc"; return $rc; }
log "ext2 start"
# ---- E3. rbmx2-500 → __knnf → __knn3 (screen bundle #10 −0.001317; knnf/knn3 verified upstream) ----
if ! stopped rbmx2-500; then
  runj rbmx2-new rbmx2-500 6 && { runj rbmx2-new rbmx2-500__knnf 4; runj rbmx2-new rbmx2-500__knn3 4; }
  pkg rcb-rbmx2-500 $PR/rbmx2-500.probex.npy $PR/rbmx2-500.qual.npy $PR/rbmx2-500.out $PR/rbmx2-500__knnf.* $PR/rbmx2-500__knn3.* $LR/rbmx2-500.* $LR/rbmx2-500__* $TR/binaries.sha256
  pkg rcb-rbmx2-500-parents $PR/rbmx2-500.trainx.npy $PR/rbmx2-500.fulltrain.npy $PR/rbmx2-500*ifeat*
  mkdir -p $E/lane-ext/rbmx2-500; evalpkg $E/lane-ext/rbmx2-500/eval $PR $E/lane-ext/rbmx2-500 base=rbmx2-500 knnf=rbmx2-500__knnf knn3=rbmx2-500__knn3
  log "rbmx2-500 eval: $(grep -E '^\| bank\+' $E/lane-ext/rbmx2-500/table.md | tr '\n' ' ')"; push $E/lane-ext $D
fi
# ---- E4. rbmx2-600 (screen #24 rbmx2-600_ep27 −0.001100; ep27 is terminal) ----
if ! stopped rbmx2-600; then
  runj rbmx2-new rbmx2-600 6
  pkg rcb-rbmx2-600 $PR/rbmx2-600.probex.npy $PR/rbmx2-600.qual.npy $PR/rbmx2-600_ep*.npy $PR/rbmx2-600.out $LR/rbmx2-600.* $TR/binaries.sha256
  mkdir -p $E/lane-ext/rbmx2-600; evalpkg $E/lane-ext/rbmx2-600/eval $PR $E/lane-ext/rbmx2-600 base=rbmx2-600
  log "rbmx2-600 eval: $(grep -E '^\| bank\+' $E/lane-ext/rbmx2-600/table.md | tr '\n' ' ')"; push $E/lane-ext $D
fi
log "ext2 queue complete"; push $D
