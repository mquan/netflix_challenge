#!/bin/zsh
# Four-day autonomous driver (2026-11-26 → 2026-11-30), declared in PLAN.md before any result.
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

log "driver start"
# ---- 1. 4800lm base: verify, release base + snapshots (both phases), then parents ----
waitfor $L48/tsvdx4-4800lm.end.json
if [ "$(exitof $L48/tsvdx4-4800lm.end.json)" != 0 ]; then
  log "4800lm base FAILED exit=$(exitof $L48/tsvdx4-4800lm.end.json); preserving partial outputs"
  pkg rcb-4800lm-partial $L48 $P48/*.out $P48/tsvdx4-4800lm_ep*.npy
  echo "4800lm base failed; releasing queue $(date -u +%FT%TZ)" > $L48/DONE; push $D
else
  mkdir -p $E/lane4800/base
  evalpkg $E/lane4800/base/eval $P48 $E/lane4800/base base=tsvdx4-4800lm snaps=tsvdx4-4800lm_ep03,tsvdx4-4800lm_ep13,tsvdx4-4800lm_ep21,tsvdx4-4800lm_ep24 allep=$(cd $P48; ls tsvdx4-4800lm_ep*.probex.npy | sed 's/.probex.npy//' | grep -vE '_ep(03|13|21|24)$' | paste -sd, -)
  log "4800lm base verified+evaluated: $(grep -E '^\| bank\+' $E/lane4800/base/table.md | tr '\n' ' ')"
  pkg rcb-4800lm-base $P48/tsvdx4-4800lm.probex.npy $P48/tsvdx4-4800lm.qual.npy $P48/tsvdx4-4800lm_ep*.npy $P48/tsvdx4-4800lm.out $L48 $T48/binaries.sha256 $T48/build.log
  push $E/lane4800 $D
  pkg rcb-4800lm-parents $P48/tsvdx4-4800lm.trainx.npy $P48/tsvdx4-4800lm.fulltrain.npy $P48/tsvdx4-4800lm*ifeat*
  push $D
fi
# ---- 2. knnf + knns (run4800.sh), then deferred knn-d with its nbstats ----
waitfor $L48/DONE
if [ -e $P48/tsvdx4-4800lm.probex.npy ]; then
  mkdir -p $E/lane4800/resid
  evalpkg $E/lane4800/resid/eval $P48 $E/lane4800/resid base=tsvdx4-4800lm snaps=tsvdx4-4800lm_ep03,tsvdx4-4800lm_ep13,tsvdx4-4800lm_ep21,tsvdx4-4800lm_ep24 resid=tsvdx4-4800lm__knnf,tsvdx4-4800lm__knns --extra bank+base+resid=base+resid
  log "4800lm knnf/knns: $(grep -E '^\| bank\+' $E/lane4800/resid/table.md | tr '\n' ' ')"
  pkg rcb-4800lm-resid $P48/tsvdx4-4800lm__knn*.npy $P48/tsvdx4-4800lm__knn*.out $L48
  push $E/lane4800 $D
  if ! stopped knn-d; then
    waitfor $L48/rtg_supp.end.json
    cd $T48/source
    for j in tsvdx4-4800lm_diff1:nbstats-new tsvdx4-4800lm__knn-d:tsvdx4-new; do n=${j%%:*}; bin=${j##*:}
      [ -e $L48/$n.end.json ] && continue
      echo "{\"job\":\"$n\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"binary_sha256\":\"$(shasum -a 256 target/release/$bin | cut -c1-64)\"}" > $L48/$n.start.json
      /usr/bin/time -l env RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 5 ./target/release/$bin $n > $L48/$n.log 2>&1; rc=$?
      echo "{\"job\":\"$n\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$rc}" > $L48/$n.end.json; log "$n exit=$rc"; [ $rc = 0 ] || break; done
    if [ -e $P48/tsvdx4-4800lm__knn-d.qual.npy ]; then
      mkdir -p $E/lane4800/full
      evalpkg $E/lane4800/full/eval $P48 $E/lane4800/full base=tsvdx4-4800lm snaps=tsvdx4-4800lm_ep03,tsvdx4-4800lm_ep13,tsvdx4-4800lm_ep21,tsvdx4-4800lm_ep24 resid=tsvdx4-4800lm__knnf,tsvdx4-4800lm__knns knnd=tsvdx4-4800lm__knn-d
      log "4800lm with knn-d: $(grep -E '^\| bank\+' $E/lane4800/full/table.md | tr '\n' ' ')"
      pkg rcb-4800lm-knnd $P48/tsvdx4-4800lm__knn-d.* $L48/tsvdx4-4800lm_diff1.* $L48/tsvdx4-4800lm__knn-d.* $L48/rtg_supp.*
      push $E/lane4800 $D
    fi
  fi
fi
# ---- 3. NLPP300 chain (run_nlpp300.sh in cf-lanes:nlpp300 starts on rcb-4800/logs/DONE) ----
G300="base=tsvdx5-300"
for st in tsvdx5-300 tsvdx5-300__nlpp_paropt tsvdx5-300__nlpp tsvdx5-300__nlpp__knn3; do
  waitfor $LN/$st.end.json; rc=$(exitof $LN/$st.end.json); log "$st exit=$rc"
  case $st in
    tsvdx5-300) files=($PN/tsvdx5-300.probex.npy $PN/tsvdx5-300.qual.npy $PN/tsvdx5-300_ep*.npy $PN/tsvdx5-300.out $PN/tsvdx5-300.trainx.npy $PN/tsvdx5-300.fulltrain.npy);;
    tsvdx5-300__nlpp_paropt) files=($PN/tsvdx5-300__nlpp_regs.npy $PN/tsvdx5-300__nlpp_paropt.out);;
    tsvdx5-300__nlpp) files=($PN/tsvdx5-300__nlpp.probex.npy $PN/tsvdx5-300__nlpp.qual.npy $PN/tsvdx5-300__nlpp.trainx.npy $PN/tsvdx5-300__nlpp.fulltrain.npy $PN/tsvdx5-300__nlpp.out); G300="$G300 nlpp=tsvdx5-300__nlpp";;
    tsvdx5-300__nlpp__knn3) files=($PN/tsvdx5-300__nlpp__knn3.probex.npy $PN/tsvdx5-300__nlpp__knn3.qual.npy $PN/tsvdx5-300__nlpp__knn3.out); G300="$G300 knn3=tsvdx5-300__nlpp__knn3";;
  esac
  pkg rcb-$st $files $LN/$st.* $TN/binaries.sha256 $TN/data.sha256
  if [ $rc = 0 ] && [ $st != tsvdx5-300__nlpp_paropt ]; then
    mkdir -p $E/lane-nlpp300/$st; evalpkg $E/lane-nlpp300/$st/eval $PN $E/lane-nlpp300/$st ${=G300}
    log "$st eval: $(grep -E '^\| bank\+' $E/lane-nlpp300/$st/table.md | tr '\n' ' ')"; fi
  push $E/lane-nlpp300 $D
  [ $rc = 0 ] || { log "NLPP300 chain stopped at $st"; break; }
done
# ---- 4. tsvdx5-400 NLPP chain (declared next; memory floor 64.96 GB, runs alone) ----
if [ -e $LN/DONE ] && ! stopped tsvdx5-400; then
  log "launching tsvdx5-400 chain"; tmux new-window -d -t cf-lanes -n nlpp400 "zsh $E/lane-nlpp300/run_nlpp.sh tsvdx5-400; sleep 86400"
  G400="base=tsvdx5-400"
  for st in tsvdx5-400 tsvdx5-400__nlpp_paropt tsvdx5-400__nlpp tsvdx5-400__nlpp__knn3; do
    waitfor $LN/$st.end.json; rc=$(exitof $LN/$st.end.json); log "$st exit=$rc"
    case $st in
      tsvdx5-400) files=($PN/tsvdx5-400.probex.npy $PN/tsvdx5-400.qual.npy $PN/tsvdx5-400.out $PN/tsvdx5-400.trainx.npy $PN/tsvdx5-400.fulltrain.npy);;
      tsvdx5-400__nlpp_paropt) files=($PN/tsvdx5-400__nlpp_regs.npy $PN/tsvdx5-400__nlpp_paropt.out);;
      tsvdx5-400__nlpp) files=($PN/tsvdx5-400__nlpp.probex.npy $PN/tsvdx5-400__nlpp.qual.npy $PN/tsvdx5-400__nlpp.trainx.npy $PN/tsvdx5-400__nlpp.fulltrain.npy $PN/tsvdx5-400__nlpp.out); G400="$G400 nlpp=tsvdx5-400__nlpp";;
      tsvdx5-400__nlpp__knn3) files=($PN/tsvdx5-400__nlpp__knn3.probex.npy $PN/tsvdx5-400__nlpp__knn3.qual.npy $PN/tsvdx5-400__nlpp__knn3.out); G400="$G400 knn3=tsvdx5-400__nlpp__knn3";;
    esac
    pkg rcb-$st $files $LN/$st.* $TN/binaries.sha256 $TN/data.sha256
    if [ $rc = 0 ] && [ $st != tsvdx5-400__nlpp_paropt ]; then
      mkdir -p $E/lane-nlpp400/$st; evalpkg $E/lane-nlpp400/$st/eval $PN $E/lane-nlpp400/$st ${=G400}
      log "$st eval: $(grep -E '^\| bank\+' $E/lane-nlpp400/$st/table.md | tr '\n' ' ')"; fi
    push $E/lane-nlpp400 $D
    [ $rc = 0 ] || { log "NLPP400 chain stopped at $st"; break; }
  done
fi
log "driver queue complete"; push $D
