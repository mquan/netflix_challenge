#!/bin/bash
# s3-rbm400pf rerun with LR=0.015 in r11: fit -> export -> upload -> push.
set -u -o pipefail
W=~/workspace/netflix_remote_weekend; R=$W/artifacts/upstream-remote-weekend-r11
E=$W/experiments/upstream-remote-weekend/execution; PY=~/workspace/netflix_upstream_repro/.venv/bin/python
JOB=s3-rbm400pf-lr015; LOG=$R/$JOB.log; D=$R/driver.log
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $D; }
[ -e $LOG ] && { say "log exists, refusing relaunch"; exit 1; }
cd $R/source
say "fit start"
env RAYON_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 EPOCHS=15 EPSAVE=1 H=400 HT=p LR=0.015 NFACT=50 \
  ./target/release/fit rbm s3-rbm400pf -p pipeline-3.toml --final --preds preds_3 > $LOG 2>&1
rc=$?; say "fit exit $rc"
cd $W
say "export"; $PY $E/export_rbm400pf_lr015.py 2>&1 | tee -a $D || { say "export failed"; exit 1; }
mkdir -p $E/$JOB
cp artifacts/exports/$JOB/verification.json artifacts/archives/$JOB/index.json $E/$JOB/
cp $LOG $E/$JOB/ 2>/dev/null
say "upload"; $PY $E/upload_release.py $JOB artifacts/archives/$JOB $E/$JOB/transfer.json 2>&1 | tee -a $D
git add $E/$JOB && git -c user.email=mquannie@gmail.com commit -qm "weekend: $JOB (fit exit $rc) exported and uploaded to draft release

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git pull -q --rebase && git push -q
say "driver done"
