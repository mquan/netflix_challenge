#!/bin/bash
# usage: run_tx.sh ARM PIPELINE PREDS [--final]
ARM=$1; PIPE=$2; PREDS=$3; FIN=${4:-}; W=~/workspace/netflix_remote_context_blend/artifacts/rcb-l2; cd $W/source
echo "{\"arm\":\"$ARM\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"pipeline\":\"$PIPE\",\"preds\":\"$PREDS\",\"final\":\"$FIN\"}" > $W/$ARM.start.json
/usr/bin/time -l env RAYON_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 SEED=20261124 EPOCHS=24 EPSAVE=1 F=300 LR_I=0.006 LR_U=0.0045 LR_W=0.001 LR_WD=0.0005 LR_Y=0.00075 NBR=100 REG_I=0.015 REG_U=0.06 REG_WD=0.002 RESET=12 \
  caffeinate -i ./target/release/fit tx s3-tx300r -p $PIPE $FIN --preds $PREDS > $W/$ARM.log 2>&1
echo "{\"arm\":\"$ARM\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$?}" > $W/$ARM.end.json
