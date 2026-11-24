#!/bin/bash
# usage: run.sh <stage> [env...]; refuses relaunch if log exists
set -u; R=~/workspace/netflix_remote_context_blend/artifacts/rcb-r1; JOB=s3-tx300r__$1; shift
LOG=$R/$JOB.log; [ -e $LOG ] && { echo "log exists, refusing"; exit 1; }
cd $R/source
echo "{\"job\":\"$JOB\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"pid\":$$}" > $R/$JOB.start.json
/usr/bin/time -l env RAYON_NUM_THREADS=6 "$@" ./target/release/fit chain $JOB -p pipeline-3.toml --final --preds preds_3 > $LOG 2>&1
echo "{\"job\":\"$JOB\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$?}" > $R/$JOB.end.json
