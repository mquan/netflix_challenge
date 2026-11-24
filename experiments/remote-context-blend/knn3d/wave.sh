#!/bin/bash
# knn3d wave: for each base, knn3 control + knn3d (a=0.5, tau=30); 3 concurrent, 4 threads each.
set -u; R=~/workspace/netflix_remote_context_blend/artifacts/rcb-r1; cd $R/source
run(){ local name=$1 pipe=$2 preds=$3 extra=$4; local LOG=$R/wave/$name.log
  [ -e $LOG ] && { echo "skip $name (log exists)"; return; }
  echo "{\"job\":\"$name\",\"start_utc\":\"$(date -u +%FT%TZ)\"}" > $R/wave/$name.start.json
  /usr/bin/time -l env RAYON_NUM_THREADS=4 $extra ./target/release/fit chain $name -p $pipe --final --preds $preds > $LOG 2>&1
  echo "{\"job\":\"$name\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$?}" > $R/wave/$name.end.json; }
mkdir -p $R/wave; jobs=()
for b in s3-tx50n s3-tx60o s3-rbm800p s3-dnn24 s3-attn32 s3-rbm200e s3-bknbrx-100w s3-mfrbmx-60h s3-rbm400pf; do
  jobs+=("${b}__knn3d pipeline-3.toml preds_3 KD_FLOOR=0.5,KD_TAU=30" "${b}__knn3 pipeline-3.toml preds_3 -"); done
for b in tsvdx4-60 tsvdx5-150o; do
  jobs+=("${b}__knn3d pipeline-new.toml preds_new KD_FLOOR=0.5,KD_TAU=30" "${b}__knn3 pipeline-new.toml preds_new -"); done
for j in "${jobs[@]}"; do set -- $j; ex=${4//,/ }; [ "$ex" = "-" ] && ex=""
  while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 5; done
  run $1 $2 $3 "$ex" & done; wait; echo "wave done $(date -u +%FT%TZ)" > $R/wave/DONE
