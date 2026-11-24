#!/bin/bash
# Waits for the knn3d wave, then: bank v2 -> blends (context v2 grid) -> 18 tree units -> combine.
# Commits + pushes receipts after each stage. No relaunch of completed units.
set -u; W=~/workspace/netflix_remote_context_blend; cd $W; E=experiments/remote-context-blend
PY=~/workspace/netflix_upstream_repro/.venv/bin/python; CB=$W/artifacts/venv-cb/bin/python
say(){ echo "| $(date -u +%FT%TZ) | $1 | $2 |" >> $E/PROGRESS.md; git add -A $E scripts >/dev/null 2>&1; git -c user.email=mquannie@gmail.com commit -qm "remote-context-blend: $1 — $2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git pull -q --rebase && git push -q; }
until [ -e artifacts/rcb-r1/wave/DONE ]; do sleep 60; done
bad=$(grep -L '"exit":0' artifacts/rcb-r1/wave/*.end.json | wc -l | tr -d ' ')
mkdir -p $E/knn3d/wave; cp artifacts/rcb-r1/wave/*.json $E/knn3d/wave/; for f in artifacts/rcb-r1/wave/*.log; do grep -E "=>|RMSE|real|maximum resident" $f > $E/knn3d/wave/$(basename $f); done
(cd artifacts/rcb-r1/source && shasum -a 256 preds_3/*__knn3*.npy preds_new/*__knn3*.npy) > $E/knn3d/wave/outputs.sha256
say "knn3d wave" "22 jobs finished, $bad non-zero exits; logs + output hashes in knn3d/wave/"
[ "$bad" = "0" ] || exit 1
RCB_BANK=v2 $PY scripts/rcb_bank.py > artifacts/bank-v2.log 2>&1 || { say "bank v2" "FAILED (see artifacts/bank-v2.log)"; exit 1; }
say "bank v2" "built: $(tail -1 artifacts/bank-v2.log)"
RCB_BANK_DIR=artifacts/bank-v2 RCB_BANK_JSON=bank-v2.json RCB_CTX_GRID=1e2,3e2,1e3,3e3,1e4,3e4,1e5 $PY scripts/rcb_blend.py --final --out $E/blend-bank2 > artifacts/blend-bank2.log 2>&1 || { say "blend bank2" FAILED; exit 1; }
mkdir -p artifacts/blend-bank2-out; mv $E/blend-bank2/*.f32 $E/blend-bank2/*.npy artifacts/blend-bank2-out/; cp $E/blend-bank2/summary.json artifacts/blend-bank2-out/
(cd artifacts/blend-bank2-out && shasum -a 256 *) > $E/blend-bank2/files.sha256
say "blend bank2" "$(tail -1 artifacts/blend-bank2.log | cut -c1-300)"
U=artifacts/tree-bank2; mkdir -p $U $E/tree-bank2; units=(); for f in 0 1 2 3 4 final; do for s in 0 1 2; do units+=("$f $s"); done; done
for x in "${units[@]}"; do set -- $x; while [ $(jobs -rp | wc -l) -ge 6 ]; do sleep 10; done
  (cd scripts && OMP_NUM_THREADS=1 $CB rcb_tree.py --bank ../artifacts/bank-v2 --out ../$U --fold $1 --slot $2 > ../$U/$1-s$2.log 2>&1) & done; wait
cp $U/*.json $E/tree-bank2/ 2>/dev/null; say "tree units" "$(ls $U/*.json | wc -l | tr -d ' ')/18 units done"
(cd scripts && $PY rcb_tree_combine.py --bank ../artifacts/bank-v2 --units ../$U --blend ../artifacts/blend-bank2-out --out ../$U/combined) > artifacts/tree-combine.log 2>&1 || { say "tree combine" FAILED; exit 1; }
cp $U/combined/summary.json $E/tree-bank2/combined-summary.json; say "tree combine" "$(tail -1 artifacts/tree-combine.log | cut -c1-300)"
echo done > artifacts/after_wave.DONE
