#!/bin/bash
# Every 30 min: snapshot CF48 progress and push to research/remote9h-handoff. Stops at 15:03Z.
R=$HOME/workspace/netflix_remote9h; P=$R/artifacts/upstream-remote9h/cfnade48-package; O=$R/experiments/upstream-remote9h/run
cd $R
while [ $(date -u +%s) -lt $(date -u -j -f %Y-%m-%dT%H:%M:%SZ 2026-11-19T15:03:28Z +%s) ]; do
  sleep 1800
  cp $P/native-source/preds_preset48/cfnade-48.phase-*.out $O/ 2>/dev/null
  {
    echo "# CF48 heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "start 2026-11-19T06:03:28Z, training deadline 2026-11-19T14:33:28Z"
    for t in trainx fulltrain; do echo "- $t latest checkpoint: $(cat $P/checkpoints/$t/latest.json | tr -d '\n ')"; done
    echo "- probex snapshots: $(ls $P/native-source/preds_preset48 | grep -c 'ep[0-9]*.probex.npy') ; qual snapshots: $(ls $P/native-source/preds_preset48 | grep -c 'ep[0-9]*.qual.npy')"
    echo '```'; ps -eo pid,pgid,rss,%cpu,etime,command | grep -E 'cfnade48-stage|remote12h_supervisor' | grep -v grep | cut -c1-160
    for ph in x qual; do tail -1 $O/cf48-$ph/resources.jsonl; done
    for f in $O/cfnade-48.phase-*.out; do echo "== $f"; tail -2 $f; done
    echo '```'
  } > $O/HEARTBEAT.md
  git add experiments/upstream-remote9h/run
  git -c user.email=mquannie@gmail.com -c user.name="Quan Nguyen" commit -q -m "remote9h CF48 heartbeat $(date -u +%H:%MZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD:research/remote9h-handoff
done
