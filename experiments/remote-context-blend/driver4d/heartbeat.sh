#!/bin/zsh
# Every 2 h: append a status line (running jobs, latest epochs, memory, disk) and push. Stops when the driver finishes.
setopt NULL_GLOB
W=~/workspace/netflix_remote_context_blend; D=$W/experiments/remote-context-blend/driver4d; A=$W/artifacts
while ! grep -q 'ext queue complete' $D/driver.log 2>/dev/null; do
  jobs=$(ps -Ao etime,command | grep -E '^ *[0-9:-]+ \./target/release/' | sed 's|./target/release/||' | awk '{print $2" "$3" ("$1")"}' | paste -sd';' -)
  ep=""; for f in $A/rcb-4800/logs/*.log $A/rcb-nlpp300/logs/*.log $A/rcb-rbm500bp/logs/*.log; do [ -e $f ] || continue
    l=$(grep -aE 'Epoch|=>' $f | tail -1); [ -n "$l" ] && ep="$ep $(basename $f .log): $l;"; done
  echo "- $(date -u +%FT%TZ) | jobs: ${jobs:-none} | ${ep} | mem free $(memory_pressure | awk '/free percentage/{print $5}') | disk free $(df -h ~ | tail -1 | awk '{print $4}')" >> $D/heartbeat.md
  cd $W; git add -f $D/heartbeat.md >/dev/null 2>&1
  git -c user.email=mquannie@gmail.com commit -qm "driver4d heartbeat

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1
  git fetch -q origin research/remote-context-blend-execution && git rebase -q --autostash FETCH_HEAD >/dev/null 2>&1 || git rebase --abort >/dev/null 2>&1
  git push -q origin HEAD:research/remote-context-blend-execution >/dev/null 2>&1
  sleep 7200; done
