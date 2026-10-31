#!/usr/bin/env bash
# Start the already-declared Bayes w1 noise control after K1's fixed decision.
# This waiter never stops another job and never keeps a Python process alive.
set -euo pipefail
cd "$(dirname "$0")/.."
plan=${1:?queue plan required}
prefix=experiments/E1-bayes-control-queue-20261030
state=$prefix.state.json
[[ ! -e "$state" ]] || { echo 'existing queue state; inspect before recovery' >&2; exit 1; }
mkdir "$prefix.lock.d"  # Atomic ownership of this queue instance.
jq_bin=$(command -v jq)
status() {
  "$jq_bin" -n --arg state "$1" --arg note "$2" --arg time "$(date -u +%FT%TZ)" \
    --argjson pid "$$" '{state:$state,note:$note,updated_utc:$time,pid:$pid}' > "$state.tmp"
  mv "$state.tmp" "$state"
}
trap 'status failed "queue command failed; inspect logs before recovery"' ERR
verify() {
  while IFS=$'\t' read -r path expected; do
    actual=$(shasum -a 256 "$path" | cut -d' ' -f1)
    [[ "$actual" == "$expected" ]] || { echo "frozen input changed: $path" >&2; return 1; }
  done < <("$jq_bin" -r '.frozen | to_entries[] | [.key,.value] | @tsv' "$plan")
}
comparison=$("$jq_bin" -r '.comparison' "$plan")
expected_score_plan=$("$jq_bin" -r '.scoring_plan_sha256' "$plan")
verify
while :; do
  if [[ -e experiments/E1-B-B4-w1.log || -e experiments/E1-B-B4-w1.json ]]; then
    status skipped_existing_control 'w1 already has artifacts; no duplicate launched'
    exit 0
  fi
  if [[ -f "$comparison" ]]; then
    "$jq_bin" -e --arg sha "$expected_score_plan" '.plan_sha256 == $sha and
      (.arms.w2.comparison.delta_rmse | type == "number") and
      (.arms.w3.comparison.delta_rmse | type == "number")' "$comparison" > /dev/null
    if ! "$jq_bin" -e '.arms.w2.comparison.delta_rmse < 0 or
        .arms.w3.comparison.delta_rmse < 0' "$comparison" > /dev/null; then
      status skipped_no_k1_gain 'both K1 treatments fail to improve w1; predeclared early-stop condition; existing jobs untouched'
      exit 0
    fi
    # K1 has written all receipts and has exited before adding four CPU threads.
    if ! pgrep -f '^build/netflix-rowweights ' > /dev/null; then
      verify
      status running 'K1 comparison has some gain; launching the fixed four-thread Bayes w1 control'
      bash scripts/e1_bayes_chain.sh build/netflix-bayesweights w1
      "$jq_bin" -e '.returncode == 0' experiments/E1-B-B4-w1.json > /dev/null
      status complete 'fixed Bayes w1 control completed; comparison pending'
      exit 0
    fi
  fi
  status waiting 'waiting for fixed K1 comparison and released K1 CPU threads'
  sleep 30
done
