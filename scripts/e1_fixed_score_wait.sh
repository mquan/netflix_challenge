#!/usr/bin/env bash
# Shell-only waiter: no persistent Python process may hold up SO1's queue.
# Usage: bash scripts/e1_fixed_score_wait.sh ACTIVE_N1_PID
set -euo pipefail
cd "$(dirname "$0")/.."
training_pid=${1:?active N1 training PID required}
[[ "$training_pid" =~ ^[0-9]+$ ]] || exit 2
state=experiments/E1-fixed-scoring-20261030.state
plan=experiments/E1-fixed-scoring-plan-20261030.json
output=artifacts/rehearsal/e1/fixed-scoring-20261030
[[ -f "$plan" && ! -e "$state" && ! -e "$output" ]] || {
  echo 'missing plan or existing scorer output/state; inspect before recovery' >&2; exit 1;
}
trap 'printf "failed at %s\n" "$(date -u +%FT%TZ)" > "$state"' ERR
while :; do
  ready=true
  for arm in w1 w2 w3; do
    for family in K1 knn256td knn64; do
      [[ -f "experiments/E1-B-$family-$arm.json" ]] || ready=false
    done
  done
  if "$ready"; then
    command=$(ps -ww -p "$training_pid" -o args= || true)
    case "$command" in
      *Python\ -B\ */choice_sequence.py\ --data\ */netflix.bin\ --output\ */N1-resume-20261030-refit\ *)
        printf 'scoring at %s while N1 PID %s is active\n' "$(date -u +%FT%TZ)" "$training_pid" > "$state"
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
          MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
          nice -n 10 /Users/quan/.cache/netflix-venv/bin/python -u scripts/e1_fixed_score.py score \
          --plan "$plan" --output "$output"
        printf 'complete at %s; %s/comparison.json\n' "$(date -u +%FT%TZ)" "$output" > "$state"
        exit 0
        ;;
      *)
        printf 'waiting: E1 arms ready; authorized active training PID %s is absent or changed\n' "$training_pid" > "$state"
        ;;
    esac
  else
    printf 'waiting: E1 training records incomplete at %s\n' "$(date -u +%FT%TZ)" > "$state"
  fi
  sleep 30
done
