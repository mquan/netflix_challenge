#!/bin/sh
set -eu
worktree=$1
file="$worktree/CONVERGENT_SOLUTION.md"
[ -s "$file" ] || { echo '{"verified":false,"result":"missing plan","artifact":"none"}'; exit 0; }
for phrase in 'day-gated household personas' '0.85725' '0.8646' 'all five' 'No score is claimed' 'Stop rules'; do
  grep -Fqi "$phrase" "$file" || { echo '{"verified":false,"result":"missing required plan statement","artifact":"none"}'; exit 0; }
done
if git -C "$worktree" status --porcelain=v1 | grep -E '^.. (src/|scripts/|tests/)'; then
  echo '{"verified":false,"result":"model code changed during plan-only run","artifact":"none"}'; exit 0
fi
printf '{"verified":true,"result":"plan document present with required claims and gates; no model code changed","artifact":"%s"}\n' "$file"
