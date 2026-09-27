#!/usr/bin/env bash
set -euo pipefail

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/train"

for movie in 1 2 3; do
  padded="$(printf '%07d' "$movie")"
  file="$tmp/train/mv_${padded}.txt"
  printf '%s:\n' "$movie" > "$file"
done

cat >> "$tmp/train/mv_0000001.txt" <<'DATA'
10,5,2005-01-01
20,3,2005-01-02
DATA
cat >> "$tmp/train/mv_0000002.txt" <<'DATA'
10,4,2005-01-03
20,2,2005-01-04
DATA
cat >> "$tmp/train/mv_0000003.txt" <<'DATA'
10,1,2005-01-05
20,1,2005-01-06
DATA

cat > "$tmp/probe.txt" <<'DATA'
1:
10
3:
20
DATA
cat > "$tmp/qualifying.txt" <<'DATA'
2:
10,2005-01-07
3:
20,2005-01-08
DATA

# The production corpus has 17,770 movie files. Fill the synthetic fixture with
# empty movie files so it exercises the exact same parser and invariants.
for movie in $(seq 4 17770); do
  padded="$(printf '%07d' "$movie")"
  printf '%s:\n' "$movie" > "$tmp/train/mv_${padded}.txt"
done

build/netflix ingest --training-dir "$tmp/train" --probe "$tmp/probe.txt" \
  --qualifying "$tmp/qualifying.txt" --output "$tmp/netflix.bin"
build/netflix baseline --data "$tmp/netflix.bin" --model "$tmp/baseline.bin" \
  --iterations 2
build/netflix eval-baseline --data "$tmp/netflix.bin" \
  --model "$tmp/baseline.bin"
build/netflix time-baseline --data "$tmp/netflix.bin" \
  --model "$tmp/time-baseline.bin" --iterations 2
build/netflix eval-time-baseline --data "$tmp/netflix.bin" \
  --model "$tmp/time-baseline.bin"
build/netflix svdpp --data "$tmp/netflix.bin" \
  --time-model "$tmp/time-baseline.bin" --model "$tmp/svdpp.bin" \
  --factors 4 --iterations 2
build/netflix eval-svdpp --data "$tmp/netflix.bin" \
  --time-model "$tmp/time-baseline.bin" --model "$tmp/svdpp.bin"
build/netflix time-svdpp --data "$tmp/netflix.bin" \
  --time-model "$tmp/time-baseline.bin" --svd-model "$tmp/svdpp.bin" \
  --model "$tmp/time-svdpp.bin" --iterations 2
build/netflix eval-time-svdpp --data "$tmp/netflix.bin" \
  --time-model "$tmp/time-baseline.bin" --svd-model "$tmp/svdpp.bin" \
  --model "$tmp/time-svdpp.bin"
build/netflix check-implicit
build/netflix joint --data "$tmp/netflix.bin" --model "$tmp/joint.bin" \
  --factors 4 --iterations 2 --predictions "$tmp/train-pred.f32"
build/netflix eval-joint --data "$tmp/netflix.bin" --model "$tmp/joint.bin" \
  --predictions "$tmp/reload-pred.f32"
cmp "$tmp/train-pred.f32" "$tmp/reload-pred.f32"
