#!/bin/sh
# Rebuild artifacts/K1-joint256.bin (original K1 checkpoint) from the split
# parts tracked in git, then verify its SHA-256. Run from anywhere in the repo.
set -eu
cd "$(git rev-parse --show-toplevel)/artifacts"
parts=K1-joint256.bin.parts
if [ -f K1-joint256.bin ] && shasum -a 256 -c "$parts/K1-joint256.bin.sha256" >/dev/null 2>&1; then
  echo "K1-joint256.bin already present and verified"; exit 0
fi
cat "$parts"/K1-joint256.bin.part-* > K1-joint256.bin.tmp
mv K1-joint256.bin.tmp K1-joint256.bin
shasum -a 256 -c "$parts/K1-joint256.bin.sha256"
