#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cutoff_epoch=$(/Users/quan/.cache/netflix-venv/bin/python -c 'import json,datetime; d=json.load(open("experiments/overnight-20261109/schedule.json")); print(int(datetime.datetime.fromisoformat(d["cutoff_utc"]).timestamp()))')
echo WAIT_UNTIL_2026_11_09_0700_PACIFIC
while [ "$(date -u +%s)" -lt "$cutoff_epoch" ]; do
    sleep 5
done
exec /Users/quan/.cache/netflix-venv/bin/python experiments/overnight-20261109/control.py --pause
