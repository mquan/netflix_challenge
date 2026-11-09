#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "P0 Bayes queue exited with status $?"' EXIT
wait_resources() {
    while true; do
        if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json; then
            sleep 30
            if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json; then return 0; fi
        else sleep 30; fi
    done
}
echo WAIT_P0_BAYES_RESOURCE_DEPENDENCIES
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes-initial-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python -m unittest discover -s tests -p test_bayes_production.py -v > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes-historical-thread-check.log 2>&1
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes-stage-0-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json --stage 0 --prepare
echo START_BAYES_0
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-bayes-0c1f29d bayes --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B2-full-r64.s0128.stage0/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B2-full-r64.s0128.stage0/model.f32 --factors 64 --samples 128 --burn-in 16 --snapshot-every 16 --checkpoint-every 8 --threads 4 --seed 2026 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B2-full-r64.s0128.stage0/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-bayes-0c1f29d bayes --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --init-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B2-full-r64.s0128.stage0/model.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B2-full-r64.s0128.stage0/reload.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B2-full-r64.s0128.stage0/reload.f32 --samples 128 --burn-in 16 --threads 4 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B2-full-r64.s0128.stage0/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json --stage 0 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B2-full-r64.s0128.stage0/audit.log 2>&1
echo COMPLETE_AUDITED_BAYES_0
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes-stage-1-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json --stage 1 --prepare
echo START_BAYES_1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-bayes-0c1f29d bayes --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B3-full-r128.s0160.stage0/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B3-full-r128.s0160.stage0/model.f32 --factors 128 --samples 160 --burn-in 16 --snapshot-every 16 --checkpoint-every 8 --threads 4 --seed 7 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B3-full-r128.s0160.stage0/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-bayes-0c1f29d bayes --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --init-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B3-full-r128.s0160.stage0/model.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B3-full-r128.s0160.stage0/reload.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B3-full-r128.s0160.stage0/reload.f32 --samples 160 --burn-in 16 --threads 4 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B3-full-r128.s0160.stage0/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json --stage 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B3-full-r128.s0160.stage0/audit.log 2>&1
echo COMPLETE_AUDITED_BAYES_1
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes-stage-2-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json --stage 2 --prepare
echo START_BAYES_2
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-bayes-0c1f29d bayes --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/model.f32 --factors 128 --samples 32 --burn-in 16 --snapshot-every 16 --checkpoint-every 8 --threads 4 --seed 13 --item-bin 2 --drift 2 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-bayes-0c1f29d bayes --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --init-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/model.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/reload.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/reload.f32 --samples 32 --burn-in 16 --threads 4 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json --stage 2 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/audit.log 2>&1
echo COMPLETE_AUDITED_BAYES_2
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes-stage-3-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json --stage 3 --prepare
echo START_BAYES_3
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-bayes-0c1f29d bayes --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --init-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage0/model.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage1/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage1/model.f32 --samples 160 --burn-in 16 --snapshot-every 16 --checkpoint-every 8 --threads 5 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage1/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-bayes-0c1f29d bayes --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --init-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage1/model.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage1/reload.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage1/reload.f32 --samples 160 --burn-in 16 --threads 5 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage1/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_bayes_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-BAYES-20261031-plan.json --stage 3 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/bayes/B4-full-r128.s0160.stage1/audit.log 2>&1
echo COMPLETE_AUDITED_BAYES_3
echo COMPLETE_P0_BAYES_3_COLUMNS_OTHER_FAMILIES_AND_FINAL_EXPORT_STILL_REQUIRED
