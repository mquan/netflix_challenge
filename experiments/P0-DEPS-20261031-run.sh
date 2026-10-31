#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "P0 dependency queue exited with status $?"' EXIT
wait_resources() {
    while true; do
        if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json; then
            sleep 30
            if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json; then return 0; fi
        else sleep 30; fi
    done
}
echo WAIT_P0_DEPENDENCY_RESOURCE_GATES
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependency-stage-0-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json --stage 0 --prepare
echo START_0_baseline
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/bin/netflix-native-fixed time-baseline --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/baseline/model.bin --iterations 40 --rate-scale 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/baseline/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/bin/netflix-native-fixed eval-time-baseline --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/baseline/model.bin > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/baseline/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json --stage 0 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/baseline/audit.log 2>&1
echo COMPLETE_AUDITED_0_baseline
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependency-stage-1-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json --stage 1 --prepare
echo START_1_graph256
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/bin/netflix-native-fixed graph --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --graph /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph256/neighbors256.bin --neighbors 256 --history-cap 256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph256/train.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json --stage 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph256/audit.log 2>&1
echo COMPLETE_AUDITED_1_graph256
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependency-stage-2-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json --stage 2 --prepare
echo START_2_graph64
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_graph.py --source /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph256/neighbors256.bin --neighbors 64 --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/derived > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/train.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json --stage 2 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/audit.log 2>&1
echo COMPLETE_AUDITED_2_graph64
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependency-stage-3-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json --stage 3 --prepare
echo START_3_neighborhood
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/bin/netflix-native-fixed neighborhood --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --graph /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/derived/neighbors64.bin --time-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/baseline/model.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/neighborhood/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/neighborhood/model.f32 --iterations 4 --fixed-epoch --learning-rate 0.005 --regularization 0.002 --time-decay 0 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/neighborhood/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/bin/netflix-native-fixed eval-neighborhood --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --graph /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/derived/neighbors64.bin --time-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/baseline/model.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/neighborhood/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/neighborhood/reload.f32 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/neighborhood/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_dependency_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-DEPS-20261031-plan.json --stage 3 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/neighborhood/audit.log 2>&1
echo COMPLETE_AUDITED_3_neighborhood
echo COMPLETE_P0_DEPENDENCIES_AND_ONE_COLUMN_OTHER_FAMILIES_AND_FINAL_BANK_STILL_REQUIRED
