#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "P0 user-kNN queue exited with status $?"' EXIT
wait_resources() {
    while true; do
        if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_user_knn_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-USERKNN-20261031-plan.json; then
            sleep 30
            if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_user_knn_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-USERKNN-20261031-plan.json; then return 0; fi
        else sleep 30; fi
    done
}
echo WAIT_P0_USER_KNN_RESOURCE_DEPENDENCIES
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-USERKNN-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn-stage-0-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_user_knn_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-USERKNN-20261031-plan.json --stage 0 --prepare
echo START_USER_KNN_0
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR11-source/netflix-K1-rebuilt joint --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/model.f32 --factors 256 --iterations 41 --learning-rate .008 --regularization .02 --decay .97 --patience 0 --threads 6 --fixed-epoch > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR11-source/netflix-K1-rebuilt eval-joint --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/reload.f32 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_user_knn_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-USERKNN-20261031-plan.json --stage 0 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/audit.log 2>&1
echo COMPLETE_AUDITED_USER_KNN_0
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-USERKNN-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn-stage-1-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_user_knn_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-USERKNN-20261031-plan.json --stage 1 --prepare
echo START_USER_KNN_1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-uknn user-knn --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --joint-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/model.f32 --support /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/model.support.f32 --support-count /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/model.count.f32 --threads 3 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/run.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/r8/bin/netflix-uknn user-knn --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --joint-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/K1/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/replay.f32 --support /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/replay.support.f32 --support-count /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/replay.count.f32 --threads 3 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/replay.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_user_knn_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-USERKNN-20261031-plan.json --stage 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/user-knn/inference/audit.log 2>&1
echo COMPLETE_AUDITED_USER_KNN_1
echo COMPLETE_P0_USER_KNN_COLUMN_AND_TWO_SUPPORTS_ALL_FAMILY_OUTPUTS_AND_FINAL_BANK_STILL_REQUIRED
