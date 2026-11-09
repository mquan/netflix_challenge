#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "P0 joint queue exited with status $?"' EXIT
wait_resources() {
    while true; do
        if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_joint_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031-plan.json; then
            sleep 30
            if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_joint_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031-plan.json; then return 0; fi
        else sleep 30; fi
    done
}
echo WAIT_P0_JOINT_RESOURCE_DEPENDENCIES
wait_resources
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-stage-9-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_joint_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031-plan.json --stage 9 --prepare
echo START_9_T6R-f256-resume
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-builds/historical-3/fixed/netflix joint --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --init-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/T6-f256-lr012-reg015-d93/model.bin --start-epoch 12 --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/T6R-f256-resume/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/T6R-f256-resume/model.f32 --factors 256 --iterations 24 --learning-rate .012 --regularization .015 --decay .93 --patience 0 --fixed-epoch --threads 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/T6R-f256-resume/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-builds/historical-3/fixed/netflix eval-joint --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/T6R-f256-resume/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/T6R-f256-resume/reload.f32 --threads 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/T6R-f256-resume/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_joint_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031-plan.json --stage 9 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/T6R-f256-resume/audit.log 2>&1
echo COMPLETE_AUDITED_9_T6R-f256-resume
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-stage-10-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_joint_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031-plan.json --stage 10 --prepare
echo START_10_I2-integrated-f64-koren
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-builds/historical-2/fixed/netflix integrated --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --graph /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/derived/neighbors64.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2-integrated-f64-koren/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2-integrated-f64-koren/model.f32 --factors 64 --iterations 17 --learning-rate .012 --regularization .015 --decay .93 --neighbor-rate .001 --neighbor-regularization .015 --patience 0 --fixed-epoch --threads 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2-integrated-f64-koren/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-builds/historical-2/fixed/netflix eval-integrated --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2-integrated-f64-koren/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2-integrated-f64-koren/reload.f32 --threads 1 --graph /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/derived/neighbors64.bin > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2-integrated-f64-koren/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_joint_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031-plan.json --stage 10 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2-integrated-f64-koren/audit.log 2>&1
echo COMPLETE_AUDITED_10_I2-integrated-f64-koren
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-stage-11-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_joint_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031-plan.json --stage 11 --prepare
echo START_11_I2R-integrated-f64-resume
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-builds/historical-3/fixed/netflix integrated --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --graph /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/derived/neighbors64.bin --init-model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2-integrated-f64-koren/model.bin --start-epoch 18 --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2R-integrated-f64-resume/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2R-integrated-f64-resume/model.f32 --factors 64 --iterations 23 --learning-rate .012 --regularization .015 --decay .93 --neighbor-rate .001 --neighbor-regularization .015 --patience 0 --fixed-epoch --threads 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2R-integrated-f64-resume/train.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint-builds/historical-3/fixed/netflix eval-integrated --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2R-integrated-f64-resume/model.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2R-integrated-f64-resume/reload.f32 --threads 1 --graph /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/dependencies/graph64/derived/neighbors64.bin > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2R-integrated-f64-resume/reload.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_joint_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-JOINT-20261031-plan.json --stage 11 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/I2R-integrated-f64-resume/audit.log 2>&1
echo COMPLETE_AUDITED_11_I2R-integrated-f64-resume
echo COMPLETE_P0_JOINT_THIRTEEN_COLUMNS_OTHER_FAMILIES_AND_FINAL_BANK_STILL_REQUIRED
