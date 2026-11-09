#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "P0 kernel queue exited with status $?"' EXIT
wait_resources() {
    while true; do
        if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json; then
            sleep 30
            if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json; then return 0; fi
        else sleep 30; fi
    done
}
echo WAIT_P0_KERNEL_RESOURCE_DEPENDENCIES
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel-stage-0-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json --stage 0 --prepare
echo START_KERNEL_0
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR12-source/moonshot_gp_initial --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/J3-joint128-reg005/model.bin --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp-full/model --neighbors 32 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp-full/run.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR12-source/moonshot_gp_initial --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/J3-joint128-reg005/model.bin --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp-full/replay --neighbors 32 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp-full/replay.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json --stage 0 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp-full/audit.log 2>&1
echo COMPLETE_AUDITED_KERNEL_0
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel-stage-1-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json --stage 1 --prepare
echo START_KERNEL_1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR12-source/moonshot_gp_persistent --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/J3-joint128-reg005/model.bin --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp2-persistent-full/model --neighbors 32 --time-scale 30 --persistent .5 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp2-persistent-full/run.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR12-source/moonshot_gp_persistent --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/J3-joint128-reg005/model.bin --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp2-persistent-full/replay --neighbors 32 --time-scale 30 --persistent .5 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp2-persistent-full/replay.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json --stage 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/gp2-persistent-full/audit.log 2>&1
echo COMPLETE_AUDITED_KERNEL_1
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel-stage-2-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json --stage 2 --prepare
echo START_KERNEL_2
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR12-source/moonshot_local --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/J3-joint128-reg005/model.bin --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-global/model --groups 1 --rank 16 --epochs 6 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-global/run.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR12-source/moonshot_local --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/J3-joint128-reg005/model.bin --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-global/replay --groups 1 --rank 16 --epochs 6 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-global/replay.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json --stage 2 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-global/audit.log 2>&1
echo COMPLETE_AUDITED_KERNEL_2
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel-stage-3-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json --stage 3 --prepare
echo START_KERNEL_3
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR12-source/moonshot_local --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/J3-joint128-reg005/model.bin --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-local/model --groups 8 --rank 8 --epochs 6 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-local/run.log 2>&1
nice -n 10 /Users/quan/workspace/netflix_challenge/artifacts/production-replay/PR12-source/moonshot_local --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/joint/J3-joint128-reg005/model.bin --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-local/replay --groups 8 --rank 8 --epochs 6 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-local/replay.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_kernel_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-KERNEL-20261031-plan.json --stage 3 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/kernel/local-local/audit.log 2>&1
echo COMPLETE_AUDITED_KERNEL_3
echo COMPLETE_P0_KERNEL_6_COLUMNS_USER_KNN_AND_FINAL_BANK_STILL_REQUIRED
