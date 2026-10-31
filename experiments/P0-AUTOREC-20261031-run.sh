#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "P0 AutoRec queue exited with status $?"' EXIT
wait_resources() {
    while true; do
        if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031-plan.json; then
            sleep 30
            if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031-plan.json; then return 0; fi
        else sleep 30; fi
    done
}
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-device-input-check.log
wait_resources
echo START_AUTOREC_MPS_PARITY_CHECK
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/verify_production_autorec_device.py --source-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-source --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-device-mps --report /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-device-mps-verification.json --device mps > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-device-mps.log 2>&1
echo COMPLETE_AUTOREC_MPS_PARITY_CHECK
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-stage-0-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031-plan.json --stage 0 --prepare
echo START_AUTOREC_0
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_source.py run --source /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-source/autorec.py --threads 1 -- --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A2-autorec512-norm/model.f32 --fixed-epoch --hidden 512 --layers 2 --epochs 1 --batch 512 --lr 0.0003 --weight-decay 0.0 --input-drop 0.25 --dropout 0.5 --patience 5 --seed 2026 --device mps > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A2-autorec512-norm/train.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_source.py run --source /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-source/autorec.py --threads 1 -- --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A2-autorec512-norm/replay.f32 --fixed-epoch --hidden 512 --layers 2 --epochs 1 --batch 512 --lr 0.0003 --weight-decay 0.0 --input-drop 0.25 --dropout 0.5 --patience 5 --seed 2026 --device mps --reload-checkpoint /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A2-autorec512-norm/model.pt > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A2-autorec512-norm/replay.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031-plan.json --stage 0 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A2-autorec512-norm/audit.log 2>&1
echo COMPLETE_AUDITED_AUTOREC_0
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-stage-1-input-check.log
wait_resources
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031-plan.json --stage 1 --prepare
echo START_AUTOREC_1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_source.py run --source /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-source/autorec.py --threads 1 -- --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/model.f32 --fixed-epoch --hidden 512 --layers 2 --epochs 44 --batch 512 --lr 0.0002 --weight-decay 1e-05 --input-drop 0.5 --dropout 0.5 --patience 5 --seed 2026 --device mps --save-epoch 14 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/train.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_source.py run --source /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-source/autorec.py --threads 1 -- --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/replay.e14.f32 --fixed-epoch --hidden 512 --layers 2 --epochs 44 --batch 512 --lr 0.0002 --weight-decay 1e-05 --input-drop 0.5 --dropout 0.5 --patience 5 --seed 2026 --device mps --save-epoch 14 --reload-checkpoint /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/model.e14.pt > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/replay.e14.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_source.py run --source /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec-source/autorec.py --threads 1 -- --data /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/train-probe-qualifying.bin --predictions /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/replay.f32 --fixed-epoch --hidden 512 --layers 2 --epochs 44 --batch 512 --lr 0.0002 --weight-decay 1e-05 --input-drop 0.5 --dropout 0.5 --patience 5 --seed 2026 --device mps --save-epoch 14 --reload-checkpoint /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/model.pt > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/replay.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_autorec_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-AUTOREC-20261031-plan.json --stage 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/autorec/A3-autorec512-reg/audit.log 2>&1
echo COMPLETE_AUDITED_AUTOREC_1
echo COMPLETE_P0_AUTOREC_THREE_COLUMNS_OTHER_FAMILIES_AND_FINAL_BANK_STILL_REQUIRED
