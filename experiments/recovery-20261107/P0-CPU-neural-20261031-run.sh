#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "P0 CPU neural queue exited with status $?"' EXIT
wait_resources() {
    while true; do
        if pgrep -f '[S]0-lane0-20261031-run.sh' >/dev/null && pgrep -f '[S]0-lane1-20261031-run.sh' >/dev/null; then
            sleep 30
            continue
        fi
        if [ "$(sysctl -n kern.memorystatus_vm_pressure_level)" != "1" ]; then
            sleep 30
            continue
        fi
        sleep 30
        if ! { pgrep -f '[S]0-lane0-20261031-run.sh' >/dev/null && pgrep -f '[S]0-lane1-20261031-run.sh' >/dev/null; } && [ "$(sysctl -n kern.memorystatus_vm_pressure_level)" = "1" ]; then
            return 0
        fi
    done
}
echo WAIT_CPU_NEURAL_FREE_SLOT_AND_NORMAL_MEMORY
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-0-input-check.log
wait_resources
echo START_0_MH3-categorical-mse-refit.f32
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural.py --column MH3-categorical-mse-refit.f32 --dataset-manifest /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/manifest.json --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/neural/MH3-categorical-mse-refit --registry /Users/quan/workspace/netflix_challenge/experiments/production-neural-recipes.json --archive /Users/quan/workspace/netflix_challenge/experiments/production-neural-sources.zip --train > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-0.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031-plan.json --stage 0 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-0-audit.log 2>&1
echo COMPLETE_AUDITED_0_MH3-categorical-mse-refit.f32
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-1-input-check.log
wait_resources
echo START_1_MT0-refit.refit.e12.f32
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural.py --column MT0-refit.refit.e12.f32 --dataset-manifest /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/manifest.json --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/neural/MT0-refit.refit.e12 --registry /Users/quan/workspace/netflix_challenge/experiments/production-neural-recipes.json --archive /Users/quan/workspace/netflix_challenge/experiments/production-neural-sources.zip --train > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-1.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031-plan.json --stage 1 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-1-audit.log 2>&1
echo COMPLETE_AUDITED_1_MT0-refit.refit.e12.f32
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-2-input-check.log
wait_resources
echo START_2_MT1-refit.refit.e10.f32
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural.py --column MT1-refit.refit.e10.f32 --dataset-manifest /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/manifest.json --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/neural/MT1-refit.refit.e10 --registry /Users/quan/workspace/netflix_challenge/experiments/production-neural-recipes.json --archive /Users/quan/workspace/netflix_challenge/experiments/production-neural-sources.zip --train > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-2.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031-plan.json --stage 2 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-2-audit.log 2>&1
echo COMPLETE_AUDITED_2_MT1-refit.refit.e10.f32
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-3-input-check.log
wait_resources
echo START_3_MT2-refit.refit.e12.f32
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural.py --column MT2-refit.refit.e12.f32 --dataset-manifest /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/manifest.json --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/neural/MT2-refit.refit.e12 --registry /Users/quan/workspace/netflix_challenge/experiments/production-neural-recipes.json --archive /Users/quan/workspace/netflix_challenge/experiments/production-neural-sources.zip --train > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-3.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031-plan.json --stage 3 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-3-audit.log 2>&1
echo COMPLETE_AUDITED_3_MT2-refit.refit.e12.f32
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-4-input-check.log
wait_resources
echo START_4_MT3-refit.refit.e7.f32
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural.py --column MT3-refit.refit.e7.f32 --dataset-manifest /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/manifest.json --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/neural/MT3-refit.refit.e7 --registry /Users/quan/workspace/netflix_challenge/experiments/production-neural-recipes.json --archive /Users/quan/workspace/netflix_challenge/experiments/production-neural-sources.zip --train > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-4.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031-plan.json --stage 4 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-4-audit.log 2>&1
echo COMPLETE_AUDITED_4_MT3-refit.refit.e7.f32
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-5-input-check.log
wait_resources
echo START_5_MT5-refit.f32
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural.py --column MT5-refit.f32 --dataset-manifest /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/manifest.json --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/neural/MT5-refit --registry /Users/quan/workspace/netflix_challenge/experiments/production-neural-recipes.json --archive /Users/quan/workspace/netflix_challenge/experiments/production-neural-sources.zip --train > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-5.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031-plan.json --stage 5 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-5-audit.log 2>&1
echo COMPLETE_AUDITED_5_MT5-refit.f32
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-6-input-check.log
wait_resources
echo START_6_MT6-refit.f32
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural.py --column MT6-refit.f32 --dataset-manifest /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/manifest.json --output-dir /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/neural/MT6-refit --registry /Users/quan/workspace/netflix_challenge/experiments/production-neural-recipes.json --archive /Users/quan/workspace/netflix_challenge/experiments/production-neural-sources.zip --train > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-6.log 2>&1
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_neural_audit.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-CPU-neural-20261031-plan.json --stage 6 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/cpu-neural-stage-6-audit.log 2>&1
echo COMPLETE_AUDITED_6_MT6-refit.f32
echo COMPLETE_P0_CPU_NEURAL_SEVEN_COLUMNS_NATIVE_AND_FINAL_BANK_STILL_REQUIRED
