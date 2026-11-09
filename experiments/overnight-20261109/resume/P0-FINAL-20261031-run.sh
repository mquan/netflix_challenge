#!/bin/bash
set -eu
cd /Users/quan/workspace/netflix_challenge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'echo "P0 final queue exited with status $?"' EXIT
wait_resources() {
    while true; do
        if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_final_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-FINAL-20261031-plan.json; then
            sleep 30
            if /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_final_resources.py --plan /Users/quan/workspace/netflix_challenge/experiments/P0-FINAL-20261031-plan.json; then return 0; fi
        else sleep 30; fi
    done
}
echo WAIT_P0_FINAL_ALL_PRODUCTION_STAGES
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-FINAL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-resolve-bank-input-check.log
wait_resources
echo START_FINAL_resolve-bank
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/resolve_production_bank.py --mapping /Users/quan/workspace/netflix_challenge/experiments/P0-all-family-planned-coverage-20261031.json --bank /Users/quan/workspace/netflix_challenge/experiments/production-blender-plan.json --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/resolved-bank.json > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-resolve-bank.log 2>&1
echo COMPLETE_FINAL_resolve-bank
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-FINAL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-fit-input-check.log
wait_resources
echo START_FINAL_fit
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_final_handoff.py fit --bank /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/resolved-bank.json --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-blender > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-fit.log 2>&1
echo COMPLETE_FINAL_fit
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-FINAL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-infer-input-check.log
wait_resources
echo START_FINAL_infer
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_final_handoff.py infer --bank /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/resolved-bank.json --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-blender --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-inference > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-infer.log 2>&1
echo COMPLETE_FINAL_infer
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-FINAL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-infer-replay-input-check.log
wait_resources
echo START_FINAL_infer-replay
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_final_handoff.py infer --bank /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/resolved-bank.json --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-blender --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-inference-replay > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-infer-replay.log 2>&1
echo COMPLETE_FINAL_infer-replay
wait_resources
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-FINAL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-export-input-check.log
wait_resources
echo START_FINAL_export
nice -n 10 /Users/quan/.cache/netflix-venv/bin/python /Users/quan/workspace/netflix_challenge/scripts/production_final_handoff.py export --bank /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/resolved-bank.json --model /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-blender --manifest /Users/quan/workspace/netflix_challenge/data/production/stage-c-v1/manifest.json --inference /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-inference --replay /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-inference-replay --output /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-export > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-export.log 2>&1
echo COMPLETE_FINAL_export
shasum -a 256 --check /Users/quan/workspace/netflix_challenge/experiments/P0-FINAL-20261031.sha256 > /Users/quan/workspace/netflix_challenge/artifacts/production/P0-20261031/final-post-export-input-check.log
echo COMPLETE_FIXED_PRODUCTION_BANK_BLENDER_AND_RAW_ORDER_EXPORT_NO_HIDDEN_SCORE_OR_TARGET_CLAIM
