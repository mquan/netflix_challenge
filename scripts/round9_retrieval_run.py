#!/usr/bin/env python3
"""Select retrieval encoder on P2, refit at fixed epochs, then score both blenders."""
import json
import pathlib
import subprocess
import time
from round9_latent_run import run
from round9_score_bank import score


def wait_for_gpu():
    # This machine is shared with other experiment worktrees. Never start a
    # second large sequence trainer while one is already using MPS.
    while True:
        processes = subprocess.check_output(['ps','-Ao','pid,command'],text=True).splitlines()
        active = [line for line in processes if 'masked_sequence.py' in line and '--device mps' in line]
        if not active: return
        print(json.dumps(dict(status='waiting_for_mps',active=active)),flush=True)
        time.sleep(30)


def main():
    common=['scripts/masked_sequence.py','--data','data/processed/netflix.bin',
            '--patience','3','--threads','1','--seed','20260928','--device','mps',
            '--suffix-prob','0','--seq-len','768','--retrieval-frac','.5']
    selection='artifacts/r9/A3-retrieval-select'
    wait_for_gpu()
    run('R9-A3-select','A2e matched architecture; context is 50% relevant, 25% recent, 25% random; P2-only retrieval geometry',
        [*common,'--selection-data','artifacts/validation/train-matched-20260927.bin',
         '--selection-only','--output',selection,'--epochs','14','--retrieval-table','artifacts/r9/retrieval-p2.npy'])
    selected=json.loads(pathlib.Path(selection+'.json').read_text())['selected_epoch']
    run('R9-A3-full-table','Full-train retrieval geometry for refit only',
        ['scripts/round9_retrieval_table.py','--data','data/processed/netflix.bin',
         '--model','artifacts/K1-joint256.bin','--output','artifacts/r9/retrieval-full.npy'])
    refit='artifacts/r9/A3-retrieval-refit'
    wait_for_gpu()
    run('R9-A3-refit','Refit retrieval encoder at P2-selected epoch; no probe epoch selection',
        [*common,'--output',refit,'--epochs',str(selected),'--retrieval-table','artifacts/r9/retrieval-full.npy'])
    score('R9-A3',refit+'.f32')


if __name__=='__main__':main()
