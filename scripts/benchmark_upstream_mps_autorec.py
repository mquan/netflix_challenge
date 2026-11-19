#!/usr/bin/env python3
"""Synthetic full-item-width denoising-autoencoder throughput, not accuracy."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'experiments/upstream-mps-autorec-benchmark'

def worker(device,batch):
    import torch
    torch.set_num_threads(2);torch.manual_seed(2026)
    if device=='mps':
        if not torch.backends.mps.is_available():raise ValueError('Available MPS required')
        torch.mps.manual_seed(2026);torch.mps.set_per_process_memory_fraction(.15)
    rng=np.random.default_rng(2026)
    observed=rng.random((batch,17770))<200/17770
    target=np.where(observed,rng.integers(1,6,size=observed.shape)-3.6,0).astype(np.float32)
    x=torch.from_numpy(target).to(device);mask=torch.from_numpy(observed).to(device)
    def model():return torch.nn.Sequential(torch.nn.Linear(17770,512),torch.nn.Tanh(),torch.nn.Linear(512,17770)).to(device)
    fitted=model();opt=torch.optim.AdamW(fitted.parameters(),lr=.0005,weight_decay=.0001)
    def sync():
        if device=='mps':torch.mps.synchronize()
    def step():
        keep=mask & (torch.rand_like(x)>.5);held=mask & ~keep
        pred=fitted(x*keep*2)
        loss=((pred-x).square()*held).sum()/held.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        return loss
    for _ in range(4):step()
    sync();start=time.perf_counter()
    for _ in range(12):loss=step()
    sync();seconds=time.perf_counter()-start
    if not torch.isfinite(loss):raise ValueError('Nonfinite benchmark loss')
    fitted.eval()
    with torch.no_grad():before=fitted(x).cpu().numpy()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'model.pt';torch.save(fitted.state_dict(),path)
        restored=model();restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));restored.eval()
        with torch.no_grad():after=restored(x).cpu().numpy()
    if not np.array_equal(before.view(np.uint32),after.view(np.uint32)):raise ValueError('Saved-model inference replay differs')
    result=dict(state='complete_synthetic_throughput_and_saved_model_replay',device=device,batch_users=batch,
        items=17770,hidden=512,parameters=sum(p.numel() for p in fitted.parameters()),
        warmup_steps=4,measured_steps=12,seconds=seconds,seconds_per_step=seconds/12,
        synthetic_observed_density=200/17770,last_loss=float(loss.detach().cpu()),
        saved_model_replay_bitwise=True,actual_full_data_accuracy_evidence=False,
        public_prediction_values_read=False,qualifying_labels_read=False,
        scope='Synthetic dense full-item-width masked denoising forward/backward/AdamW benchmark. No dataset preparation, original native recipe parity, full-data timing, trained predictor or qualifying score claimed.')
    REPORT.mkdir(exist_ok=True);(REPORT/f'{device}-{batch}.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

def run():
    REPORT.mkdir(exist_ok=True)
    with (ROOT/'experiments/upstream-source-extension/run.lock').open('a') as lock:
        (REPORT/'status.json').write_text(json.dumps(dict(state='waiting_for_existing_meta_fit_lock'))+'\n')
        fcntl.flock(lock,fcntl.LOCK_EX)
        for device in ['cpu','mps']:
            for batch in [64,256]:
                with (REPORT/f'{device}-{batch}.log').open('x') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__)),'--device',device,'--batch',str(batch)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        results=[json.loads((REPORT/f'{d}-{b}.json').read_text()) for b in [64,256] for d in ['cpu','mps']]
        ratios={str(b):next(r['seconds_per_step'] for r in results if r['device']=='cpu' and r['batch_users']==b)/
            next(r['seconds_per_step'] for r in results if r['device']=='mps' and r['batch_users']==b) for b in [64,256]}
        (REPORT/'comparison.json').write_text(json.dumps(dict(state='complete_synthetic_device_benchmark',cpu_over_mps_speed_ratio=ratios,
            all_saved_model_replays_bitwise=all(r['saved_model_replay_bitwise'] for r in results),
            full_data_speed_or_accuracy_claimed=False,results=results),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps']);p.add_argument('--batch',type=int,choices=[64,256]);a=p.parse_args()
    worker(a.device,a.batch) if a.device else run()
