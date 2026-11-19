#!/usr/bin/env python3
"""Measure a small real fit and saved-model replay; no accuracy evidence."""
import argparse
import datetime as dt
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch

def run(backend):
    if backend=='mps':import upstream_whole110_mps_context as experiment
    else:import upstream_whole110_wide_context as experiment
    experiment.configure();n=experiment.n
    rng=np.random.default_rng(42);x=rng.normal(size=(32768,337)).astype(np.float32);y=rng.normal(0,.1,len(x)).astype(np.float32)
    warm,_=n.train_nn(x,y,experiment.CONFIG,2026,1);del warm
    if backend=='mps':torch.mps.synchronize();torch.mps.empty_cache()
    start=time.perf_counter();model,_=n.train_nn(x,y,experiment.CONFIG,2026,6)
    if backend=='mps':torch.mps.synchronize()
    elapsed=time.perf_counter()-start
    inference=np.tile(x,(2,1));expected=n.nn_predict(model,inference);again=n.nn_predict(model,inference)
    if not np.array_equal(expected.view(np.uint32),again.view(np.uint32)):raise ValueError('Repeated inference is not bitwise deterministic')
    with tempfile.TemporaryDirectory(prefix='netflix-mps-context-') as folder:
        path=Path(folder)/'model.pt';torch.save(dict(state=model.state_dict(),config=experiment.CONFIG,feature_width=337,seed=2026,epochs=6),path)
        restored=experiment.base.verifier.neural_model(path)
        actual=n.nn_predict(restored,inference)
        if not np.array_equal(expected.view(np.uint32),actual.view(np.uint32)):raise ValueError('Actual saved-model reload/inference differs')
    result=dict(state='complete_synthetic_execution_benchmark',observed=dt.datetime.now(dt.timezone.utc).isoformat(),backend=backend,
        rows=len(x),width=337,inference_rows=len(inference),epochs=6,hidden=128,context_rank=32,batch=experiment.CONFIG['nn_batch'],
        fit_seconds=elapsed,repeated_inference_bitwise=True,saved_model_replay_bitwise=True,actual_accuracy_evidence=False,
        source_script_sha256=experiment.u.sha(Path(experiment.__file__)))
    if backend=='mps':result.update(mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    report=experiment.u.ROOT/'experiments/upstream-mps-context-benchmark';experiment.u.write(report/(backend+'.json'),result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['cpu','mps']);a=p.parse_args();run(a.backend)
