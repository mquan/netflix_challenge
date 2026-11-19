#!/usr/bin/env python3
"""Float32 Apple-GPU execution for a separately declared own neural recipe."""
from pathlib import Path
import numpy as np
import upstream_nonlinear as n

MODEL_FACTORY=None

def available():
    import torch
    return torch.backends.mps.is_available()

def predict(model,features):
    import torch
    if next(model.parameters()).device.type!='mps':raise ValueError('Explicit MPS model required; no silent backend substitution')
    model.eval();parts=[]
    with torch.no_grad():
        for block in np.array_split(features,max(1,(len(features)+65535)//65536)):
            x=torch.from_numpy(np.ascontiguousarray(block,dtype=np.float32)).to('mps')
            parts.append(model(x).squeeze(1).cpu().numpy())
    return np.concatenate(parts)

def train_nn(features,residual,cfg,seed,epochs,valid=None):
    import torch
    if not available() or MODEL_FACTORY is None or cfg.get('neural_backend')!='torch-mps-float32':
        raise ValueError('Declared available float32 MPS backend/factory required')
    torch.set_num_threads(cfg['threads']);torch.manual_seed(seed);torch.mps.manual_seed(seed)
    torch.mps.set_per_process_memory_fraction(.25)
    model=MODEL_FACTORY(features.shape[1],cfg).to('mps')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['nn_lr'],weight_decay=cfg['nn_weight_decay'])
    tx=torch.from_numpy(np.ascontiguousarray(features,dtype=np.float32)).to('mps')
    ty=torch.from_numpy(np.asarray(residual,dtype=np.float32)).to('mps')
    best,chosen=(float(np.mean(valid[1]**2)),0) if valid is not None else (None,epochs)
    for epoch in range(1,epochs+1):
        model.train()
        # Keep permutation generation on CPU explicitly. Device dropout has its
        # own seeded stream, so this is a new backend recipe, not CPU bit parity.
        for batch in torch.randperm(len(tx)).split(cfg['nn_batch']):
            ix=batch.to('mps');optimizer.zero_grad(set_to_none=True)
            (model(tx[ix]).squeeze(1)-ty[ix]).square().mean().backward();optimizer.step()
        if valid is not None:
            loss=float(np.mean((predict(model,valid[0])-valid[1])**2))
            if loss<best:best,chosen=loss,epoch
            n.event(stage='nn_epoch',seed=seed,epoch=epoch,inner_mse=loss,chosen=chosen,backend='mps_float32')
            if epoch-chosen>=cfg['nn_patience']:break
    torch.mps.synchronize();return model,chosen

def load_model(checkpoint,cfg):
    import torch
    torch.set_num_threads(cfg['threads'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['config']!=cfg or cfg.get('neural_backend')!='torch-mps-float32':raise ValueError('Frozen MPS recipe differs')
    model=MODEL_FACTORY(saved['feature_width'],cfg);model.load_state_dict(saved['state']);return model.to('mps')
