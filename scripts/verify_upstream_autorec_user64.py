#!/usr/bin/env python3
"""Third encoder pass with independent CSR batches and stable covariance merge."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import hashlib
import numpy as np
import upstream_autorec_user64 as experiment

u=experiment.u

def merge_moments(count,mean,m2,batch):
    n=len(batch);bm=batch.mean(axis=0);delta=bm-mean;centered=batch-bm;total=count+n
    return total,mean+delta*(n/total),m2+centered.T@centered+np.outer(delta,delta)*(count*n/total)

def verify():
    import torch
    receipt=u.read(experiment.REPORT/'receipt.json');plan=u.read(experiment.REPORT/'plan.json')
    if receipt['config']!=experiment.CONFIG or plan['config']!=experiment.CONFIG or not receipt['actual_full_saved_encoder_replay_bitwise']:
        raise ValueError('Fixed complete own encoder extraction required')
    for name,digest in plan['code_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Declared projection/check code differs')
    weights,queries,sources=experiment.source_inputs()
    for name,digest in receipt['source_sha256'].items():
        if sources.get(name)!=digest:raise ValueError('Original own encoder lineage differs')
    for name,digest in receipt['output_sha256'].items():
        if u.sha(experiment.RUN/name)!=digest:raise ValueError('Stored projection output differs')
    saved=torch.load(weights,map_location='cpu',weights_only=False);cfg=experiment.CONFIG
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.20)
    encoder=torch.nn.Sequential(torch.nn.Linear(cfg['items'],cfg['hidden']),torch.nn.Tanh())
    encoder.load_state_dict({k:saved['model'][k] for k in ['0.weight','0.bias']});encoder=encoder.to('mps').eval()
    base=experiment.raw.data.RUN/'trainx';offsets=np.load(base/'offsets.npy');items=np.load(base/'items.npy',mmap_mode='r');ratings=np.load(base/'ratings.npy',mmap_mode='r')
    with np.load(experiment.RUN/'projection.npz') as p:center,basis,scales,values,covariance=[p[k] for k in ['center','basis','scales','eigenvalues','covariance']]
    table=np.load(experiment.RUN/'user64.npy',mmap_mode='r');digest=hashlib.sha256();count=0;mean=np.zeros(cfg['hidden']);m2=np.zeros((cfg['hidden'],cfg['hidden']))
    if table.shape!=(cfg['users'],cfg['rank']) or table.dtype!=np.float32:raise ValueError('Full canonical user table required')
    with torch.no_grad():
        for lo in range(0,cfg['users'],cfg['batch_users']):
            hi=min(lo+cfg['batch_users'],cfg['users']);start,end=int(offsets[lo]),int(offsets[hi])
            x=np.zeros((hi-lo,cfg['items']),dtype=np.float32)
            rows=np.repeat(np.arange(hi-lo),np.diff(offsets[lo:hi+1].astype(np.int64)))
            x[rows,items[start:end]]=ratings[start:end].astype(np.float32)-np.float32(experiment.raw.CONFIG['center'])
            h=np.ascontiguousarray(encoder(torch.from_numpy(x).to('mps')).cpu().numpy());digest.update(h.tobytes())
            a=h.astype(np.float64);count,mean,m2=merge_moments(count,mean,m2,a)
            replay=((a-center)@basis/scales).astype(np.float32)
            if not np.array_equal(replay.view(np.uint32),table[lo:hi].view(np.uint32)):raise ValueError('Every canonical user projection must replay bitwise')
    if count!=cfg['users'] or digest.hexdigest()!=receipt['hidden_c_order_sha256']:raise ValueError('Independent full saved encoder replay differs')
    independent=m2/count
    if not np.allclose(mean,center,rtol=0,atol=1e-12) or not np.allclose(independent,covariance,rtol=1e-9,atol=1e-12):raise ValueError('Independent stable covariance differs')
    spectrum=np.linalg.eigvalsh(independent)[::-1][:cfg['rank']]
    if not np.allclose(spectrum,values,rtol=1e-9,atol=1e-12) or not np.allclose(basis.T@basis,np.eye(cfg['rank']),rtol=0,atol=1e-11):raise ValueError('Top spectrum or orthogonality differs')
    residual=float(np.max(np.abs(independent@basis-basis*values)))
    if residual>1e-9 or not (basis[np.abs(basis).argmax(axis=0),np.arange(cfg['rank'])]>0).all() or not np.array_equal(scales,np.sqrt(values)):
        raise ValueError('Canonical PCA eigenpairs/scales differ')
    if experiment.query_hashes(table,queries)!=receipt['query_features']:raise ValueError('Complete virtual canonical query gathers differ')
    bindings=dict(sources)
    for name,digest in receipt['output_sha256'].items():bindings[str((experiment.RUN/name).relative_to(u.ROOT))]=digest
    for p in [experiment.REPORT/'plan.json',experiment.REPORT/'receipt.json']:bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Evidence changed during independent verification')
    u.write(experiment.REPORT/'verification.json',dict(state='complete_independent_user_encoder_covariance_and_projection',
        receipt_sha256=u.sha(experiment.REPORT/'receipt.json'),verifier_sha256=u.sha(Path(__file__)),bindings=bindings,
        rows=count,components=cfg['rank'],max_eigenpair_residual=residual,actual_third_saved_encoder_pass_bitwise=True,
        all_user_projection_rows_replayed_bitwise=True,independent_csr_batch_construction=True,independent_stable_covariance_checked=True,
        all_virtual_query_gathers_verified=True,qualifying_labels_read=False,
        scope='Third saved encoder pass uses independently vectorized original CSR histories; every hidden vector and projected user row matches. Stable covariance merge, top spectrum, eigenpairs and complete query gathers checked. No nonlinear fit or accuracy claim.'))

if __name__=='__main__':verify()
