"""Complete terminal predictive state and durable outputs for the own GPU RBM.

All learned predictive tensors are retained. This format intentionally is not a
resumable optimizer snapshot; it cannot restart/extend fitting. No CLI or GPU
permission is provided. An external owner must supply original resource guards.
"""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import shutil
import numpy as np
from upstream_factored_rbm_training import Config,SHARED,LOCAL
from upstream_factored_rbm_prediction import predict

FORMAT='own_factored_bipolar_complete_terminal_predictor_v1'


def freeze(state,phase,admission_sha256,expected_config):
    if phase not in ('known','qualifying'):raise ValueError('Original phase required')
    if (asdict(state.config)!=expected_config or state.active_epoch is not None or state.completed_epoch!=state.config.epochs
            or state.input_digest!=admission_sha256):
        raise ValueError('Only fully completed input-bound terminal fit may be saved')
    expected=(len(state.reader.nonempty_users)+state.config.batch_users-1)//state.config.batch_users
    if state.completed_batches!=expected:raise ValueError('Every original nonempty-user batch required')
    return dict(format=FORMAT,phase=phase,config=asdict(state.config),admission_sha256=admission_sha256,
        dimensions=[state.reader.n_users,state.reader.n_items,len(state.reader.day_values)],
        completed_epoch=state.completed_epoch,completed_batches=state.completed_batches,
        model={k:v.cpu().clone() for k,v in dict(state.shared,**state.bias).items()},
        terminal_rng_state=state.generator.get_state().clone(),frequency_bias='zero_LR_BIF0',
        optimizer_state_retained=False,training_resume_supported=False)


def validate(record,reader,phase,admission_sha256,expected_config):
    import torch
    if (record['config']!=expected_config or record['format']!=FORMAT or record['phase']!=phase or phase not in ('known','qualifying')
            or record['admission_sha256']!=admission_sha256 or record['frequency_bias']!='zero_LR_BIF0'
            or record['optimizer_state_retained'] is not False or record['training_resume_supported'] is not False
            or record['dimensions']!=[reader.n_users,reader.n_items,len(reader.day_values)]):
        raise ValueError('Complete original-phase input-bound terminal predictor required')
    config=Config(**record['config']);config.validate()
    if record['completed_epoch']!=config.epochs or record['completed_batches']!=(len(reader.nonempty_users)+config.batch_users-1)//config.batch_users:
        raise ValueError('Complete original epoch and batch counters required')
    model=record['model'];ni,nu,nd=reader.n_items,reader.n_users,len(reader.day_values)
    shapes=dict(p=(ni,5,config.factors),q=(config.factors,config.hidden),d=(ni,config.hidden),bv=(ni,5),bh=(config.hidden,),bu=(nu,5),but=(nd,5))
    if set(model)!=set(SHARED+LOCAL):raise ValueError('Every actual predictive tensor required')
    for name,shape in shapes.items():
        v=model[name]
        if (v.device.type!='cpu' or v.dtype!=torch.float32 or tuple(v.shape)!=shape or not bool(torch.isfinite(v).all())):
            raise ValueError('Exact finite CPU f32 predictive model shape required')
    return config


def replay(record,reader,phase,admission_sha256,expected_config,guard,device='cpu'):
    import torch
    if not callable(guard):raise ValueError('External original resource guard required')
    guard();config=validate(record,reader,phase,admission_sha256,expected_config)
    shared={k:record['model'][k].to(device) for k in SHARED}
    output=np.empty(len(reader.qr['user_idxs']),dtype=np.float32);seen=np.zeros(len(output),dtype=bool)
    users=np.flatnonzero(reader.qr['user_cnts'])
    for lo in range(0,len(users),config.batch_users):
        guard();b=reader.batch(users[lo:lo+config.batch_users]);model=dict(shared)
        for k,indices in [('bu',b['global_users']),('but',b['global_days'])]:
            index=torch.tensor(indices,dtype=torch.int64,device='cpu')
            model[k]=record['model'][k].index_select(0,index).to(device)
        support={k:torch.tensor(v,dtype=torch.int64,device=device) for k,v in b['support'].items()}
        query={k:torch.tensor(v,dtype=torch.int64,device=device) for k,v in b['query'].items()}
        values=predict(model,support,query,config.tile_rows)['prediction'].cpu().numpy()
        rows=b['query_rows']
        if len(values)!=len(rows) or np.any(seen[rows]) or not np.isfinite(values).all():
            raise ValueError('Every finite original query row required')
        output[rows]=values;seen[rows]=True;guard()
    if not seen.all():raise ValueError('Complete original query coverage required')
    guard();return output


class GuardedStream:
    def __init__(self,stream,root,guard,floor_bytes,reserve_bytes):
        self.stream,self.root,self.guard=stream,root,guard
        self.floor,self.reserve=floor_bytes,reserve_bytes

    def write(self,data):
        part=memoryview(data).cast('B');total=len(part)
        for lo in range(0,total,1048576):
            block=part[lo:lo+1048576]
            # The production guard must self-suspend/resume this same process
            # until this extra reservation is safe, preserving the write stack.
            self.guard(extra_disk_bytes=self.reserve+len(block))
            if shutil.disk_usage(self.root).free<self.floor+self.reserve+len(block):
                # A production guard must normally preserve this same process
                # before this condition. Never silently shrink or retry output.
                raise RuntimeError('Original disk floor/output reserve unavailable')
            count=self.stream.write(block)
            if count!=len(block):raise IOError('Complete guarded payload write required')
        return total

    def tell(self):return self.stream.tell()
    def flush(self):return self.stream.flush()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_directory(path,root,guard):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('Output directory must remain inside owned root')
    path.mkdir(parents=True,exist_ok=True)
    current=path
    while True:
        guard();sync_directory(current)
        if current==root:break
        current=current.parent
    guard();sync_directory(root.parent)


def durable_payload(path,writer,root,guard,floor_bytes=int(18.75*2**30),reserve_bytes=64*2**20):
    """Exclusive, complete, fsynced publication; failed staging files retained."""
    path=Path(path);temporary=path.with_name(path.name+'.staging')
    if path.exists() or temporary.exists():raise ValueError('Original output/staging preserved; no overwrite or automatic retry')
    durable_directory(path.parent,root,guard);guard()
    with temporary.open('xb') as stream:
        wrapped=GuardedStream(stream,root,guard,floor_bytes,reserve_bytes)
        writer(wrapped);wrapped.flush();os.fsync(stream.fileno())
    guard();os.link(temporary,path)  # Atomic no-clobber publication of complete bytes.
    sync_directory(path.parent)
    temporary.unlink()
    sync_directory(path.parent)
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            guard();block=stream.read(1048576)
            if not block:break
            h.update(block)
    guard();return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
