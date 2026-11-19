#!/usr/bin/env python3
"""Own masked-denoising raw predictor, with durable per-phase epoch checkpoints."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import numpy as np
import upstream_mps_autorec_data as data

u=data.u
RUN=u.ROOT/'artifacts/upstream-mps-autorec'
REPORT=u.ROOT/'experiments/upstream-mps-autorec'
CONFIG=dict(users=480189,items=17770,hidden=512,epochs_per_phase=50,batch_users=256,
    lr=.0005,weight_decay=.0001,input_drop_probability=.5,center=3.6,seed=2026,
    snapshot_epochs=[5,10,20,30,40,50],backend='torch_mps_float32',prediction_clip=[1.,5.],
    loss='MSE on observed training ratings hidden from corrupted input',optimizer='AdamW')
HISTORIES={}

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def code_hashes():
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(data.__file__),
        u.ROOT/'scripts/verify_upstream_mps_autorec_data.py']}

def prepare():
    path=REPORT/'plan.json'
    if path.exists():return verify_plan()
    if RUN.exists():raise ValueError('Existing partial raw-model directory preserved')
    sources={}
    for phase in data.PHASES:
        rp=data.REPORT/(phase+'-receipt.json');vp=data.REPORT/(phase+'-verification.json');r,v=u.read(rp),u.read(vp)
        if (v['state']!='complete_independent_lossless_csr_and_query_verification' or v['receipt_sha256']!=u.sha(rp) or
            not v['every_training_tuple_replayed'] or not v['query_pairs_absent_from_training'] or not v['full_query_projection_checked']):
            raise ValueError('Complete independently checked training histories required')
        for collection in [r['source_sha256'],{str((data.RUN/phase/name).relative_to(u.ROOT)):digest for name,digest in r['outputs'].items()}]:
            for name,digest in collection.items():
                if name in sources and sources[name]!=digest:raise ValueError('Conflicting source lineage')
                sources[name]=digest
        for p in [rp,vp]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen original history changed')
    RUN.mkdir();REPORT.mkdir(parents=True,exist_ok=True)
    plan=dict(state='own_raw_model_recipe_fixed_before_fitting',created=stamp(),config=CONFIG,code_sha256=code_hashes(),source_sha256=sources,
        versions={name:importlib.metadata.version(name) for name in ['numpy','torch']},qualifying_labels_read=False,
        known_probe_labels_used_for_stopping_or_configuration=False,
        rule='Both original history phases run50 full epochs, alternating at durable epoch boundaries. Fixed snapshots5/10/20/30/40/50; no accuracy-based stopping or epoch selection. Actual saved-model query replay required for every snapshot.',
        checkpoint_retention='Last two optimizer/RNG checkpoints per phase; all declared snapshot model weights and query vectors retained. Historical optimizer hashes are log evidence, not a claim that pruned files remain available.',
        scope='Additional own raw masked-denoising predictor. No public numerical prediction input or original native-recipe parity claim. Original CPU95-epoch jobs are unchanged.')
    u.write(path,plan);return plan

def verify_plan():
    p=u.read(REPORT/'plan.json')
    if p['config']!=CONFIG or p['code_sha256']!=code_hashes() or p['versions']!={name:importlib.metadata.version(name) for name in ['numpy','torch']}:
        raise ValueError('Frozen raw-model recipe changed')
    for name,digest in p['source_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen raw-model source changed: '+name)
    return p

def make_model(cfg):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(cfg['items'],cfg['hidden']),torch.nn.Tanh(),torch.nn.Linear(cfg['hidden'],cfg['items']))

def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def dense_history(phase,users):
    base=data.RUN/phase
    key=(str(base),phase)
    if key not in HISTORIES:
        HISTORIES[key]=tuple(np.load(base/(name+'.npy'),mmap_mode='r') for name in ['offsets','items','ratings'])
    offsets,items,ratings=HISTORIES[key]
    result=np.zeros((len(users),CONFIG['items']),dtype=np.float32)
    for row,user in enumerate(users):
        lo,hi=int(offsets[user]),int(offsets[user+1]);result[row,items[lo:hi]]=ratings[lo:hi].astype(np.float32)-np.float32(CONFIG['center'])
    return result

def infer(model,phase):
    import torch
    base=data.RUN/phase;qoff=np.load(base/'query-offsets.npy');qitems=np.load(base/'query-items.npy',mmap_mode='r');order=np.load(base/'query-order.npy',mmap_mode='r')
    users=np.flatnonzero(np.diff(qoff.astype(np.int64)));result=np.empty(len(order),dtype=np.float32);filled=np.zeros(len(order),dtype=bool)
    model.eval()
    with torch.no_grad():
        for lo in range(0,len(users),CONFIG['batch_users']):
            batch=users[lo:lo+CONFIG['batch_users']];x=torch.from_numpy(dense_history(phase,batch)).to('mps')
            predicted=model(x).cpu().numpy()
            for row,user in enumerate(batch):
                a,b=int(qoff[user]),int(qoff[user+1]);dest=order[a:b]
                result[dest]=np.clip(predicted[row,qitems[a:b]]+np.float32(CONFIG['center']),*CONFIG['prediction_clip']);filled[dest]=True
    if not filled.all() or not np.isfinite(result).all():raise ValueError('Query inference coverage or finiteness differs')
    return result

def latest(phase):
    path=RUN/phase/'latest.json'
    if not path.exists():return None
    p=u.read(path);cp=RUN/phase/p['checkpoint']
    if u.sha(cp)!=p['checkpoint_sha256']:raise ValueError('Durable checkpoint pointer differs')
    return p

def worker(phase,epoch):
    import torch
    verify_plan()
    if not torch.backends.mps.is_available():raise ValueError('Declared MPS backend unavailable')
    torch.set_num_threads(2);torch.manual_seed(CONFIG['seed']);torch.mps.manual_seed(CONFIG['seed']);torch.mps.set_per_process_memory_fraction(.20)
    root=RUN/phase;root.mkdir(exist_ok=True);pointer=latest(phase)
    if epoch!=(pointer['epoch']+1 if pointer else 1):raise ValueError('Only the next durable epoch may run')
    model=make_model(CONFIG).to('mps');opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    if pointer:
        state=torch.load(root/pointer['checkpoint'],map_location='cpu',weights_only=False)
        if state['config']!=CONFIG or state['phase']!=phase or state['plan_sha256']!=u.sha(REPORT/'plan.json'):raise ValueError('Checkpoint configuration/lineage differs')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng']);torch.mps.set_rng_state(state['mps_rng'])
    off=np.load(data.RUN/phase/'offsets.npy');eligible=np.flatnonzero(np.diff(off.astype(np.int64)))
    users=np.random.default_rng(CONFIG['seed']+1009*epoch).permutation(eligible)
    model.train();numerator=torch.zeros((),device='mps');denominator=torch.zeros((),device='mps');start=time.monotonic()
    for lo in range(0,len(users),CONFIG['batch_users']):
        batch=users[lo:lo+CONFIG['batch_users']];x=torch.from_numpy(dense_history(phase,batch)).to('mps');observed=x!=0
        keep=observed & (torch.rand_like(x)>CONFIG['input_drop_probability']);held=observed & ~keep
        residual=model(x*keep/(1-CONFIG['input_drop_probability']))-x
        squared=(residual.square()*held).sum();count=held.sum();loss=squared/count.clamp_min(1)
        if not torch.isfinite(loss):raise ValueError('Nonfinite raw-model training loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        numerator+=squared.detach();denominator+=count
    torch.mps.synchronize();seconds=time.monotonic()-start
    snapshot=None
    if epoch in CONFIG['snapshot_epochs']:
        snap=root/'snapshots'/f'epoch-{epoch:05}';snap.mkdir(parents=True,exist_ok=False)
        weights=snap/'model.pt';torch.save(dict(model=cpu_tree(model.state_dict()),config=CONFIG,phase=phase,epoch=epoch,plan_sha256=u.sha(REPORT/'plan.json')),weights)
        values=infer(model,phase);np.save(snap/'prediction.npy',values)
        saved=torch.load(weights,map_location='cpu',weights_only=False);restored=make_model(CONFIG).to('mps');restored.load_state_dict(saved['model'])
        replay=infer(restored,phase)
        if not np.array_equal(values.view(np.uint32),replay.view(np.uint32)):raise ValueError('Actual raw snapshot model replay differs')
        del restored
        snapshot=dict(state='complete_actual_saved_snapshot_query_replay',phase=phase,epoch=epoch,rows=len(values),
            model_sha256=u.sha(weights),prediction_sha256=u.sha(snap/'prediction.npy'),plan_sha256=u.sha(REPORT/'plan.json'),
            actual_saved_model_replay_bitwise=True,qualifying_labels_read=False,full50_epoch_model_complete=False)
        u.write(REPORT/f'{phase}-snapshot-{epoch:05}.json',snapshot)
    checkpoint=root/f'epoch-{epoch:05}.pt';tmp=checkpoint.with_suffix('.tmp')
    torch.save(dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),cpu_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state(),config=CONFIG,phase=phase,epoch=epoch,plan_sha256=u.sha(REPORT/'plan.json')),tmp)
    tmp.replace(checkpoint);u.write(root/'latest.json',dict(epoch=epoch,checkpoint=checkpoint.name,checkpoint_sha256=u.sha(checkpoint)))
    u.write(REPORT/f'{phase}-epoch-{epoch:05}.json',dict(state='complete_durable_training_epoch',phase=phase,epoch=epoch,users=len(users),
        masked_training_mse=float((numerator/denominator.clamp_min(1)).cpu()),training_seconds=seconds,observed=stamp(),
        checkpoint_sha256=u.sha(checkpoint),checkpoint_retention='Only latest two optimizer states retained; all declared snapshot weights retained.',
        plan_sha256=u.sha(REPORT/'plan.json'),qualifying_labels_read=False))
    for old in root.glob('epoch-*.pt'):
        if int(old.stem.split('-')[1])<epoch-1:old.unlink()
    print(dict(phase=phase,epoch=epoch,seconds=seconds,snapshot=bool(snapshot)),flush=True)

def paired(epoch):
    receipts={p:u.read(REPORT/f'{p}-snapshot-{epoch:05}.json') for p in data.PHASES}
    paths={}
    for phase,r in receipts.items():
        path=RUN/phase/'snapshots'/f'epoch-{epoch:05}'/'prediction.npy'
        if not r['actual_saved_model_replay_bitwise'] or u.sha(path)!=r['prediction_sha256']:raise ValueError('Paired snapshot replay binding differs')
        paths[data.PHASES[phase][0]]=str(path.relative_to(u.ROOT))
    u.write(REPORT/f'pair-{epoch:05}.json',dict(state='complete_own_raw_snapshot_pair_with_actual_saved_model_replays',epoch=epoch,pairs=paths,
        source_pin=u.PIN,plan_sha256=u.sha(REPORT/'plan.json'),receipts={p:u.sha(REPORT/f'{p}-snapshot-{epoch:05}.json') for p in data.PHASES},
        full50_epoch_model_complete=(epoch==CONFIG['epochs_per_phase'] and all(latest(p)['epoch']>=epoch for p in data.PHASES)),qualifying_labels_read=False,
        scope='Both complete fixed snapshot query vectors, from own disjoint canonical histories. Larger fixed50-epoch trajectory continues; no qualifying score claimed.'))

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not all((data.REPORT/(p+'-verification.json')).exists() for p in data.PHASES):
            u.write(REPORT/'status.json',dict(state='waiting_for_independently_verified_real_histories',observed=stamp()));time.sleep(20)
        prepare();previous=data.resources.memory.memory()
        for epoch in range(1,CONFIG['epochs_per_phase']+1):
            for phase in data.PHASES:
                pointer=latest(phase)
                if pointer and pointer['epoch']>=epoch:continue
                with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
                    u.write(REPORT/'status.json',dict(state='waiting_for_shared_epoch_slot',phase=phase,epoch=epoch,observed=stamp()))
                    fcntl.flock(shared,fcntl.LOCK_EX)
                    while True:
                        current=data.resources.memory.memory();workers=data.resources.footprints();free=shutil.disk_usage(u.ROOT).free
                        if (sum(v['bytes'] for v in workers.values())<18*2**30 and current['compressor_bytes']<5*2**30 and
                            current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and free>20*2**30):break
                        u.write(REPORT/'status.json',dict(state='waiting_for_raw_epoch_headroom',phase=phase,epoch=epoch,observed=stamp(),workers=workers,memory=current,disk_free_bytes=free))
                        previous=current;time.sleep(30)
                    with (REPORT/f'{phase}-epoch-{epoch:05}.log').open('x') as log:
                        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker','--phase',phase,'--epoch',str(epoch)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT)
                        u.write(REPORT/'status.json',dict(state='fitting_complete_declared_raw_epoch',phase=phase,epoch=epoch,pid=child.pid,observed=stamp()))
                        if child.wait():raise RuntimeError('Raw epoch failed; durable checkpoints and partial outputs preserved')
                    previous=data.resources.memory.memory()
                time.sleep(.5)
            if epoch in CONFIG['snapshot_epochs']:paired(epoch)
        verify_plan()
        u.write(REPORT/'completion.json',dict(state='complete_both_originally_declared50_epoch_own_raw_phases',observed=stamp(),
            plan_sha256=u.sha(REPORT/'plan.json'),epochs_per_phase=CONFIG['epochs_per_phase'],snapshot_epochs=CONFIG['snapshot_epochs'],
            all_declared_snapshot_pairs_complete=all((REPORT/f'pair-{e:05}.json').exists() for e in CONFIG['snapshot_epochs']),
            qualifying_labels_read=False,scope='Own new raw-model recipe complete, with all fixed snapshot actual inference replays. No original native-recipe parity or qualifying score claimed.'))
        u.write(REPORT/'status.json',dict(state='own_raw_training_complete_independent_source_review_required',observed=stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--phase',choices=list(data.PHASES));p.add_argument('--epoch',type=int);a=p.parse_args()
    worker(a.phase,a.epoch) if a.worker else run()
