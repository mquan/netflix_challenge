#!/usr/bin/env python3
"""Matched E2 fine-tuning from a complete frozen peer cache.

No scheduler or default experiment recipe. A hash-bound plan supplies the fixed
budget; its control/treatment runs share masks, order, sampling and dropout.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from e2_code_cache import CodeCache
from e2_corater_data import Coraters
from e2_joint_context import JointContextModel, attach_peers
from e2_query_codes import parameter_hash
from gbdt_blend import load
import masked_history as mh
import masked_sequence as ms
from train_holdout import digest


def rating_batch(d, users, config, rng, index, lookup, seed, cap):
    batch = ms.prepare_batch(d, users, config, rng=rng)
    # Recover target dates from identities alone, without changing the original
    # mask/sequence RNG consumption or reading target rating values.
    rows, source, _ = mh.gather(d['train_offsets'], users)
    stride = d['items']+1
    keys = rows*stride+d['train_items'][source]
    order = np.argsort(keys)
    targets = (batch['target_rows'].cpu().numpy()*stride
               +batch['target_items'].cpu().numpy()+1)
    positions = np.searchsorted(keys[order], targets)
    if np.any(positions >= len(keys)) or np.any(keys[order[positions]] != targets):
        raise ValueError('masked target absent from training context')
    days = d['train_days'][source[order[positions]]]
    return attach_peers(batch, users, days, index, lookup, seed=seed, cap=cap)


def train_epoch(model, d, optimizer, epoch, seed, index, lookup, peer_seed, cap, max_batches=0):
    c = model.config
    host = dataclasses.replace(c, device='cpu')
    counts = np.diff(d['train_offsets']).astype(np.int64)
    active = np.flatnonzero(counts > 0)
    repeats = np.minimum(np.maximum(-(-counts[active]//c.seq_len), 1), max(c.windows, 1))
    rng = np.random.default_rng(seed+epoch)
    groups = ms.length_batches(ms.capped_lengths(d,c), rng.permutation(np.repeat(active,repeats)), c, rng)
    def prepare(group):
        return rating_batch(d,group,host,rng,index,lookup,peer_seed,cap)
    model.train()
    started = time.monotonic()
    sse, targets, batches = 0., 0, 0
    stream = ms.prefetched(prepare,groups,c.device)
    try:
        for batch_no, (_, batch) in enumerate(stream,1):
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_batch(batch)
            loss = mh.loss_value(model,logits,batch['target_ratings'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite E2 rating loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                sse += float(((model.expected(logits)-batch['target_ratings']-1)**2).sum())
            targets += batch['target_ratings'].numel()
            batches += 1
            if batch_no % 100 == 0:
                print(json.dumps(dict(epoch=epoch,batch=batch_no,targets=targets,
                    train_rmse=math.sqrt(sse/targets),gradient_norm=float(norm),
                    seconds=time.monotonic()-started)),flush=True)
            if max_batches and batch_no >= max_batches:
                break
    finally:
        stream.close()
    if not targets:
        raise ValueError('no masked rating targets')
    return dict(epoch=epoch,batches=batches,targets=targets,train_rmse=math.sqrt(sse/targets),
                seconds=time.monotonic()-started)


@torch.no_grad()
def predict(model,d,index,lookup,peer_seed,cap,queries=None):
    queries = d['probe'] if queries is None else queries
    if not len(queries):
        return np.empty(0,dtype=np.float32)
    users = queries['user'].astype(np.int64)
    order = np.argsort(users,kind='stable')
    offsets = np.searchsorted(users[order],np.arange(d['users']+1))
    c = dataclasses.replace(model.config,batch_tokens=2*model.config.batch_tokens,
                            batch_users=2*model.config.batch_users)
    host = dataclasses.replace(c,device='cpu')
    lengths = ms.capped_lengths(d,c,np.diff(offsets))
    result = np.empty(len(queries),dtype=np.float32)
    model.eval()
    def prepare(group):
        rows, ids, _ = mh.gather(offsets,group)
        q = order[ids]
        days = queries['day'][q]
        batch = ms.prepare_batch(d,group,host,queries=(rows,queries['item'][q].astype(np.int64)-1,days))
        return attach_peers(batch,group,days,index,lookup,seed=peer_seed,cap=cap)
    for group,batch in ms.prefetched(prepare,ms.length_batches(lengths,np.unique(users),c),c.device):
        q = order[mh.gather(offsets,group)[1]]
        result[q] = model.expected(model.forward_batch(batch)).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite E2 predictions')
    return result


def run(plan_path,output,arm):
    if arm not in ('control','treatment'):
        raise ValueError('unknown E2 arm')
    plan_path,output = Path(plan_path),Path(output)
    p = json.loads(plan_path.read_text())
    required = [p[key] for key in ('data','warm_checkpoint','warm_report','cache_plan')]
    required += [str(Path(__file__).resolve().with_name(name)) for name in
        ('e2_train.py','e2_code_cache.py','e2_corater_data.py','e2_joint_context.py',
         'e2_query_codes.py','masked_history.py','masked_sequence.py','gbdt_blend.py','train_holdout.py')]
    required += [str(Path(p['index'])/'manifest.json'),str(Path(p['cache'])/'manifest.json')]
    if not set(required) <= set(p['frozen']):
        raise ValueError('training plan lacks required input/source bindings')
    def verify():
        for path,expected in p['frozen'].items():
            if digest(path) != expected:
                raise ValueError('frozen input changed: '+path)
    verify()
    if (p['device'] not in ('cpu','mps') or min(p['epochs'],p['threads'],p['peer_cap']) < 1
            or not math.isfinite(p['lr']) or p['lr'] <= 0 or p['max_batches'] < 0):
        raise ValueError('invalid fixed learning budget')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(p['threads'])
    torch.manual_seed(p['seed'])
    saved = torch.load(p['warm_checkpoint'],map_location='cpu',weights_only=True)
    warm = json.loads(Path(p['warm_report']).read_text())
    if (warm['selected_epoch'] != saved['epoch'] or warm['config'] != saved['config']
            or warm['selection_sha256'] != p['frozen'][p['data']]):
        raise ValueError('warm checkpoint/report/context differ')
    c = ms.Config(**saved['config'])
    if c.head != 'distribution' or c.loss != 'mse' or c.objective != 'masked' or c.retrieval_table:
        raise ValueError('requires ordinary masked A2h MSE distribution checkpoint')
    c = dataclasses.replace(c,device=p['device'],lr=p['lr'])
    model = JointContextModel(saved['items'],c,saved['weeks'],use_peer_values=arm=='treatment')
    model.load_rating_state(saved['state'])
    # Check that the cache's frozen producer is this exact original backbone.
    backbone = ms.SequenceModel(saved['items'],c,saved['weeks'])
    backbone.load_state_dict(saved['state'],strict=True)
    binding = p['cache_binding']
    cache_plan = json.loads(Path(p['cache_plan']).read_text())
    if (binding['plan_sha256'] != digest(p['cache_plan'])
            or binding['model_sha256'] != parameter_hash(backbone)
            or cache_plan['data'] != p['data'] or cache_plan['checkpoint'] != p['warm_checkpoint']
            or cache_plan['frozen'][p['data']] != p['frozen'][p['data']]
            or cache_plan['frozen'][p['warm_checkpoint']] != p['frozen'][p['warm_checkpoint']]):
        raise ValueError('cache producer differs from warm backbone/context')
    del backbone
    lookup = CodeCache(p['cache'],binding=binding)
    index = Coraters(p['data'],p['index'])
    d = load(p['data'])
    if d['items'] != saved['items'] or ms.weeks(d) != saved['weeks']:
        raise ValueError('checkpoint dimensions differ from context')
    model.to(c.device)
    # Reset after all setup: both arms start with identical model/dropout RNG.
    torch.manual_seed(p['seed'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=c.weight_decay)
    report = dict(plan_sha256=digest(plan_path),arm=arm,config=dataclasses.asdict(c),
        warm_epoch=saved['epoch'],cache_binding=binding,records=[],
        smoke_only=bool(p['max_batches']),prediction_epoch=p['epochs'],
        torch_version=str(torch.__version__),numpy_version=np.__version__)
    output.mkdir(parents=True)
    mh.atomic_json(output/'report.json',report)
    for epoch in range(1,p['epochs']+1):
        record = train_epoch(model,d,optimizer,epoch,p['seed'],index,lookup,p['peer_seed'],
                             p['peer_cap'],p['max_batches'])
        report['records'].append(record)
        mh.atomic_json(output/'report.json',report)
    predictions = predict(model,d,index,lookup,p['peer_seed'],p['peer_cap'])
    mh.write_f32(output/'predictions.f32',predictions)
    torch.save(dict(config=dataclasses.asdict(c),items=d['items'],weeks=saved['weeks'],
        epoch=p['epochs'],arm=arm,warm_epoch=saved['epoch'],state=model.state_dict()),output/'model.pt')
    report.update(prediction_sha256=digest(output/'predictions.f32'),
                  checkpoint_sha256=digest(output/'model.pt'))
    # Deliberately no automatic scoring or epoch selection. Shared gate owns scores.
    verify()
    report['state'] = 'complete'
    mh.atomic_json(output/'report.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--arm',required=True,choices=('control','treatment'))
    args = parser.parse_args()
    run(args.plan,args.output,args.arm)
