#!/usr/bin/env python3
"""NN2 recipe on a streaming_blend bank without the full dense ridge design."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from gbdt_blend import load, blend_fold
from round8_nn_blend import train, predict, net
from streaming_blend import digest, weights, predict_ridge, rmse


def standardized(features, fit, block=16, output=None, return_scaler=False, zero_columns=None, replacement_columns=None):
    out = (np.lib.format.open_memmap(output, mode='w+', dtype='<f4',
                                    shape=features.shape, fortran_order=True)
           if output else np.empty(features.shape, dtype=np.float32))
    scaler={k:np.empty(features.shape[1],dtype=np.float64) for k in ('lo','hi','mean','std')}
    zeros=set(zero_columns or [])
    for start in range(0, features.shape[1], block):
        stop=min(start+block,features.shape[1])
        if all(column in zeros for column in range(start,stop)):
            out[:,start:stop]=0
            for name in ('lo','hi','mean'):scaler[name][start:stop]=0
            scaler['std'][start:stop]=1e-6
            continue
        source = features[:, start:start+block].astype(np.float64)
        for column in (zero_columns or []):
            if start<=column<start+source.shape[1]:source[:,column-start]=0
        for column,values in (replacement_columns or {}).items():
            if start<=column<start+source.shape[1]:source[:,column-start]=values
        lo, hi = np.percentile(source[fit], [1, 99], axis=0)
        source = np.clip(source, lo, hi)
        mean, std = source[fit].mean(0), np.maximum(source[fit].std(0), 1e-6)
        out[:, start:start+block] = (source-mean)/std
        for name,value in zip(('lo','hi','mean','std'),(lo,hi,mean,std)):
            scaler[name][start:start+block]=value
    if output: out.flush()
    return (out,scaler) if return_scaler else out


def predict_device(model, x, device):
    if device == 'cpu': return predict(model, x)
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(b.to(device)).squeeze(1).cpu().numpy()
                               for b in torch.from_numpy(x).split(65536)])


def train_device(x, r, a, seed, epochs, valid=None, sample_weights=None):
    if a.device == 'cpu' and sample_weights is None: return train(x,r,a,seed,epochs,valid=valid)
    torch.manual_seed(seed)
    model=net(x.shape[1],a.hidden,a.layers,a.dropout).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    resident=getattr(a,'resident_features',False)
    tx,tr=torch.from_numpy(x),torch.from_numpy(r.astype(np.float32))
    if resident:
        # Copy once instead of gathering random rows from a column-major
        # memory map for every minibatch. Keep the CPU shuffle unchanged.
        tx=torch.from_numpy(np.ascontiguousarray(x)).to(a.device)
        tr=tr.to(a.device)
    tw=None
    if sample_weights is not None:
        sw=np.asarray(sample_weights,dtype=np.float64)
        if sw.shape!=(len(x),) or np.any(sw<=0) or not np.isfinite(sw).all(): raise ValueError('invalid training weights')
        tw=torch.from_numpy((sw/sw.mean()).astype(np.float32))
        if resident:tw=tw.to(a.device)
    best,best_epoch,state=float('inf'),0,None
    for epoch in range(1,epochs+1):
        model.train()
        for batch in torch.randperm(len(tx)).split(a.batch):
            opt.zero_grad(set_to_none=True)
            index=batch.to(a.device) if resident else batch
            pred=model(tx[index].to(a.device)).squeeze(1)
            loss=(pred-tr[index].to(a.device)).square()
            if tw is not None:loss=loss*tw[index].to(a.device)
            loss.mean().backward()
            opt.step()
        if valid is None: continue
        error=(predict_device(model,valid[0],a.device)-valid[1])**2
        score=float(np.average(error,weights=valid[2])) if len(valid)>2 else float(np.mean(error))
        if score<best: best,best_epoch,state=score,epoch,copy.deepcopy(model.state_dict())
        if epoch-best_epoch>=a.patience: break
    if valid is not None: model.load_state_dict(state)
    return model,best_epoch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank', required=True, help='streaming tree prefix, not its .json')
    ap.add_argument('--output', required=True)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=.2)
    ap.add_argument('--lr', type=float, default=.0003)
    ap.add_argument('--weight-decay', type=float, default=.01)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seeds', nargs='+', type=int, default=[2026, 7, 13])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--disk-features', action='store_true', help='bound wide-bank RAM with file-backed features')
    ap.add_argument('--device', choices=['cpu','mps'], default='cpu')
    ap.add_argument('--resident-features',action='store_true',help='keep training features on MPS for fast row gathering')
    ap.add_argument('--reuse-features',help='reuse verified raw NN encoding, never a fold-normalized matrix')
    ap.add_argument('--drop-unrated',action='store_true',help='suppress the shifted unlabeled-interaction-count meta feature')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.device=='mps' and not torch.backends.mps.is_available(): ap.error('MPS unavailable')
    prefix = Path(a.output); prefix.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.output+'.json').exists(): ap.error('output already exists')
    record = json.loads(Path(a.bank+'.inputs.json').read_text())
    if digest(record['config']['data']) != record['data_sha256']: raise ValueError('dataset changed')
    if digest(record['config']['meta']) != record['meta_sha256']: raise ValueError('context changed')
    d = load(record['config']['data']); y = d['probe']['rating'].astype(np.float64)
    users = d['probe']['user'].copy(); del d
    sample_weight=None
    if record['config'].get('sample_weights'):
        weight_path=record['config']['sample_weights']
        if digest(weight_path)!=record['sample_weights_sha256']: raise ValueError('training weights changed')
        sample_weight=np.load(weight_path,allow_pickle=False)
        if sample_weight.shape!=y.shape:raise ValueError('weight row count differs')
    outer = blend_fold(users)
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    raw = np.load(a.bank+'.x.npy', mmap_mode='r')
    p = raw[:, :record['columns']]
    context = np.load(record['config']['meta'])['context']
    saved = np.load(a.bank+'.stats.npz'); stats = tuple(saved[k] for k in ('gram', 'rhs', 'count'))
    shape = (len(y), raw.shape[1]+1)
    if a.reuse_features:
        features=np.load(a.reuse_features,mmap_mode='r')
        if features.shape!=shape: raise ValueError('cached feature shape differs')
        rows=np.linspace(0,len(y)-1,min(256,len(y)),dtype=int)
        block=np.array(raw[rows],dtype=np.float32,order='F');bp=block[:,:record['columns']]
        bm=bp.mean(1);expected=np.empty((len(rows),shape[1]),dtype=np.float32)
        expected[:,0]=bm-3.6;expected[:,1:1+bp.shape[1]]=bp-bm[:,None]
        expected[:,1+bp.shape[1]:]=block[:,bp.shape[1]:]
        if not np.array_equal(features[rows],expected): raise ValueError('cached encoding differs')
        record['reused_feature_sha256']=digest(a.reuse_features)
    else:
        features = (np.lib.format.open_memmap(a.output+'.features.npy', mode='w+', dtype='<f4',
                                        shape=shape, fortran_order=True)
                if a.disk_features else np.empty(shape, dtype=np.float32))
        mean = p.mean(1)
        features[:, 0] = mean-3.6
        for j in range(p.shape[1]): features[:, 1+j] = p[:, j]-mean
        features[:, 1+p.shape[1]:] = raw[:, p.shape[1]:]
    out = np.empty(len(y)); records = []
    zero_columns=[]
    if a.drop_unrated:
        names=record['extra_meta_names']
        zero_columns=[features.shape[1]-len(names)+names.index('log_unrated_n')]
    print(json.dumps(dict(stage='start', config=vars(a), columns=p.shape[1], features=features.shape[1])), flush=True)
    for fold in range(5):
        start = time.monotonic()
        tr, held = outer != fold, outer == fold
        fit, valid = tr & ~inner, tr & inner
        x = standardized(features, tr, output=a.output+'.normalized.npy' if a.disk_features else None,
                         zero_columns=zero_columns)
        inner_base = predict_ridge(p, context, weights(stats, fold, True))
        base = predict_ridge(p, context, weights(stats, fold))
        correction = np.zeros(held.sum()); chosen = []
        for seed in a.seeds:
            _, epochs = train_device(x[fit], y[fit]-inner_base[fit], a, seed+fold, a.epochs,
                                     valid=((x[valid], y[valid]-inner_base[valid]) if sample_weight is None else
                                            (x[valid], y[valid]-inner_base[valid],sample_weight[valid])),
                                     sample_weights=None if sample_weight is None else sample_weight[fit])
            chosen.append(epochs)
            print(json.dumps(dict(stage='selection', fold=fold, seed=seed, epochs=epochs)), flush=True)
            if epochs:
                model, _ = train_device(x[tr], y[tr]-base[tr], a, seed+fold, epochs,
                                        sample_weights=None if sample_weight is None else sample_weight[tr])
                correction += predict_device(model, x[held], a.device)/len(a.seeds)
            print(json.dumps(dict(stage='refit', fold=fold, seed=seed, seconds=time.monotonic()-start)), flush=True)
        out[held] = base[held]+correction
        out[held].astype('<f4').tofile(a.output+f'.fold{fold}.f32')
        rec = dict(fold=fold, nn_rmse=rmse(out[held], y[held]),
                   ridge_rmse=rmse(base[held], y[held]), chosen_epochs=chosen,
                   seconds=time.monotonic()-start)
        if sample_weight is not None:
            rec['weighted_nn_rmse']=float(np.sqrt(np.average((np.clip(out[held],1,5)-y[held])**2,weights=sample_weight[held])))
        records.append(rec); print(json.dumps(rec), flush=True)
        del x
    out.astype('<f4').tofile(a.output+'.nn.f32')
    report = dict(config=vars(a), bank_input_sha256=digest(a.bank+'.inputs.json'),
                  reused_feature_sha256=record.get('reused_feature_sha256'),
                  stats_sha256=digest(a.bank+'.stats.npz'), folds=records,
                  nn_rmse=rmse(out, y), nn_fold_rmse=[rmse(out[outer == k], y[outer == k]) for k in range(5)],
                  metric=record['config'].get('metric_label','Original Probe')+' user-fold development CV; no private-test claim',
                  torch_version=torch.__version__, target_rmse=.85725)
    if sample_weight is not None:
        report['weighted_nn_rmse']=float(np.sqrt(np.average((np.clip(out,1,5)-y)**2,weights=sample_weight)))
        report['weighted_metric_limitation']='Inverse-inclusion estimate, not exact full original Probe or qualifying Test'
    tree = Path(a.bank+'.refit.f32')
    if tree.exists():
        avg = (out+np.fromfile(tree, dtype='<f4'))/2
        avg.astype('<f4').tofile(a.output+'.avg.f32')
        report.update(avg_rmse=rmse(avg, y), avg_fold_rmse=[rmse(avg[outer == k], y[outer == k]) for k in range(5)])
    Path(a.output+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k.endswith('rmse')}), flush=True)


if __name__ == '__main__': main()
