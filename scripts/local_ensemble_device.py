#!/usr/bin/env python3
"""CPU/GPU execution adapters; statistical recipes stay in the original modules."""
import argparse
import copy
import json
import importlib.metadata
from pathlib import Path
import sys
import numpy as np


def choose_device(name):
    global torch
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable in this PyTorch installation')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Apple GPU requested but unavailable')
    return torch.device(name)


def nn_predict(model, x):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(b, device=device)).squeeze(1).cpu().numpy()
                               for b in np.array_split(x, max(1, (len(x)+16383)//16384))])


def nn_train(device, module, x, r, a, seed, epochs, valid=None):
    torch.manual_seed(seed)
    model = module.net(x.shape[1], a.hidden, a.layers, a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tx = torch.as_tensor(x, device=device)
    tr = torch.as_tensor(r.astype(np.float32), device=device)
    best, best_epoch, state = float('inf'), 0, None
    for epoch in range(1, epochs+1):
        model.train()
        for batch in torch.randperm(len(tx)).split(a.batch):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            (model(tx[batch]).squeeze(1)-tr[batch]).square().mean().backward()
            opt.step()
        if valid is None:
            continue
        score = float(np.mean((nn_predict(model, valid[0])-valid[1])**2))
        if score < best:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch-best_epoch >= a.patience:
            break
    if valid is not None:
        model.load_state_dict(state)
    return model, best_epoch


def enn_predict(model, tx, rows):
    device = next(model.parameters()).device
    tx = tx.to(device)
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(tx[b.to(device)]).cpu().numpy() for b in rows.split(4096)])


def enn_train(device, module, tx, index, rows, r, a, seed, epochs, valid=None):
    torch.manual_seed(seed)
    model = module.ENN(index, tx.shape[1], a.hidden, seed).to(device)
    tx = tx.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tr = torch.as_tensor(r.astype(np.float32), device=device)
    n = len(index)
    snap = {k: v.detach().clone() for k, v in model.state_dict().items() if k != 'mask'}
    if valid is None:
        target = torch.as_tensor(np.asarray(epochs))
        last = int(target.max()) if n else 0
    else:
        vr = valid[1].astype(np.float64)
        best = np.full(n, np.mean(vr**2))
        best_epoch = np.zeros(n, dtype=np.int64)
        last = epochs
    history = [float(np.sqrt(best.mean()))] if valid is not None else []
    for epoch in range(1, last+1):
        model.train()
        for batch in torch.randperm(len(rows)).split(a.batch):
            opt.zero_grad(set_to_none=True)
            out = model(tx[rows[batch].to(device)])
            (out-tr[batch.to(device), None]).square().mean(0).sum().backward()
            opt.step()
        if valid is None:
            done = target == epoch
            if done.any():
                mask = done.to(device)
                for k in snap:
                    snap[k][mask] = model.state_dict()[k][mask]
            continue
        score = np.mean((enn_predict(model, tx, valid[0])-vr[:, None])**2, axis=0)
        better = score < best
        best[better], best_epoch[better] = score[better], epoch
        if better.any():
            mask = torch.as_tensor(better, device=device)
            for k in snap:
                snap[k][mask] = model.state_dict()[k][mask]
        history.append(float(np.sqrt(score.mean())))
        if a.verbose:
            print(json.dumps(dict(epoch=epoch, valid_rmse=history[-1], improved=int(better.sum()))), flush=True)
        if np.all(epoch-best_epoch >= a.patience):
            break
    model.load_state_dict(snap, strict=False)
    return model, (None if valid is None else best_epoch), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['tree', 'nn', 'enn'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--tree-threads', type=int, default=2)
    a, remaining = parser.parse_known_args()
    if a.tree_threads < 1:
        parser.error('tree-threads must be positive')
    sys.argv = [sys.argv[0], *remaining]
    if a.family == 'tree':
        import round4_boost as module
        original = module.lgb.train
        def train(params, *args, **kwargs):
            return original(dict(params, num_threads=a.tree_threads), *args, **kwargs)
        module.lgb.train = train
        device = 'cpu'
    else:
        device = choose_device(a.device)
        if a.family == 'nn':
            import round8_nn_blend as module
            module.train = lambda *args, **kwargs: nn_train(device, module, *args, **kwargs)
            module.predict = nn_predict
        else:
            import round9_enn_blend as module
            module.train = lambda *args, **kwargs: enn_train(device, module, *args, **kwargs)
            module.predict = enn_predict
    print(json.dumps(dict(execution_device=str(device), family=a.family,
                          tree_threads=a.tree_threads)), flush=True)
    module.main()
    output = remaining[remaining.index('--output')+1]
    path = Path(output+'.json')
    report = json.loads(path.read_text())
    report['execution'] = dict(device=str(device), tree_threads=a.tree_threads,
                               torch_version=importlib.metadata.version('torch'),
                               limitation='GPU/backend RNG and floating-point differences require the matched fresh control')
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
