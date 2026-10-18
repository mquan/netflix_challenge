#!/usr/bin/env python3
"""Netflix probe adapter for the published DeepRecommender training recipe.

Uses only training ratings as input and loss targets. Probe labels select the
best checkpoint, so its score is a development result, not a held-out test.
Architecture and optimization follow github.com/NVIDIA/DeepRecommender.
"""
import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from gbdt_blend import load


class DeepRecommender(nn.Module):
    def __init__(self, items, widths=(512, 512, 1024), dropout=.8):
        super().__init__()
        dims = (items, *widths)
        self.encoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.decoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:0:-1], dims[-2::-1]))
        self.dropout = nn.Dropout(dropout)
        for layer in (*self.encoder, *self.decoder):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.encoder:
            x = nn.functional.selu(layer(x))
        x = self.dropout(x)
        for layer in self.decoder:
            x = nn.functional.selu(layer(x))
        return x


def training_csr(d):
    # The source file is already grouped by user; CSR preserves that order.
    return sp.csr_matrix((d['train_ratings'].astype(np.float32),
                          d['train_items'].astype(np.int32) - 1,
                          d['train_offsets'].astype(np.int64)),
                         shape=(d['users'], d['items']))


def predict(net, matrix, probe, device, batch):
    order = np.argsort(probe['user'], kind='stable')
    sorted_user = probe['user'][order]
    users, starts, counts = np.unique(sorted_user, return_index=True, return_counts=True)
    pred = np.empty(len(probe), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for start in range(0, len(users), batch):
            bu = users[start:start + batch]
            x = torch.from_numpy(matrix[bu].toarray()).to(device)
            out = net(x).cpu().numpy()
            n = counts[start:start + len(bu)]
            dest = np.concatenate([order[s:s + k] for s, k in zip(starts[start:start + len(bu)], n)])
            row = np.repeat(np.arange(len(bu)), n)
            pred[dest] = out[row, probe['item'][dest].astype(np.int64) - 1]
    return pred


def train(a):
    for target in (a.predictions, a.model, a.report):
        path = pathlib.Path(target)
        if path.exists():
            raise ValueError(f'output exists: {target}')
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(a.threads)
    d = load(a.data)
    csr = training_csr(d)
    net = DeepRecommender(d['items'], tuple(a.widths), a.dropout).to(a.device)
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=.9)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[24, 36, 48, 66, 72], gamma=.5)
    users = np.flatnonzero(np.diff(csr.indptr))
    y = d['probe']['rating'].astype(np.float64)
    best, stale, records = np.inf, 0, []
    print(json.dumps(dict(config=vars(a), torch=torch.__version__, users=len(users),
                          ratings=csr.nnz, probe=len(y))), flush=True)
    for epoch in range(1, a.epochs + 1):
        start = time.monotonic()
        rng.shuffle(users)
        net.train()
        sse, nr, steps = 0., 0, 0
        for begin in range(0, len(users), a.batch):
            bu = users[begin:begin + a.batch]
            x = torch.from_numpy(csr[bu].toarray()).to(a.device)
            n = int(np.count_nonzero(csr[bu].data))
            out = net(x)
            loss = ((out - x).square() * (x != 0)).sum() / n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sse += float(loss.detach()) * n
            nr += n
            if a.augmentation:
                # Original run.py re-feeds the dense output once and trains it
                # toward itself; the re-fed tensor is detached first.
                aug = out.detach()
                generated = net(aug)
                loss2 = (generated - aug).square().mean()
                opt.zero_grad(set_to_none=True)
                loss2.backward()
                opt.step()
            steps += 1
            if steps % 250 == 0:
                print(json.dumps(dict(epoch=epoch, step=steps, train_rmse=float(np.sqrt(sse/nr)),
                                      seconds=round(time.monotonic()-start, 1))), flush=True)
            if a.max_batches and steps >= a.max_batches:
                break
        train_seconds = time.monotonic() - start
        if a.max_batches:
            print(json.dumps(dict(pilot_batches=steps, seconds=train_seconds,
                                  projected_epoch_seconds=train_seconds * np.ceil(len(users)/a.batch)/steps)), flush=True)
            break
        pred = predict(net, csr, d['probe'], a.device, a.eval_batch)
        score = float(np.sqrt(np.mean((y - np.clip(pred.astype(np.float64), 1, 5)) ** 2)))
        improved = score < best
        best, stale = min(best, score), 0 if improved else stale + 1
        if improved:
            temp = pathlib.Path(a.predictions + '.tmp')
            pred.astype('<f4').tofile(temp)
            temp.replace(a.predictions)
            temp = pathlib.Path(a.model + '.tmp')
            torch.save(dict(epoch=epoch, state=net.state_dict(), config=vars(a)), temp)
            temp.replace(a.model)
        record = dict(epoch=epoch, train_rmse=float(np.sqrt(sse/nr)), probe_rmse=score,
                      best=best, train_seconds=train_seconds, seconds=time.monotonic()-start)
        records.append(record)
        print(json.dumps(record), flush=True)
        pathlib.Path(a.report).write_text(json.dumps(dict(config=vars(a), epochs=records,
            status='running', best=best), indent=2) + '\n')
        sched.step()
        if stale >= a.patience:
            break
    if not a.max_batches:
        pathlib.Path(a.report).write_text(json.dumps(dict(config=vars(a), epochs=records,
            status='complete', best=best,
            predictions_sha256=hashlib.sha256(pathlib.Path(a.predictions).read_bytes()).hexdigest()), indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--widths', type=int, nargs='+', default=[512, 512, 1024])
    ap.add_argument('--dropout', type=float, default=.8)
    ap.add_argument('--lr', type=float, default=.005)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--eval-batch', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=102)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--augmentation', type=int, choices=[0, 1], default=1)
    ap.add_argument('--max-batches', type=int, default=0, help='timing pilot, no output')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    if min(*a.widths, a.batch, a.eval_batch, a.epochs, a.patience) < 1:
        ap.error('widths, batch sizes, epochs, and patience must be positive')
    train(a)


if __name__ == '__main__':
    main()
