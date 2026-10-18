#!/usr/bin/env python3
"""Masked rating-history pilot inspired by CF-NADE, not an exact reproduction.

Shared item/rating embeddings encode a random context; a factored softmax
decoder predicts disjoint targets. CE + ranked probability score trains the
five-star distribution. Sparse and session arms change one setting each.
Probe ratings select epochs only: all reported scores are development scores.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn
from gbdt_blend import load


def split_rows(n, rng, context_cap, targets):
    """Random ordering/prefix, capped for compute; context and targets disjoint."""
    if n < 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    cut = int(rng.integers(0, n))
    nc, nt = min(cut, context_cap), min(n - cut, targets)
    rows = rng.choice(n, nc + nt, replace=False)
    return rows[:nc], rows[nc:]


def batch(d, users, rng, cap, targets, training=True):
    b = len(users)
    token = np.zeros((b, cap), np.int64)
    days = np.zeros((b, cap), np.float32)
    item = np.zeros((b, targets), np.int64)
    target_days = np.zeros((b, targets), np.float32)
    labels = np.zeros((b, targets), np.int64)
    mask = np.zeros((b, targets), bool)
    for row, u in enumerate(users):
        lo, hi = map(int, d['train_offsets'][u:u + 2])
        if training:
            c, t = split_rows(hi - lo, rng, cap, targets)
        else:
            # Fixed, evenly spaced history sample, independent of probe labels.
            c = np.linspace(0, hi - lo - 1, min(hi - lo, cap), dtype=np.int64)
            t = np.array([], dtype=np.int64)
        c, t = c + lo, t + lo
        # Token zero is padding; the five possible ratings have separate tokens.
        token[row, :len(c)] = 1 + (d['train_items'][c].astype(np.int64) - 1) * 5 + d['train_ratings'][c] - 1
        days[row, :len(c)] = d['train_days'][c]
        item[row, :len(t)] = d['train_items'][t] - 1
        labels[row, :len(t)] = d['train_ratings'][t] - 1
        target_days[row, :len(t)] = d['train_days'][t]
        mask[row, :len(t)] = True
    return dict(token=token, days=days, item=item, target_days=target_days,
                labels=labels, mask=mask)


class HistoryNet(nn.Module):
    def __init__(self, items, rank=64, hidden=128, session=False):
        super().__init__()
        self.rank, self.session = rank, session
        self.input = nn.Embedding(items * 5 + 1, rank, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(rank + 1, hidden), nn.Tanh(),
                                     nn.Dropout(.15), nn.Linear(hidden, rank))
        self.output = nn.Embedding(items, rank * 5)
        self.bias = nn.Embedding(items, 5)
        if session:
            self.local = nn.Linear(rank, rank, bias=False)
        nn.init.normal_(self.input.weight, std=.05)
        nn.init.normal_(self.output.weight, std=.05)
        nn.init.zeros_(self.bias.weight)
        with torch.no_grad():
            self.input.weight[0].zero_()

    def forward(self, token, days, item, target_days):
        valid = token != 0
        emb = self.input(token) * valid[..., None]
        count = valid.sum(1).clamp(min=1).to(emb.dtype)
        pooled = emb.sum(1) / count.sqrt()[:, None]
        context = self.encoder(torch.cat([pooled, torch.log1p(count)[:, None] / 5], 1))
        context = context[:, None, :].expand(-1, item.shape[1], -1)
        if self.session:
            weight = torch.exp(-torch.abs(target_days[:, :, None] - days[:, None, :]) / 30)
            weight = weight * valid[:, None, :]
            local = torch.bmm(weight, emb) / weight.sum(-1).clamp(min=1).sqrt()[..., None]
            context = context + self.local(local)
        out = self.output(item).reshape(*item.shape, 5, self.rank)
        return (out * context[:, :, None, :]).sum(-1) + self.bias(item)


def objective(logits, labels, mask, weights, ordinal=.2):
    ce = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').reshape_as(labels)
    cdf = logits.softmax(-1).cumsum(-1)[..., :4]
    truth = labels[..., None] <= torch.arange(4, device=labels.device)
    rps = ((cdf - truth.to(cdf.dtype)) ** 2).mean(-1)
    w = mask * weights[:, None]
    return ((ce + ordinal * rps) * w).sum() / w.sum().clamp(min=1)


def tensor_batch(values, device):
    return {k: torch.from_numpy(v).to(device) for k, v in values.items()}


def predict(net, d, device, cap, batch_size):
    probe = d['probe']
    order = np.argsort(probe['user'], kind='stable')
    offsets = np.searchsorted(probe['user'][order], np.arange(d['users'] + 1))
    users = np.flatnonzero(np.diff(offsets))
    query_counts = np.diff(offsets)
    pred = np.empty(len(probe), np.float32)
    net.eval()
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            bu = users[start:start + batch_size]
            # Bound query width so prolific probe users cannot allocate huge tensors.
            context = batch(d, bu, None, cap, 1, training=False)
            width = int(np.max(query_counts[bu]))
            for chunk in range(0, width, 32):
                rows = [order[offsets[u] + chunk:min(offsets[u + 1], offsets[u] + chunk + 32)] for u in bu]
                item, days = np.zeros((len(bu), 32), np.int64), np.zeros((len(bu), 32), np.float32)
                for j, rr in enumerate(rows):
                    item[j, :len(rr)] = probe['item'][rr] - 1
                    days[j, :len(rr)] = probe['day'][rr]
                context.update(item=item, target_days=days)
                tb = tensor_batch(context, device)
                logits = net(tb['token'], tb['days'], tb['item'], tb['target_days'])
                value = (logits.softmax(-1) * torch.arange(1, 6, device=device)).sum(-1).cpu().numpy()
                for j, rr in enumerate(rows):
                    pred[rr] = value[j, :len(rr)]
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--arm', choices=['history', 'sparse', 'session'], default='history')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--context', type=int, default=128)
    ap.add_argument('--targets', type=int, default=16)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--max-batches', type=int, default=0, help='timing pilot only; zero trains all users')
    ap.add_argument('--fixed-epochs', action='store_true', help='save final epoch, independent of probe labels')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    if min(a.epochs, a.context, a.targets, a.batch, a.rank, a.hidden) < 1:
        ap.error('sizes and epochs must be positive')
    for path in (a.predictions, a.model):
        if pathlib.Path(path).exists():
            ap.error('output exists: ' + path)
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = load(a.data)
    net = HistoryNet(d['items'], a.rank, a.hidden, a.arm == 'session').to(a.device)
    # Smoothed item rating histogram initializes the decoder from training only.
    histogram = np.ones((d['items'], 5), dtype=np.float64)
    for s in range(0, len(d['train_items']), 1_000_000):
        idx = d['train_items'][s:s + 1_000_000].astype(np.int64) - 1
        rating = d['train_ratings'][s:s + 1_000_000].astype(np.int64) - 1
        np.add.at(histogram, (idx, rating), 1)
    with torch.no_grad():
        net.bias.weight.copy_(torch.tensor(np.log(histogram / histogram.sum(1, keepdims=True)), dtype=torch.float32, device=a.device))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    support = np.diff(d['train_offsets']).astype(np.int64)
    users = np.flatnonzero(support)
    best, stale, records = float('inf'), 0, []
    print(json.dumps(dict(config=vars(a), torch=torch.__version__, numpy=np.__version__, users=len(users))), flush=True)
    for epoch in range(1, a.epochs + 1):
        start = time.monotonic()
        rng.shuffle(users)
        net.train()
        loss_total, steps = 0., 0
        for s in range(0, len(users), a.batch):
            bu = users[s:s + a.batch]
            tb = tensor_batch(batch(d, bu, rng, a.context, a.targets), a.device)
            weights = np.clip(np.sqrt(50 / support[bu]), .5, 3) if a.arm == 'sparse' else np.ones(len(bu))
            logits = net(tb['token'], tb['days'], tb['item'], tb['target_days'])
            loss = objective(logits, tb['labels'], tb['mask'], torch.tensor(weights, dtype=torch.float32, device=a.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
            loss_total += float(loss.detach().cpu()); steps += 1
            if steps % 100 == 0:
                print(f'epoch={epoch} step={steps} loss={loss_total / steps:.6f} seconds={time.monotonic()-start:.1f}', flush=True)
            if a.max_batches and steps >= a.max_batches:
                break
        train_seconds = time.monotonic() - start
        pred = predict(net, d, a.device, a.context, a.batch)
        y = d['probe']['rating'].astype(np.float64)
        score = float(np.sqrt(np.mean((pred.astype(np.float64) - y) ** 2)))
        improved = score < best
        best, stale = min(best, score), 0 if improved else stale + 1
        if improved and not a.fixed_epochs or a.fixed_epochs and epoch == a.epochs:
            # Atomic replace: consumers must still wait for the completion report.
            tmp = pathlib.Path(a.predictions + '.tmp')
            pred.astype('<f4').tofile(tmp); tmp.replace(a.predictions)
            tmp = pathlib.Path(a.model + '.tmp')
            torch.save(dict(config=vars(a), state=net.state_dict(), epoch=epoch), tmp)
            tmp.replace(a.model)
        record = dict(epoch=epoch, loss=loss_total/steps, probe_rmse=score, best=best,
                      train_seconds=train_seconds, seconds=time.monotonic()-start)
        records.append(record)
        print(json.dumps(record), flush=True)
        if not a.fixed_epochs and stale >= a.patience:
            break
    pathlib.Path(a.predictions + '.json').write_text(json.dumps(dict(config=vars(a), epochs=records,
        status='complete', metric='probe development RMSE; checkpoint selection uses probe unless fixed'), indent=2) + '\n')


if __name__ == '__main__':
    main()
