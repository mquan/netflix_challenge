#!/usr/bin/env python3
"""M0 memory/throughput pilot for the moonshot trainer (MOONSHOT_REBUILD.md).

Trains a fixed number of optimizer steps per config on real data, no
prediction pass and no saved model, and records time per step and MPS memory.
The largest-token batches run first (worst case for memory), then random ones.
Not a score: nothing here is evaluated.
"""
import argparse
import dataclasses
import json
import sys
import time

import numpy as np
import torch
from torch import nn

import moonshot_sequence as ms
from gbdt_blend import load
from masked_history import loss_value

CONFIGS = {
    'N1': dict(),
    'M1': dict(d_model=128, layers=4, ff=512, seq_len=3072, attn_chunk=512, head_mlp=256,
               warmup_steps=500, cosine_epochs=12),
}
N1 = dict(head='distribution', loss='mse', rank=64, hidden=256, dropout=.1, input_drop=.5, lr=.001,
          weight_decay=.001, batch_users=512, nnz_budget=131072, seed=20260928, device='mps',
          context='session', suffix_prob=0., time_bins=32, week_days=0, seq_len=1536, recent_frac=.5,
          max_targets=128, d_model=64, layers=2, heads=4, ff=256, batch_tokens=16384, bucket=32,
          windows=1, pad_width=32, pad_rows=64, choice_tokens=1)


def mem():
    return dict(allocated_mb=torch.mps.current_allocated_memory() / 2**20,
                driver_mb=torch.mps.driver_allocated_memory() / 2**20)


def pilot(d, name, worst, steps):
    c = ms.Config(**{**N1, **CONFIGS[name]})
    torch.manual_seed(c.seed)
    model = ms.SequenceModel(d['items'], c, ms.weeks(d)).to(c.device)
    model.initialize_bias(d)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    rng = np.random.default_rng(c.seed)
    counts = np.diff(d['train_offsets']).astype(np.int64)
    lengths = ms.capped_lengths(d, c)
    groups = ms.length_batches(lengths, rng.permutation(np.flatnonzero(counts > 0)), c, rng)
    size = np.array([len(g) * int(lengths[g].max()) for g in groups])
    order = list(np.argsort(-size, kind='stable')[:worst]) + list(rng.choice(len(groups), steps - worst, replace=False))
    host = dataclasses.replace(c, device='cpu')
    peak, rows = 0., []
    model.train()
    for i, g in enumerate(order):
        b = ms.prepare_batch(d, groups[g], host, rng=rng)
        b = {k: v.to(c.device) if torch.is_tensor(v) else v for k, v in b.items()}
        torch.mps.synchronize()
        start = time.monotonic()
        raw = model.forward_batch(b)
        loss = loss_value(model, raw, b['target_ratings'])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        torch.mps.synchronize()
        m = mem()
        peak = max(peak, m['driver_mb'])
        rows.append(dict(step=i, worst=i < worst, users=len(groups[g]), width=int(lengths[groups[g]].max()),
                         tokens=int(b['seq_lengths'].sum()), targets=int(b['target_ratings'].numel()),
                         seconds=time.monotonic() - start, loss=float(loss), **m))
        print(json.dumps(dict(config=name, **rows[-1])), flush=True)
    random = [r for r in rows if not r['worst']][3:]  # skip warm-up steps
    tok_per_s = sum(r['tokens'] for r in random) / sum(r['seconds'] for r in random)
    est_epoch_h = int(lengths[counts > 0].sum()) / tok_per_s / 3600
    out = dict(config=name, overrides=CONFIGS[name], parameters=params, batches_per_epoch=len(groups),
               peak_driver_mb=peak, worst_batch_seconds=max(r['seconds'] for r in rows if r['worst']),
               random_tokens_per_second=tok_per_s, estimated_epoch_hours=est_epoch_h)
    del model, optimizer
    torch.mps.empty_cache()
    return out, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--configs', default='N1,M1')
    p.add_argument('--worst', type=int, default=5)
    p.add_argument('--steps', type=int, default=40)
    a = p.parse_args()
    torch.set_num_threads(1)
    d = load(a.data)
    report = dict(data=a.data, torch_version=torch.__version__, source_sha256=ms.digest(ms.__file__),
                  note='memory/throughput pilot only; no score', results=[], steps={})
    for name in a.configs.split(','):
        out, rows = pilot(d, name, a.worst, a.steps)
        report['results'].append(out)
        report['steps'][name] = rows
        ms.atomic_json(a.output, report)
        print(json.dumps(out), flush=True)


if __name__ == '__main__':
    sys.exit(main())
