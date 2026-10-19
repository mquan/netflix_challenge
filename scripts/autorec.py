#!/usr/bin/env python3
"""User-based denoising autoencoder (AutoRec-style) on the Netflix training CSR.

Input per user: centered training ratings plus an indicator of every known
(train, probe, qualifying) movie, i.e. the same implicit feedback as N(u).
Loss is computed only on training ratings; a random fraction of those ratings is
hidden from the input, so the network learns to predict unseen movies. Probe
labels are used only to choose the epoch, unless --fixed-epoch is requested.
All-zero qualifying-label sentinels require --fixed-epoch and are never scored.
Requires numpy and torch.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gbdt_blend import load  # noqa: E402

CENTER = 3.6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--fixed-epoch', action='store_true',
                    help='keep the final requested epoch without query-label selection')
    ap.add_argument('--save-epoch', type=int, action='append', default=[],
                    help='also write this fixed trajectory checkpoint as PREFIX.eN.f32')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--input-drop', type=float, default=.25)
    ap.add_argument('--dropout', type=float, default=.5)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = ap.parse_args()
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    if args.save_epoch and (not args.fixed_epoch or any(e < 1 or e > args.epochs for e in args.save_epoch)):
        raise ValueError('saved epochs require --fixed-epoch and must lie within its budget')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    d = load(args.data)
    users, items = d['users'], d['items']
    to = torch.from_numpy(d['train_offsets'].astype(np.int64))
    t_items = torch.from_numpy(d['train_items'].astype(np.int64) - 1)
    t_ratings = torch.from_numpy(d['train_ratings'].astype(np.float32) - CENTER)
    io = torch.from_numpy(d['implicit_offsets'].astype(np.int64))
    i_items = torch.from_numpy(d['implicit_items'].astype(np.int64) - 1)
    probe = d['probe']
    p_user = probe['user'].astype(np.int64)
    p_item = torch.from_numpy(probe['item'].astype(np.int64) - 1)
    p_rating = probe['rating'].astype(np.float64)
    labeled_queries = bool(np.all((p_rating >= 1) & (p_rating <= 5)))
    if not len(p_rating) or (not labeled_queries and
                            (not args.fixed_epoch or not np.all(p_rating == 0))):
        raise ValueError('unknown query labels must all be zero and require --fixed-epoch')
    order = np.argsort(p_user, kind='stable')
    po = np.searchsorted(p_user[order], np.arange(users + 1))
    probe_users = np.unique(p_user)

    def gather(batch_users, offsets, values):
        lo, hi = offsets[batch_users], offsets[batch_users + 1]
        counts = hi - lo
        rows = torch.repeat_interleave(torch.arange(len(batch_users)), counts)
        starts = torch.repeat_interleave(lo - torch.cumsum(counts, 0) + counts, counts)
        idx = torch.arange(int(counts.sum())) + starts
        return rows, idx

    layers, width = [], 2 * items
    for _ in range(args.layers):
        layers += [torch.nn.Linear(width, args.hidden), torch.nn.SELU()]
        width = args.hidden
    layers += [torch.nn.Dropout(args.dropout), torch.nn.Linear(width, items)]
    net = torch.nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch_input(bu, drop):
        bu_t = torch.from_numpy(bu)
        r_rows, r_idx = gather(bu_t, to, None)
        k_rows, k_idx = gather(bu_t, io, None)
        x = torch.zeros(len(bu), 2 * items)
        cols, vals = t_items[r_idx], t_ratings[r_idx]
        keep = torch.rand(len(r_idx)) >= drop if drop > 0 else torch.ones(len(r_idx), dtype=torch.bool)
        # Scale each block by 1/sqrt(count), as for N(u) in SVD++, so heavy
        # raters do not saturate the first layer.
        r_norm = torch.bincount(r_rows[keep], minlength=len(bu)).clamp(min=1).float().rsqrt()
        k_norm = torch.bincount(k_rows, minlength=len(bu)).clamp(min=1).float().rsqrt()
        x[r_rows[keep], cols[keep]] = vals[keep] * r_norm[r_rows[keep]]
        x[k_rows, items + i_items[k_idx]] = k_norm[k_rows]
        return x.to(dev, non_blocking=True), r_rows, cols, vals

    def evaluate():
        net.eval()
        pred = np.zeros(len(p_rating), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(probe_users), 2048):
                bu = probe_users[s:s + 2048]
                x, _, _, _ = batch_input(bu, 0)
                out = net(x).cpu()
                for row, u in enumerate(bu):
                    rows = order[po[u]:po[u + 1]]
                    pred[rows] = out[row, p_item[rows]].numpy() + CENTER
        net.train()
        if not np.isfinite(pred).all():
            raise FloatingPointError('nonfinite predictions')
        score = float(np.sqrt(np.mean((p_rating - np.clip(pred, 1, 5)) ** 2))) if labeled_queries else None
        return pred, score

    print(f'autorec hidden={args.hidden} layers={args.layers} batch={args.batch} lr={args.lr} '
          f'input_drop={args.input_drop} dropout={args.dropout} wd={args.weight_decay} device={dev}', flush=True)
    active = np.flatnonzero(np.diff(d['train_offsets']) > 0)
    best, stale = np.inf, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        rng.shuffle(active)
        sse, n = 0.0, 0
        for s in range(0, len(active), args.batch):
            bu = active[s:s + args.batch]
            x, r_rows, cols, vals = batch_input(bu, args.input_drop)
            out = net(x)
            target = vals.to(dev)
            err = out[r_rows.to(dev), cols.to(dev)] - target
            loss = (err * err).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sse += float(loss.detach()) * len(vals); n += len(vals)
        pred, score = evaluate()
        if epoch in args.save_epoch:
            snapshot = pathlib.Path(args.predictions).with_suffix(f'.e{epoch}.f32')
            pred.astype('<f4').tofile(snapshot)
        if args.fixed_epoch or score < best:
            best, stale = score, 0
            pred.astype('<f4').tofile(args.predictions)
        else:
            stale += 1
        metric = f'probe_rmse={score:.10f} best={best:.10f} ' if labeled_queries else 'query_labels=unknown '
        print(f'epoch={epoch} train_rmse={np.sqrt(sse / n):.10f} {metric}'
              f'seconds={time.time() - start:.1f}', flush=True)
        if not args.fixed_epoch and stale >= args.patience:
            print(f'early_stop patience={args.patience}', flush=True)
            break


if __name__ == '__main__':
    main()
