#!/usr/bin/env python3
"""Item-based AutoRec (I-AutoRec, Sedhain et al. 2015) on the Netflix training CSR.

Each movie is encoded from the users who rated it: centered training ratings
plus an indicator of every known (train, probe, qualifying) rater, i.e. the
item-side counterpart of N(u). The network is kept sparse: the first layer is
a weighted sum of per-user input embeddings, and the reconstruction is only
evaluated at the needed users, as h_i . V[u] + c[u]. Loss uses training
ratings only; a random fraction of each movie's ratings is hidden from its
input. Probe labels are used only to choose the epoch, as with other models.
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


def item_major(offsets, items, n_items, *values):
    """Transpose a user-major CSR into item-major (item offsets, users, values...)."""
    owner = np.repeat(np.arange(len(offsets) - 1, dtype=np.int32), np.diff(offsets))
    order = np.argsort(items, kind='stable')
    io = np.zeros(n_items + 1, dtype=np.int64)
    np.cumsum(np.bincount(items, minlength=n_items), out=io[1:])
    return (io, owner[order]) + tuple(v[order] for v in values)


def batches(offsets, ids, budget):
    """Split ids into consecutive groups whose total nnz stays under budget."""
    out, cur, size = [], [], 0
    for i in ids:
        n = int(offsets[i + 1] - offsets[i])
        if cur and size + n > budget:
            out.append(np.array(cur)); cur, size = [], 0
        cur.append(i); size += n
    if cur:
        out.append(np.array(cur))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--nnz-budget', type=int, default=250_000, help='ratings per batch')
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--input-drop', type=float, default=.25)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--no-implicit', action='store_true')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    d = load(args.data)
    users, items = d['users'], d['items']
    to, t_users, t_ratings = item_major(
        d['train_offsets'].astype(np.int64), d['train_items'].astype(np.int64) - 1, items,
        d['train_ratings'].astype(np.float32) - CENTER)
    ko, k_users = item_major(d['implicit_offsets'].astype(np.int64),
                             d['implicit_items'].astype(np.int64) - 1, items)
    probe = d['probe']
    p_user = probe['user'].astype(np.int64)
    p_item = probe['item'].astype(np.int64) - 1
    p_rating = probe['rating'].astype(np.float64)
    order = np.argsort(p_item, kind='stable')
    po = np.searchsorted(p_item[order], np.arange(items + 1))
    t_users_d = torch.from_numpy(t_users.astype(np.int64)).to(dev)
    t_ratings_d = torch.from_numpy(t_ratings).to(dev)
    k_users_d = torch.from_numpy(k_users.astype(np.int64)).to(dev)
    p_user_d = torch.from_numpy(p_user[order]).to(dev)

    h = args.hidden
    scale = h ** -.5
    w_in = torch.nn.Parameter(torch.randn(users, h, device=dev) * scale)
    w_imp = torch.nn.Parameter(torch.randn(users, h, device=dev) * scale)
    b_in = torch.nn.Parameter(torch.zeros(h, device=dev))
    v_out = torch.nn.Parameter(torch.randn(users, h, device=dev) * scale)
    c_out = torch.nn.Parameter(torch.zeros(users, device=dev))
    params = [w_in, b_in, v_out, c_out] + ([] if args.no_implicit else [w_imp])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    drop = torch.nn.Dropout(args.dropout)

    def count(rows, n):
        # torch.bincount is very slow on MPS; index_add is not.
        return torch.zeros(n, device=dev).index_add(0, rows, torch.ones(len(rows), device=dev))

    def segment(offsets, batch):
        """Flat indices and batch-row ids of every entry of the batch's items."""
        lo, hi = offsets[batch], offsets[batch + 1]
        counts = hi - lo
        rows = np.repeat(np.arange(len(batch)), counts)
        idx = np.arange(counts.sum()) + np.repeat(lo - np.cumsum(counts) + counts, counts)
        return torch.from_numpy(rows).to(dev), torch.from_numpy(idx).to(dev)

    def encode(batch, drop_rate):
        r_rows, r_idx = segment(to, batch)
        u, r = t_users_d[r_idx], t_ratings_d[r_idx]
        keep = (torch.rand(len(r_idx), device=dev) >= drop_rate) if drop_rate > 0 else None
        kr, ku, kv = (r_rows, u, r) if keep is None else (r_rows[keep], u[keep], r[keep])
        # Scale each block by 1/sqrt(count), as for N(u) in SVD++.
        norm = count(kr, len(batch)).clamp(min=1).rsqrt()
        pre = torch.zeros(len(batch), h, device=dev).index_add(
            0, kr, w_in[ku] * (kv * norm[kr])[:, None])
        if not args.no_implicit:
            k_rows, k_idx = segment(ko, batch)
            knorm = count(k_rows, len(batch)).clamp(min=1).rsqrt()
            pre = pre.index_add(0, k_rows, w_imp[k_users_d[k_idx]] * knorm[k_rows][:, None])
        return drop(torch.sigmoid(pre + b_in)), r_rows, u, r

    def evaluate():
        drop.eval()
        pred = np.full(len(p_rating), CENTER, dtype=np.float32)
        with torch.no_grad():
            for batch in batches(to, np.arange(items), args.nnz_budget):
                code, _, _, _ = encode(batch, 0)
                lo, hi = po[batch], po[batch + 1]
                counts = hi - lo
                rows = torch.from_numpy(np.repeat(np.arange(len(batch)), counts)).to(dev)
                sel = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
                u = p_user_d[torch.from_numpy(sel).to(dev)]
                out = (code[rows] * v_out[u]).sum(1) + c_out[u]
                pred[order[sel]] = out.cpu().numpy() + CENTER
        drop.train()
        return pred, float(np.sqrt(np.mean((p_rating - np.clip(pred, 1, 5)) ** 2)))

    print(f'autorec-item hidden={h} nnz_budget={args.nnz_budget} lr={args.lr} '
          f'input_drop={args.input_drop} dropout={args.dropout} wd={args.weight_decay} '
          f'implicit={not args.no_implicit} device={dev}', flush=True)
    active = np.flatnonzero(np.diff(to) > 0)
    best, stale = np.inf, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        rng.shuffle(active)
        sse, n = 0.0, 0
        for batch in batches(to, active, args.nnz_budget):
            code, rows, u, r = encode(batch, args.input_drop)
            err = (code[rows] * v_out[u]).sum(1) + c_out[u] - r
            loss = (err * err).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sse += float(loss.detach()) * len(r); n += len(r)
        pred, score = evaluate()
        if score < best:
            best, stale = score, 0
            pred.astype('<f4').tofile(args.predictions)
        else:
            stale += 1
        print(f'epoch={epoch} train_rmse={np.sqrt(sse / n):.10f} probe_rmse={score:.10f} '
              f'best={best:.10f} seconds={time.time() - start:.1f}', flush=True)
        if stale >= args.patience:
            print(f'early_stop patience={args.patience}', flush=True)
            break


if __name__ == '__main__':
    main()
