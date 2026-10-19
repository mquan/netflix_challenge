#!/usr/bin/env python3
"""A2 transformer plus dated choice tokens (N1, NEW_BASE_MODELS.md).

Copy of the A2h trainer (worktree agent-ae222b177c95376d0, bf3c719) with one
change behind --choice-tokens 1: movies the user chose but whose rating is not
in training (implicit-only pairs: probe and qualifying, plus P2 in a holdout
dataset) enter the sequence as tokens at their dates with a sixth star value
CHOICE. Only identifiers and dates are read. At prediction time the queried
pairs themselves are dropped from the choice tokens (they are the [MASK]
targets). With --choice-tokens 0 the model equals masked_sequence.py except
for one unused embedding row per item.

Original A2 description:
Masked-target transformer over each user's date-ordered rating history (Bet A2).

Sibling of masked_history.py: same masking (random input drop mixed with
date-suffix masks), same session/date context MLP, item x date-bin biases,
distribution head and MSE loss, and the same selection/refit protocol. Only the
user encoder changes: instead of a bag of item x star embeddings, a small
pre-LN transformer reads a date-ordered token sequence

    [USER] t_1 t_2 ... t_L

where [USER] summarizes the implicit history and visible-rating histogram, and
each t is item + item x star (or item x [MASK]) + star (or [MASK]) + week-of-
date embedding + relative-day features. Hidden training targets and probe
queries enter only as [MASK] tokens at their dates; their ratings never enter
any input. The transformer output at each [MASK] token is that target's user
code, which the session MLP shifts and the item output head decodes.

Long histories are capped per sample at --seq-len tokens: all targets of the
sample, then the most recent visible ratings (--recent-frac of the remaining
budget) and a random subset of older ones. Selection is by dates and random
keys only (hash keys at prediction time, so predictions are deterministic and
independent of query order).
"""
import argparse
import dataclasses
import json
import os
import pathlib
import queue
import threading
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import masked_history as mh
from gbdt_blend import load
from masked_history import atomic_json, digest, gather, loss_value, suffix_mask, write_f32


@dataclasses.dataclass
class Config(mh.Config):
    # Masked-history fields (defaults: the MT2 recipe, i.e. MT1 + 50/50 suffix masks).
    loss: str = 'mse'
    context: str = 'session'
    time_bins: int = 32
    suffix_prob: float = .5
    batch_users: int = 512
    # Sequence encoder.
    seq_len: int = 384
    recent_frac: float = .5
    max_targets: int = 128
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    ff: int = 256
    batch_tokens: int = 65536
    bucket: int = 32
    windows: int = 1
    # Pad batch shapes to few distinct sizes; MPS specializes kernels per shape.
    pad_width: int = 32
    pad_rows: int = 64
    choice_tokens: int = 0


MASK = 5
CHOICE = 6
STARS = 7
TIME_FEATURES = 3


def splitmix(x):
    x = x.astype(np.uint64) + np.uint64(0x9e3779b97f4a7c15)
    with np.errstate(over='ignore'):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> np.uint64(31))) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def hash_keys(users, items, salt):
    key = (users.astype(np.uint64) << np.uint64(20)) ^ items.astype(np.uint64) ^ (np.uint64(salt) << np.uint64(52))
    return splitmix(key)


def make_keep(d, users, c, rng):
    """masked_history.prepare's training mask, reproduced so the sequence
    builder can see it. Decisions use the RNG and dates, never ratings."""
    rows, index, offsets = gather(d['train_offsets'], users)
    keep = rng.random(len(index)) >= c.input_drop
    if c.suffix_prob > 0:
        keep = suffix_mask(keep, rows, d['train_days'][index].astype(np.int64), offsets, users, c, rng)
    counts = np.diff(offsets)
    visible = np.bincount(rows, keep.astype(np.int64), minlength=len(users))
    hide = np.flatnonzero((counts > 0) & (visible == counts))
    keep[offsets[hide + 1] - 1] = False
    reveal = np.flatnonzero((counts > 1) & (visible == 0))
    keep[offsets[reveal]] = True
    return rows, index, offsets, keep


def rank_within(rows, key):
    """0-based rank of each element by ascending key within its row."""
    order = np.lexsort((key, rows))
    sorted_rows = rows[order]
    start = np.r_[0, np.flatnonzero(np.diff(sorted_rows)) + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(order)]))
    rank = np.empty(len(rows), dtype=np.int64)
    rank[order] = np.arange(len(rows)) - first
    return rank


def select_targets(rows, keep, c, rng):
    """Up to max_targets hidden ratings per row, chosen by the RNG only."""
    hidden = np.flatnonzero(~keep)
    if not len(hidden):
        return hidden
    rank = rank_within(rows[hidden], rng.random(len(hidden)))
    return hidden[rank < c.max_targets]


def window(rows, days, tie, rand, n_rows, n_targets, c):
    """Boolean selection of context tokens: the most recent recent_frac of the
    per-row budget (by date, ties by key) plus random older ones."""
    count = np.bincount(rows, minlength=n_rows)
    budget = np.maximum(c.seq_len - 1 - n_targets, c.seq_len // 4)
    if not len(rows) or (count <= budget).all():
        return np.ones(len(rows), dtype=bool)
    from_end = count[rows] - 1 - rank_within(rows, days + tie)
    n_recent = np.minimum(np.ceil(budget * c.recent_frac).astype(np.int64), budget)
    recent = from_end < n_recent[rows]
    older = np.flatnonzero(~recent)
    take = recent | (count <= budget)[rows]
    if len(older):
        older_rank = rank_within(rows[older], rand[older])
        take[older[older_rank < (budget - n_recent)[rows[older]]]] = True
    return take


def choices(d):
    """CSR of implicit-only (user, item, day) events: chosen, rating not in training."""
    if 'choice_offsets' not in d:
        io = d['implicit_offsets'].astype(np.int64)
        owner = np.repeat(np.arange(d['users'], dtype=np.int64), np.diff(io))
        key = owner * 32768 + d['implicit_items'].astype(np.int64)
        to = d['train_offsets'].astype(np.int64)
        t_owner = np.repeat(np.arange(d['users'], dtype=np.int64), np.diff(to))
        rated = t_owner * 32768 + d['train_items'].astype(np.int64)
        only = ~np.isin(key, rated)
        d['choice_offsets'] = np.r_[0, np.cumsum(np.bincount(owner[only], minlength=d['users']))]
        d['choice_items'] = d['implicit_items'][only].astype(np.int64) - 1
        d['choice_days'] = d['implicit_days'][only].astype(np.int64)
    return d['choice_offsets'], d['choice_items'], d['choice_days']


def sequence_inputs(d, users, rows, index, keep, t_rows, t_items, t_days, c, rng=None, predicting=False):
    """Padded date-ordered token batch. t_* are the [MASK] target tokens in the
    order prepare() uses for its targets; context tokens are visible ratings
    and, with choice_tokens, implicit-only choices (star value CHOICE)."""
    ctx = np.flatnonzero(keep)
    c_rows = rows[ctx]
    c_items = d['train_items'][index[ctx]].astype(np.int64) - 1
    c_ratings = d['train_ratings'][index[ctx]].astype(np.int64) - 1
    c_days = d['train_days'][index[ctx]].astype(np.int64)
    n_rows = len(users)
    if c.choice_tokens:
        offsets, items, days = choices(d)
        k_rows, k_index, _ = gather(offsets, users)
        k_items, k_days = items[k_index], days[k_index]
        if predicting and len(t_rows):
            drop = np.isin(k_rows * 32768 + k_items, t_rows * 32768 + t_items)
            k_rows, k_items, k_days = k_rows[~drop], k_items[~drop], k_days[~drop]
        c_rows, c_items, c_days = np.r_[c_rows, k_rows], np.r_[c_items, k_items], np.r_[c_days, k_days]
        c_ratings = np.r_[c_ratings, np.full(len(k_rows), CHOICE, dtype=np.int64)]
    n_targets = np.bincount(t_rows, minlength=n_rows)
    if rng is None:
        c_tie = hash_keys(users[c_rows], c_items, 1) * .5
        c_rand = hash_keys(users[c_rows], c_items, 2)
        t_tie = hash_keys(users[t_rows], t_items, 1) * .5
    else:
        c_tie, c_rand = rng.random(len(c_rows)) * .5, rng.random(len(c_rows))
        t_tie = rng.random(len(t_rows)) * .5
    take = window(c_rows, c_days, c_tie, c_rand, n_rows, n_targets, c)
    tok_rows = np.r_[c_rows[take], t_rows]
    tok_items = np.r_[c_items[take], t_items]
    tok_ratings = np.r_[c_ratings[take], np.full(len(t_rows), MASK, dtype=np.int64)]
    tok_days = np.r_[c_days[take], t_days]
    tok_target = np.r_[np.full(int(take.sum()), -1, dtype=np.int64), np.arange(len(t_rows))]
    order = np.lexsort((np.r_[c_tie[take], t_tie], tok_days, tok_rows))
    tok_rows, tok_items, tok_ratings, tok_days, tok_target = (
        a[order] for a in (tok_rows, tok_items, tok_ratings, tok_days, tok_target))
    lengths = np.bincount(tok_rows, minlength=n_rows)
    starts = np.r_[0, np.cumsum(lengths)]
    position = np.arange(len(tok_rows)) - starts[tok_rows] + 1  # slot 0 is [USER]
    width = -(-(int(lengths.max(initial=0)) + 1) // c.pad_width) * c.pad_width
    flat = tok_rows * width + position
    # Relative-day features: gap to the previous token in the window, and
    # distance to the user's first/last interaction (implicit days, label-free).
    gap = np.zeros(len(tok_days), dtype=np.float64)
    same_row = np.r_[False, tok_rows[1:] == tok_rows[:-1]]
    gap[same_row] = (tok_days[1:] - tok_days[:-1])[same_row[1:]]
    imp_rows, imp_index, _ = gather(d['implicit_offsets'], users)
    imp_days = d['implicit_days'][imp_index].astype(np.int64)
    first = np.full(n_rows, d['max_day'], dtype=np.int64)
    last = np.full(n_rows, d['min_day'], dtype=np.int64)
    np.minimum.at(first, imp_rows, imp_days)
    np.maximum.at(last, imp_rows, imp_days)
    features = np.column_stack([np.log1p(gap) / 6,
                                np.log1p(np.maximum(last[tok_rows] - tok_days, 0)) / 7,
                                np.log1p(np.maximum(tok_days - first[tok_rows], 0)) / 7])
    target_flat = np.empty(len(t_rows), dtype=np.int64)
    is_target = tok_target >= 0
    target_flat[tok_target[is_target]] = flat[is_target]
    item_star = tok_items * STARS + tok_ratings
    out = dict(seq_flat=flat, seq_items=tok_items, seq_item_star=item_star, seq_stars=tok_ratings,
               seq_weeks=(tok_days - d['min_day']) // 7, seq_features=features.astype(np.float32),
               seq_lengths=lengths + 1, seq_target_flat=target_flat)
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(c.device) for k, v in out.items()}, width


def prepare_batch(d, users, c, rng=None, queries=None):
    """Training (rng) or prediction (queries=(rows, items, days)) batch."""
    if queries is None:
        rows, index, _, keep = make_keep(d, users, c, rng)
        chosen = select_targets(rows, keep, c, rng)
        t_rows = rows[chosen]
        t_items = d['train_items'][index[chosen]].astype(np.int64) - 1
        t_days = d['train_days'][index[chosen]].astype(np.int64)
        t_ratings = d['train_ratings'][index[chosen]].astype(np.int64) - 1
    else:
        rows, index, _ = gather(d['train_offsets'], users)
        keep = np.ones(len(index), dtype=bool)
        t_rows, t_items, t_days = (np.asarray(q, dtype=np.int64) for q in queries)
        t_ratings = np.zeros(0, dtype=np.int64)
    # prepare() builds the session/date context, histogram features and
    # implicit bag from the visible ratings only; targets come in as queries.
    b = mh.prepare(d, users, c, keep=keep, queries=(t_rows, t_items, t_days))
    b['target_ratings'] = torch.from_numpy(t_ratings).to(c.device)
    s, width = sequence_inputs(d, users, rows, index, keep, t_rows, t_items, t_days, c, rng,
                               predicting=queries is not None)
    b.update(s)
    b['seq_width'] = width
    return b


def length_batches(lengths, users, c, rng=None):
    """Bucket users by capped sequence length; each batch respects
    batch_users, batch_tokens (padded) and a raw-rating budget for prepare()."""
    raw_budget = 4 * c.nnz_budget
    chunk = c.batch_users * c.bucket
    out = []
    for start in range(0, len(users), chunk):
        part = users[start:start + chunk]
        part = part[np.argsort(lengths[part], kind='stable')]
        group, width, raw = [], 0, 0
        for user in part:
            n = int(lengths[user])
            w = max(width, n)
            if group and (len(group) == c.batch_users or w * (len(group) + 1) > c.batch_tokens
                          or raw + n > raw_budget):
                out.append(np.array(group, dtype=np.int64))
                group, w, raw = [], n, 0
            group.append(user)
            width, raw = w, raw + n
        if group:
            out.append(np.array(group, dtype=np.int64))
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


class Block(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        n, length, width = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(n, length, width)))
        return x + self.drop(self.ff(self.ln2(x)))


class SequenceModel(mh.HistoryModel):
    def __init__(self, items, c, weeks=400):
        super().__init__(items, c)
        del self.explicit, self.encoder  # replaced by the sequence encoder
        if c.d_model % c.heads:
            raise ValueError('d_model must be divisible by heads')
        dm = c.d_model
        self.item_emb = nn.Parameter(torch.empty(items, dm))
        self.item_star = nn.Parameter(torch.empty(items * STARS, dm))
        self.star_emb = nn.Parameter(torch.empty(STARS, dm))
        self.week_emb = nn.Parameter(torch.empty(weeks, dm))
        for p in (self.item_emb, self.item_star, self.star_emb, self.week_emb):
            nn.init.normal_(p, std=.04)
        self.time_in = nn.Linear(TIME_FEATURES, dm)
        self.user_in = nn.Linear(c.rank + 6, dm)
        self.blocks = nn.ModuleList(Block(dm, c.heads, c.ff, c.dropout) for _ in range(c.layers))
        self.norm = nn.LayerNorm(dm)
        self.code_out = nn.Linear(dm, c.rank)

    def encode(self, b):
        """Per-target user code from the transformer output at its [MASK] token."""
        z = F.embedding_bag(b['implicit_index'], self.implicit, b['implicit_offsets'],
                            mode='sum', include_last_offset=True)
        z = z * b['implicit_counts'].clamp(min=1).rsqrt()[:, None]
        user = self.user_in(torch.cat([z, b['features']], 1))
        n, width = user.shape[0], b['seq_width']
        rows = -(-n // self.config.pad_rows) * self.config.pad_rows
        tokens = (self.item_emb[b['seq_items']] + self.item_star[b['seq_item_star']]
                  + self.star_emb[b['seq_stars']] + self.week_emb[b['seq_weeks'].clamp(max=len(self.week_emb) - 1)]
                  + self.time_in(b['seq_features']))
        x = torch.zeros(rows * width, tokens.shape[1], dtype=tokens.dtype, device=tokens.device)
        x = x.index_copy(0, b['seq_flat'], tokens)
        x = x.index_copy(0, torch.arange(n, device=x.device) * width, user)
        x = x.view(rows, width, -1)
        lengths = torch.cat([b['seq_lengths'], b['seq_lengths'].new_ones(rows - n)])  # pad rows: [USER] slot only
        mask = (torch.arange(width, device=x.device)[None, :] < lengths[:, None])[:, None, None, :]
        for block in self.blocks:
            x = block(x, mask)
        h = self.norm(x).reshape(rows * width, -1)[b['seq_target_flat']]
        return torch.tanh(self.code_out(h))

    def decode(self, code, features, rows, items, b):
        vector = code
        if self.config.context == 'session':
            zero = torch.zeros(1, self.config.rank, dtype=code.dtype, device=code.device)
            parts = [code] + self.channel(b, 'day', self.session_implicit, self.session_explicit, items, zero)
            if self.config.week_days > 1:
                parts += self.channel(b, 'week', self.week_implicit, self.week_explicit, items, zero)
            vector = code + self.session_net(torch.cat(parts + [b['target_features']], 1))
        raw = (self.output[items] * vector[:, None, :]).sum(2) + self.bias[items]
        if self.config.time_bins > 0:
            raw = raw + self.time_bias[items * self.config.time_bins + b['target_bins']]
        stars = torch.arange(1, 6, dtype=features.dtype, device=features.device)
        user_shift = features[rows, :5] @ stars - 3.65
        if self.config.head == 'scalar':
            return raw[:, 0] + user_shift
        return raw + user_shift[:, None] * (stars - 3)[None, :] / 1.2

    def forward_batch(self, b):
        return self.decode(self.encode(b), b['features'], b['target_rows'], b['target_items'], b)


def weeks(d):
    return (d['max_day'] - d['min_day']) // 7 + 1


def capped_lengths(d, c, extra=None):
    n = np.diff(d['train_offsets']).astype(np.int64)
    if c.choice_tokens:
        n = n + np.diff(choices(d)[0])
    if extra is not None:
        n = n + extra
    return np.minimum(n, c.seq_len - 1) + 1


def prefetched(make, items, device, depth=3):
    """Yield (item, batch) with make(item) run in ONE background thread, in
    order (so RNG consumption is unchanged), on CPU; tensors are moved to the
    device in the caller's thread. Overlaps numpy batch building with the GPU."""
    channel, stop = queue.Queue(depth), threading.Event()

    def put(value):
        while not stop.is_set():
            try:
                channel.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def work():
        try:
            for item in items:
                if stop.is_set():
                    return
                put((item, make(item)))
        except BaseException as error:  # surfaced in the caller
            put(error)
            return
        put(None)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        while True:
            value = channel.get()
            if value is None:
                return
            if isinstance(value, BaseException):
                raise value
            item, b = value
            yield item, {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
    finally:
        stop.set()
        worker.join()


@torch.no_grad()
def predict(model, d, queries=None):
    """Reads only user/item/day identifiers from queries; labels never enter inputs."""
    queries = d['probe'] if queries is None else queries
    users = queries['user'].astype(np.int64)
    order = np.argsort(users, kind='stable')
    offsets = np.searchsorted(users[order], np.arange(d['users'] + 1))
    result = np.empty(len(queries), dtype=np.float32)
    model.eval()
    # No activations are kept for backward, so prediction batches can be larger.
    c = dataclasses.replace(model.config, batch_tokens=2 * model.config.batch_tokens,
                            batch_users=2 * model.config.batch_users)
    lengths = capped_lengths(d, c, np.diff(offsets))
    host = dataclasses.replace(c, device='cpu')

    def make(group):
        rows, index, _ = gather(offsets, group)
        q = order[index]
        items = queries['item'][q].astype(np.int64) - 1
        return prepare_batch(d, group, host, queries=(rows, items, queries['day'][q]))

    for group, b in prefetched(make, length_batches(lengths, np.unique(users), c), c.device):
        q = order[gather(offsets, group)[1]]
        result[q] = model.expected(model.forward_batch(b)).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite predictions')
    return result


def memory():
    if torch.backends.mps.is_available():
        return dict(mps_allocated_mb=torch.mps.current_allocated_memory() / 2**20,
                    mps_driver_mb=torch.mps.driver_allocated_memory() / 2**20)
    return {}


def fit(d, c, epochs, prefix, select=False, patience=3, max_batches=0, save_epochs=False):
    torch.manual_seed(c.seed)
    model = SequenceModel(d['items'], c, weeks(d)).to(c.device)
    model.initialize_bias(d)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    counts = np.diff(d['train_offsets']).astype(np.int64)
    active = np.flatnonzero(counts > 0)
    lengths = capped_lengths(d, c)
    repeats = np.minimum(np.maximum(-(-counts[active] // c.seq_len), 1), max(c.windows, 1))
    samples = np.repeat(active, repeats)
    records, best, best_epoch = [], float('inf'), 0
    selected_path = pathlib.Path(str(prefix) + '.pt')
    host = dataclasses.replace(c, device='cpu')
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        rng = np.random.default_rng(c.seed + epoch)
        model.train()
        losses, sse, targets, examples, tokens = 0., 0., 0, 0, 0
        groups = length_batches(lengths, rng.permutation(samples), c, rng)
        stream = prefetched(lambda group: prepare_batch(d, group, host, rng=rng), groups, c.device)
        for batch_no, (group, b) in enumerate(stream, 1):
            if b['target_ratings'].numel() == 0:
                continue
            raw = model.forward_batch(b)
            loss = loss_value(model, raw, b['target_ratings'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            count = b['target_ratings'].numel()
            with torch.no_grad():
                sse += float(((model.expected(raw) - b['target_ratings'] - 1) ** 2).sum())
            losses += float(loss.detach()) * count
            targets += count
            examples += len(group)
            tokens += int(b['seq_lengths'].sum())
            if batch_no % 100 == 0 or (max_batches and batch_no == max_batches):
                print(json.dumps(dict(phase=str(prefix), epoch=epoch, batch=batch_no, users=examples,
                                      targets=targets, tokens=tokens, train_rmse=(sse / targets)**.5,
                                      seconds=time.monotonic() - start, **memory())), flush=True)
            if max_batches and batch_no >= max_batches:
                break
        stream.close()  # stop the prefetch worker after --max-batches
        rec = dict(epoch=epoch, loss=losses / max(targets, 1), train_rmse=(sse / max(targets, 1))**.5,
                   users=examples, targets=targets, tokens=tokens, train_seconds=time.monotonic() - start)
        if select:
            pred = predict(model, d)
            score = float(np.sqrt(np.mean((np.clip(pred.astype(np.float64), 1, 5) - d['probe']['rating'])**2)))
            rec['validation_rmse'] = score
        else:
            score = 0.
            if save_epochs:
                # Selection never reads these; the holdout-selected epoch's
                # column is picked afterwards, so refit can run concurrently.
                write_f32(str(prefix) + f'.e{epoch}.f32', predict(model, d))
        if score < best or not select:
            best, best_epoch = score, epoch
            temporary = pathlib.Path(str(selected_path) + '.tmp')
            torch.save(dict(config=dataclasses.asdict(c), items=d['items'], weeks=weeks(d), epoch=epoch,
                            state=model.state_dict()), temporary)
            os.replace(temporary, selected_path)
        rec['seconds'] = time.monotonic() - start
        records.append(rec)
        atomic_json(str(prefix) + '.progress.json', dict(records=records, selected_epoch=best_epoch,
                                                       config=dataclasses.asdict(c)))
        print(json.dumps(dict(phase=str(prefix), **rec, selected_epoch=best_epoch)), flush=True)
        if select and epoch - best_epoch >= patience:
            break
    saved = torch.load(selected_path, map_location=c.device, weights_only=True)
    model.load_state_dict(saved['state'])
    return model, records, best_epoch


def run(a):
    if a.threads < 1 or a.epochs < 1:
        raise ValueError('epochs and threads must be positive')
    torch.set_num_threads(a.threads)
    c = Config(**{field.name: getattr(a, field.name) for field in dataclasses.fields(Config)})
    if c.head == 'scalar' and c.loss != 'mse':
        raise ValueError('scalar head requires MSE')
    if c.representation != 'categorical' or c.objective != 'masked':
        raise ValueError('the sequence encoder supports categorical masked training only')
    if c.context != 'session' and (c.time_bins > 0 or c.week_days > 1):
        raise ValueError('--time-bins and --week-days need --context session')
    if not 0 < c.input_drop < 1 or min(c.rank, c.hidden, c.batch_users, c.nnz_budget, c.d_model,
                                       c.layers, c.heads, c.batch_tokens, c.max_targets,
                                       c.pad_width, c.pad_rows) < 1:
        raise ValueError('invalid model dimensions or input drop')
    if not c.max_targets < c.seq_len - 1 or not 0 <= c.recent_frac <= 1 or c.batch_tokens < c.seq_len:
        raise ValueError('need max_targets < seq_len - 1 <= batch_tokens and recent_frac in [0, 1]')
    output = pathlib.Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = pathlib.Path(str(output) + '.json')
    if any(pathlib.Path(str(output) + suffix).exists() for suffix in ('.json', '.pt', '.select.pt', '.refit.pt', '.f32')):
        raise FileExistsError('choose a new output prefix')
    if a.selection_data and pathlib.Path(a.selection_data).resolve() == pathlib.Path(a.data).resolve():
        raise ValueError('selection data must be a separate training-only holdout')
    start = time.monotonic()
    here = pathlib.Path(__file__)
    report = dict(config=dataclasses.asdict(c), arguments=vars(a), torch_version=str(torch.__version__),
                  numpy_version=np.__version__, source_sha256=digest(__file__),
                  base_source_sha256=digest(here.with_name('masked_history.py')),
                  loader_sha256=digest(here.with_name('gbdt_blend.py')),
                  data_sha256=digest(a.data), metric='Original probe development RMSE; no independent test claim')
    atomic_json(report_path, report)
    epochs = a.epochs
    if a.selection_data:
        selection = load(a.selection_data)
        report['selection_sha256'] = digest(a.selection_data)
        _, records, epochs = fit(selection, c, a.epochs, str(output) + '.select', select=True,
                                 patience=a.patience, max_batches=a.max_batches)
        report['selection'] = records
        report['selected_epoch'] = epochs
        atomic_json(report_path, report)
        if a.selection_only:
            report['elapsed_seconds'] = time.monotonic() - start
            atomic_json(report_path, report)
            return report
    elif a.selection_only:
        raise ValueError('--selection-only requires --selection-data')
    d = load(a.data)
    model, records, _ = fit(d, c, epochs, str(output) + '.refit', max_batches=a.max_batches,
                            save_epochs=a.save_epochs)
    predict_start = time.monotonic()
    pred = predict(model, d)
    pred_path = pathlib.Path(str(output) + '.f32')
    write_f32(pred_path, pred)
    report.update(refit=records, selected_epoch=epochs, predictions_sha256=digest(pred_path),
                  checkpoint_sha256=digest(str(output) + '.refit.pt'),
                  probe_rmse=float(np.sqrt(np.mean((np.clip(pred.astype(np.float64), 1, 5) - d['probe']['rating'])**2))),
                  predict_seconds=time.monotonic() - predict_start, elapsed_seconds=time.monotonic() - start,
                  **memory())
    atomic_json(report_path, report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--selection-data')
    p.add_argument('--selection-only', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--save-epochs', action='store_true',
                   help='write each refit epoch\'s probe predictions (choose later by holdout selection)')
    p.add_argument('--max-batches', type=int, default=0, help='throughput/smoke pilot only; 0 means complete epochs')
    choices = dict(representation=['categorical'], head=['scalar', 'distribution'],
                   loss=['mse', 'ce', 'ordinal'], objective=['masked'], context=['none', 'session'])
    for field in dataclasses.fields(Config):
        p.add_argument('--' + field.name.replace('_', '-'), default=field.default,
                       type=type(field.default), **({'choices': choices[field.name]} if field.name in choices else {}))
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    print(json.dumps({k: result[k] for k in ('selected_epoch', 'probe_rmse', 'elapsed_seconds') if k in result}), flush=True)
