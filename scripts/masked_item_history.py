#!/usr/bin/env python3
"""Item-side twin of masked_history.py: predict masked ratings of a movie from
its other dated raters.

Each training visit takes a batch of MOVIES, splits every movie's training
ratings into visible context and hidden targets, and encodes the movie from
its visible raters (who rated it, how, and when), the users who interacted
with it (the implicit channel), and, with --context session, the movie's
raters on the target date. The target user enters only through a learned
output vector and rating-distribution bias; no hidden rating value is an input.

Protocol is identical to masked_history.py: select the epoch on a training-only
holdout (--selection-data), refit on --data from the same initialization, and
use the original probe only for the final reported score. --save-epochs lets
the refit run concurrently with selection.

Parameter budget: the encoded axis is the movie, so every per-entity table is
indexed by user. To keep the dense optimizer state at three users x rank
tables, a rater's rating r is encoded as rater[u] * scale[r] + shift[r]
instead of a separate users x 5 x rank table, and the 5-class head is a
rank -> 5*rank projection of the movie code dotted with the user's output
vector.
"""
import argparse
import dataclasses
import json
import os
import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gbdt_blend import load
from masked_history import atomic_json, digest, gather, loss_value, session_inputs, suffix_mask, write_f32


@dataclasses.dataclass
class ItemConfig:
    head: str = 'distribution'
    loss: str = 'ce'
    objective: str = 'masked'
    rank: int = 64
    hidden: int = 256
    dropout: float = .1
    input_drop: float = .5
    lr: float = .001
    weight_decay: float = .001
    batch_items: int = 512
    nnz_budget: int = 131072
    seed: int = 20260928
    device: str = 'cpu'
    context: str = 'none'
    suffix_prob: float = 0.
    time_bins: int = 0


def transpose(d):
    """Movie-major view of a loaded dataset.

    Keys keep the user-major names so masked_history's gather, suffix and
    session helpers apply unchanged: offsets index movies and the
    '*_items' arrays hold user id + 1. Within a movie, entries keep the
    source (user-major) order."""
    t = {k: d[k] for k in ('users', 'items', 'mean', 'min_day', 'max_day', 'probe')}
    for prefix in ('train', 'implicit'):
        offsets = np.asarray(d[prefix + '_offsets'], dtype=np.int64)
        items = np.asarray(d[prefix + '_items'])
        order = np.argsort(items, kind='stable')
        owner = np.repeat(np.arange(1, d['users'] + 1, dtype=np.int32), np.diff(offsets))
        t[prefix + '_items'] = owner[order]
        del owner
        t[prefix + '_days'] = np.asarray(d[prefix + '_days'])[order]
        if prefix == 'train':
            t['train_ratings'] = np.asarray(d['train_ratings'])[order]
        del order
        counts = np.bincount(items.astype(np.int64) - 1, minlength=d['items'])
        t[prefix + '_offsets'] = np.r_[0, np.cumsum(counts)].astype(np.int64)
    return t


def batches(t, movies, c):
    counts = np.diff(t['train_offsets']).astype(np.int64)
    group, size = [], 0
    for movie in movies:
        n = counts[movie]
        if group and (size + n > c.nnz_budget or len(group) == c.batch_items):
            yield np.array(group, dtype=np.int64)
            group, size = [], 0
        group.append(movie)
        size += n
    if group:
        yield np.array(group, dtype=np.int64)


def prepare(t, movies, c, rng=None, keep=None, queries=None):
    """Mask values before constructing every input. t is a transpose() view.

    queries, when given, is (rows, users, days) for prediction targets."""
    rows, index, offsets = gather(t['train_offsets'], movies)
    raters = t['train_items'][index].astype(np.int64) - 1
    ratings = t['train_ratings'][index].astype(np.int64) - 1
    days = t['train_days'][index].astype(np.int64)
    if keep is None:
        keep = np.ones(len(index), dtype=bool) if rng is None else rng.random(len(index)) >= c.input_drop
        if rng is not None and c.suffix_prob > 0:
            # For a random subset of movies hide their latest raters by date.
            keep = suffix_mask(keep, rows, days, offsets, movies, c, rng)
        if rng is not None:
            counts = np.diff(offsets)
            visible = np.bincount(rows, keep.astype(np.int64), minlength=len(movies))
            hide = np.flatnonzero((counts > 0) & (visible == counts))
            keep[offsets[hide + 1] - 1] = False
            reveal = np.flatnonzero((counts > 1) & (visible == 0))
            keep[offsets[reveal]] = True
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != index.shape:
        raise ValueError('mask shape mismatch')
    context_rows = rows[keep]
    context_counts = np.bincount(context_rows, minlength=len(movies))
    context_offsets = np.r_[0, np.cumsum(context_counts)]
    hist = np.bincount(context_rows * 5 + ratings[keep], minlength=len(movies) * 5).reshape(-1, 5)
    prior = np.array([.05, .10, .25, .35, .25], dtype=np.float32)
    hist = (hist + 5 * prior) / (context_counts[:, None] + 5)
    features = np.column_stack([hist, np.log1p(context_counts) / 8]).astype(np.float32)
    implicit_rows, implicit_index, implicit_offsets = gather(t['implicit_offsets'], movies)
    implicit_users = t['implicit_items'][implicit_index].astype(np.int64) - 1
    explicit_index = raters[keep] * 5 + ratings[keep]
    targets = ~keep if c.objective == 'masked' else np.ones(len(keep), dtype=bool)
    if queries is None:
        target_rows, target_users = rows[targets], raters[targets]
        target_ratings = ratings[targets]
        target_days = days[targets]
    else:
        target_rows, target_users, target_days = (np.asarray(q, dtype=np.int64) for q in queries)
        target_ratings = np.zeros(0, dtype=np.int64)
    data = dict(explicit_index=explicit_index, explicit_offsets=context_offsets,
                implicit_index=implicit_users, implicit_offsets=implicit_offsets, features=features,
                target_rows=target_rows, target_users=target_users,
                target_items=np.asarray(movies, dtype=np.int64)[target_rows],
                target_ratings=target_ratings, counts=context_counts.astype(np.float32),
                implicit_counts=np.diff(implicit_offsets).astype(np.float32))
    if c.context == 'session':
        # The movie's raters on the target date: implicit (target user removed
        # by the model) and visible ratings, plus movie-age date features.
        data.update(session_inputs(dict(t, time_bins=c.time_bins), rows, index, keep, explicit_index,
                                   implicit_rows, implicit_index, target_rows, target_days, movies))
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(c.device) for k, v in data.items()}


def segment_sum(values, offsets):
    counts = offsets[1:] - offsets[:-1]
    ids = torch.repeat_interleave(torch.arange(len(counts), device=values.device), counts)
    out = torch.zeros(len(counts), values.shape[1], dtype=values.dtype, device=values.device)
    return out.index_add(0, ids, values)


class ItemHistoryModel(nn.Module):
    def __init__(self, users, items, c):
        super().__init__()
        self.config = c
        classes = 5 if c.head == 'distribution' else 1
        self.rater = nn.Parameter(torch.empty(users, c.rank))
        self.rating_scale = nn.Parameter(torch.ones(5, c.rank))
        self.rating_shift = nn.Parameter(torch.empty(5, c.rank))
        self.implicit = nn.Parameter(torch.empty(users, c.rank))
        self.output = nn.Parameter(torch.empty(users, c.rank))
        self.bias = nn.Parameter(torch.zeros(users, classes))
        self.head = nn.Linear(c.rank, classes * c.rank, bias=False)
        self.encoder = nn.Sequential(nn.Linear(c.rank * 2 + 6, c.hidden), nn.Tanh(),
                                     nn.Dropout(c.dropout), nn.Linear(c.hidden, c.rank), nn.Tanh())
        for p in (self.rater, self.rating_shift, self.implicit, self.output):
            nn.init.normal_(p, std=.04)
        if c.context == 'session':
            self.session_scale = nn.Parameter(torch.ones(5, c.rank))
            self.session_shift = nn.Parameter(torch.empty(5, c.rank))
            nn.init.normal_(self.session_shift, std=.04)
            self.session_net = nn.Sequential(nn.Linear(c.rank * 3 + 5, c.hidden), nn.Tanh(),
                                             nn.Dropout(c.dropout), nn.Linear(c.hidden, c.rank))
            nn.init.zeros_(self.session_net[-1].weight)
            nn.init.zeros_(self.session_net[-1].bias)
        if c.time_bins > 0:
            self.time_bias = nn.Parameter(torch.zeros(items * c.time_bins, classes))

    def initialize_bias(self, t):
        """Per-user rating distribution (shrunk to the population)."""
        users = t['train_items'].astype(np.int64) - 1
        counts = np.zeros(self.bias.shape[0] * 5, dtype=np.float64)
        for start in range(0, len(users), 1_000_000):
            r = t['train_ratings'][start:start + 1_000_000].astype(np.int64) - 1
            counts += np.bincount(users[start:start + 1_000_000] * 5 + r, minlength=len(counts))
        counts = counts.reshape(-1, 5)
        prior = (counts.sum(0) + 1) / (counts.sum() + 5)
        probabilities = (counts + 25 * prior) / (counts.sum(1, keepdims=True) + 25)
        value = np.log(probabilities) if self.config.head == 'distribution' else probabilities @ np.arange(1, 6)[:, None]
        with torch.no_grad():
            self.bias.copy_(torch.tensor(value, dtype=torch.float32, device=self.bias.device))

    def rated(self, index, scale, shift):
        """Embedding of (rater, rating) pairs encoded as user * 5 + rating."""
        return self.rater[index // 5] * scale[index % 5] + shift[index % 5]

    def encode(self, b):
        x = segment_sum(self.rated(b['explicit_index'], self.rating_scale, self.rating_shift),
                        b['explicit_offsets'])
        z = F.embedding_bag(b['implicit_index'], self.implicit, b['implicit_offsets'],
                            mode='sum', include_last_offset=True)
        x = x * b['counts'].clamp(min=1).rsqrt()[:, None]
        z = z * b['implicit_counts'].clamp(min=1).rsqrt()[:, None]
        return self.encoder(torch.cat([x, z, b['features']], 1))

    def target_code(self, code, b):
        """Shift each target's movie code by the movie's same-day raters."""
        rows, users = b['target_rows'], b['target_users']
        zero = torch.zeros(1, self.config.rank, dtype=code.dtype, device=code.device)
        imp = F.embedding_bag(b['day_imp_index'], self.implicit, b['day_imp_offsets'],
                              mode='sum', include_last_offset=True)
        imp = torch.cat([imp, zero])[b['day_imp_session']]
        imp = imp - self.implicit[users] * b['day_imp_present'][:, None]
        imp = imp * b['day_imp_counts'].clamp(min=1).rsqrt()[:, None]
        exp = segment_sum(self.rated(b['day_exp_index'], self.session_scale, self.session_shift),
                          b['day_exp_offsets'])
        exp = torch.cat([exp, zero])[b['day_exp_session']]
        exp = exp * b['day_exp_counts'].clamp(min=1).rsqrt()[:, None]
        shift = self.session_net(torch.cat([code[rows], imp, exp, b['target_features']], 1))
        return code[rows] + shift

    def decode(self, code, b):
        rows, users = b['target_rows'], b['target_users']
        vector = self.target_code(code, b) if self.config.context == 'session' else code[rows]
        classes = self.bias.shape[1]
        movie = self.head(vector).view(-1, classes, self.config.rank)
        raw = (movie * self.output[users][:, None, :]).sum(2) + self.bias[users]
        if self.config.time_bins > 0:
            raw = raw + self.time_bias[b['target_items'] * self.config.time_bins + b['target_bins']]
        features = b['features']
        stars = torch.arange(1, 6, dtype=features.dtype, device=features.device)
        movie_shift = features[rows, :5] @ stars - 3.65
        if self.config.head == 'scalar':
            return raw[:, 0] + movie_shift
        return raw + movie_shift[:, None] * (stars - 3)[None, :] / 1.2

    def expected(self, raw):
        if self.config.head == 'scalar':
            return raw
        return raw.softmax(1) @ torch.arange(1, 6, dtype=raw.dtype, device=raw.device)


@torch.no_grad()
def predict(model, t, queries=None):
    """Reads only user/item/day identifiers from queries; labels never enter."""
    queries = t['probe'] if queries is None else queries
    movies = queries['item'].astype(np.int64) - 1
    order = np.argsort(movies, kind='stable')
    offsets = np.searchsorted(movies[order], np.arange(t['items'] + 1))
    result = np.empty(len(queries), dtype=np.float32)
    model.eval()
    for group in batches(t, np.unique(movies), model.config):
        rows, index, _ = gather(offsets, group)
        q = order[index]
        b = prepare(t, group, model.config,
                    queries=(rows, queries['user'][q].astype(np.int64), queries['day'][q]))
        raw = model.decode(model.encode(b), b)
        result[q] = model.expected(raw).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite predictions')
    return result


def fit(t, c, epochs, prefix, select=False, patience=3, max_batches=0, save_epochs=False):
    torch.manual_seed(c.seed)
    model = ItemHistoryModel(t['users'], t['items'], c).to(c.device)
    model.initialize_bias(t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    active = np.flatnonzero(np.diff(t['train_offsets']) > 0)
    records, best, best_epoch = [], float('inf'), 0
    selected_path = pathlib.Path(str(prefix) + '.pt')
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        rng = np.random.default_rng(c.seed + epoch)
        group_order = rng.permutation(active)
        model.train()
        losses, sse, targets, examples = 0., 0., 0, 0
        for batch_no, group in enumerate(batches(t, group_order, c), 1):
            b = prepare(t, group, c, rng=rng)
            if b['target_ratings'].numel() == 0:
                continue
            raw = model.decode(model.encode(b), b)
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
            if batch_no % 100 == 0 or (max_batches and (batch_no == max_batches or batch_no % 25 == 0)):
                print(json.dumps(dict(phase=str(prefix), epoch=epoch, batch=batch_no, movies=examples,
                                      targets=targets, loss=losses / targets, train_rmse=(sse / targets)**.5,
                                      seconds=time.monotonic() - start)), flush=True)
            if max_batches and batch_no >= max_batches:
                break
        rec = dict(epoch=epoch, loss=losses / max(targets, 1), train_rmse=(sse / max(targets, 1))**.5,
                   movies=examples, targets=targets, train_seconds=time.monotonic() - start)
        if select:
            pred = predict(model, t)
            score = float(np.sqrt(np.mean((np.clip(pred.astype(np.float64), 1, 5) - t['probe']['rating'])**2)))
            rec['validation_rmse'] = score
        else:
            score = 0.
            if save_epochs:
                write_f32(str(prefix) + f'.e{epoch}.f32', predict(model, t))
        if score < best or not select:
            best, best_epoch = score, epoch
            temporary = pathlib.Path(str(selected_path) + '.tmp')
            torch.save(dict(config=dataclasses.asdict(c), users=t['users'], items=t['items'], epoch=epoch,
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
    c = ItemConfig(**{field.name: getattr(a, field.name) for field in dataclasses.fields(ItemConfig)})
    if c.head == 'scalar' and c.loss != 'mse':
        raise ValueError('scalar head requires MSE')
    if not 0 < c.input_drop < 1 or min(c.rank, c.hidden, c.batch_items, c.nnz_budget) < 1:
        raise ValueError('invalid model dimensions or input drop')
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
                  numpy_version=np.__version__, source_sha256=digest(here),
                  helpers_sha256=digest(here.with_name('masked_history.py')),
                  loader_sha256=digest(here.with_name('gbdt_blend.py')),
                  data_sha256=digest(a.data), metric='Original probe development RMSE; no independent test claim')
    atomic_json(report_path, report)
    epochs = a.epochs
    if a.selection_data:
        selection = transpose(load(a.selection_data))
        report['selection_sha256'] = digest(a.selection_data)
        _, records, epochs = fit(selection, c, a.epochs, str(output) + '.select', select=True,
                                 patience=a.patience, max_batches=a.max_batches)
        del selection
        report['selection'] = records
        report['selected_epoch'] = epochs
        atomic_json(report_path, report)
        if a.selection_only:
            report['elapsed_seconds'] = time.monotonic() - start
            atomic_json(report_path, report)
            return report
    elif a.selection_only:
        raise ValueError('--selection-only requires --selection-data')
    t = transpose(load(a.data))
    model, records, _ = fit(t, c, epochs, str(output) + '.refit', max_batches=a.max_batches,
                            save_epochs=a.save_epochs)
    pred = predict(model, t)
    pred_path = pathlib.Path(str(output) + '.f32')
    write_f32(pred_path, pred)
    report.update(refit=records, selected_epoch=epochs, predictions_sha256=digest(pred_path),
                  checkpoint_sha256=digest(str(output) + '.refit.pt'),
                  probe_rmse=float(np.sqrt(np.mean((np.clip(pred.astype(np.float64), 1, 5) - t['probe']['rating'])**2))),
                  elapsed_seconds=time.monotonic() - start)
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
    choices = dict(head=['scalar', 'distribution'], loss=['mse', 'ce', 'ordinal'],
                   objective=['masked', 'reconstruct'], context=['none', 'session'])
    for field in dataclasses.fields(ItemConfig):
        p.add_argument('--' + field.name.replace('_', '-'), default=field.default,
                       type=type(field.default), **({'choices': choices[field.name]} if field.name in choices else {}))
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    print(json.dumps({k: result[k] for k in ('selected_epoch', 'probe_rmse', 'elapsed_seconds') if k in result}), flush=True)
