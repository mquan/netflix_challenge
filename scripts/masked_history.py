#!/usr/bin/env python3
"""Sparse masked-history rating prediction with training-only epoch selection.

The selection dataset's probe must be a holdout of ORIGINAL TRAINING ratings.
After selection, refit from the same initialization on --data at the selected
epoch. Original probe values are used only for the final reported score.
This is a masked-history adaptation inspired by CF-NADE, not its reproduction.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gbdt_blend import load


@dataclasses.dataclass
class Config:
    representation: str = 'categorical'
    head: str = 'distribution'
    loss: str = 'ce'
    objective: str = 'masked'
    rank: int = 64
    hidden: int = 256
    dropout: float = .1
    input_drop: float = .5
    lr: float = .001
    weight_decay: float = .001
    batch_users: int = 512
    nnz_budget: int = 131072
    seed: int = 20260928
    device: str = 'cpu'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = pathlib.Path(path)
    tmp = pathlib.Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def gather(offsets, users):
    counts = (offsets[users + 1] - offsets[users]).astype(np.int64)
    starts = np.r_[0, np.cumsum(counts)]
    rows = np.repeat(np.arange(len(users)), counts)
    index = np.arange(starts[-1]) + np.repeat(offsets[users].astype(np.int64) - starts[:-1], counts)
    return rows, index, starts


def batches(d, users, c):
    counts = np.diff(d['train_offsets']).astype(np.int64)
    group, size = [], 0
    for user in users:
        n = counts[user]
        if group and (size + n > c.nnz_budget or len(group) == c.batch_users):
            yield np.array(group, dtype=np.int64)
            group, size = [], 0
        group.append(user)
        size += n
    if group:
        yield np.array(group, dtype=np.int64)


def prepare(d, users, c, rng=None, keep=None):
    """Mask values before constructing all explicit inputs and summary features."""
    rows, index, offsets = gather(d['train_offsets'], users)
    items = d['train_items'][index].astype(np.int64) - 1
    ratings = d['train_ratings'][index].astype(np.int64) - 1
    if keep is None:
        keep = np.ones(len(index), dtype=bool) if rng is None else rng.random(len(index)) >= c.input_drop
        if rng is not None:
            # Every nonempty user supplies a target. Keep a context rating when
            # possible, without making any decision using its rating value.
            counts = np.diff(offsets)
            visible = np.bincount(rows, keep.astype(np.int64), minlength=len(users))
            hide = np.flatnonzero((counts > 0) & (visible == counts))
            keep[offsets[hide + 1] - 1] = False
            reveal = np.flatnonzero((counts > 1) & (visible == 0))
            keep[offsets[reveal]] = True
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != index.shape:
        raise ValueError('mask shape mismatch')
    context_rows = rows[keep]
    context_counts = np.bincount(context_rows, minlength=len(users))
    context_offsets = np.r_[0, np.cumsum(context_counts)]
    hist = np.bincount(context_rows * 5 + ratings[keep], minlength=len(users) * 5).reshape(-1, 5)
    # A fixed population-shaped prior is identical in all arms; no target
    # rating, dataset probe value, or full-user rating summary is an input.
    prior = np.array([.05, .10, .25, .35, .25], dtype=np.float32)
    hist = (hist + 5 * prior) / (context_counts[:, None] + 5)
    features = np.column_stack([hist, np.log1p(context_counts) / 8]).astype(np.float32)
    implicit_rows, implicit_index, implicit_offsets = gather(d['implicit_offsets'], users)
    implicit_items = d['implicit_items'][implicit_index].astype(np.int64) - 1
    values = ratings[keep].astype(np.float32) + 1 - 3.6
    explicit_index = items[keep] if c.representation == 'scalar' else items[keep] * 5 + ratings[keep]
    targets = ~keep if c.objective == 'masked' else np.ones(len(keep), dtype=bool)
    data = dict(explicit_index=explicit_index, explicit_offsets=context_offsets,
                explicit_values=values, implicit_index=implicit_items,
                implicit_offsets=implicit_offsets, features=features,
                target_rows=rows[targets], target_items=items[targets],
                target_ratings=ratings[targets], counts=context_counts.astype(np.float32),
                implicit_counts=np.diff(implicit_offsets).astype(np.float32))
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(c.device) for k, v in data.items()}


class HistoryModel(nn.Module):
    def __init__(self, items, c):
        super().__init__()
        self.config = c
        n_input = items * (5 if c.representation == 'categorical' else 1)
        self.explicit = nn.Parameter(torch.empty(n_input, c.rank))
        self.implicit = nn.Parameter(torch.empty(items, c.rank))
        classes = 5 if c.head == 'distribution' else 1
        self.output = nn.Parameter(torch.empty(items, classes, c.rank))
        self.bias = nn.Parameter(torch.zeros(items, classes))
        self.encoder = nn.Sequential(nn.Linear(c.rank * 2 + 6, c.hidden), nn.Tanh(),
                                     nn.Dropout(c.dropout), nn.Linear(c.hidden, c.rank), nn.Tanh())
        for p in (self.explicit, self.implicit, self.output):
            nn.init.normal_(p, std=.04)

    def initialize_bias(self, d):
        counts = np.zeros((d['items'], 5), dtype=np.float64)
        for start in range(0, len(d['train_ratings']), 1_000_000):
            i = d['train_items'][start:start + 1_000_000].astype(np.int64) - 1
            r = d['train_ratings'][start:start + 1_000_000].astype(np.int64) - 1
            counts += np.bincount(i * 5 + r, minlength=d['items'] * 5).reshape(-1, 5)
        prior = (counts.sum(0) + 1) / (counts.sum() + 5)
        probabilities = (counts + 25 * prior) / (counts.sum(1, keepdims=True) + 25)
        value = np.log(probabilities) if self.config.head == 'distribution' else probabilities @ np.arange(1, 6)[:, None]
        with torch.no_grad():
            self.bias.copy_(torch.tensor(value, dtype=torch.float32, device=self.bias.device))

    def encode(self, b):
        weighted = b['explicit_values'] if self.config.representation == 'scalar' else None
        x = F.embedding_bag(b['explicit_index'], self.explicit, b['explicit_offsets'],
                            mode='sum', per_sample_weights=weighted, include_last_offset=True)
        z = F.embedding_bag(b['implicit_index'], self.implicit, b['implicit_offsets'],
                            mode='sum', include_last_offset=True)
        x = x * b['counts'].clamp(min=1).rsqrt()[:, None]
        z = z * b['implicit_counts'].clamp(min=1).rsqrt()[:, None]
        return self.encoder(torch.cat([x, z, b['features']], 1))

    def decode(self, code, features, rows, items):
        raw = (self.output[items] * code[rows, None, :]).sum(2) + self.bias[items]
        stars = torch.arange(1, 6, dtype=features.dtype, device=features.device)
        user_shift = features[rows, :5] @ stars - 3.65
        if self.config.head == 'scalar':
            return raw[:, 0] + user_shift
        return raw + user_shift[:, None] * (stars - 3)[None, :] / 1.2

    def expected(self, raw):
        if self.config.head == 'scalar':
            return raw
        return raw.softmax(1) @ torch.arange(1, 6, dtype=raw.dtype, device=raw.device)


def loss_value(model, raw, ratings):
    target = ratings.to(torch.float32) + 1
    if model.config.loss == 'mse':
        return F.mse_loss(model.expected(raw), target)
    if model.config.loss == 'ce':
        return F.cross_entropy(raw, ratings)
    # Ordered cumulative log loss, an explicit adaptation rather than the
    # listwise ordinal objective in the CF-NADE paper.
    cdf = raw.softmax(1).cumsum(1)[:, :4].clamp(1e-6, 1 - 1e-6)
    binary = (ratings[:, None] <= torch.arange(4, device=raw.device)).to(raw.dtype)
    return F.binary_cross_entropy(cdf, binary)


@torch.no_grad()
def predict(model, d, queries=None):
    """Reads only user/item identifiers from queries; labels never enter inputs."""
    queries = d['probe'] if queries is None else queries
    users = queries['user'].astype(np.int64)
    order = np.argsort(users, kind='stable')
    offsets = np.searchsorted(users[order], np.arange(d['users'] + 1))
    result = np.empty(len(queries), dtype=np.float32)
    model.eval()
    for group in batches(d, np.unique(users), model.config):
        b = prepare(d, group, model.config)
        code = model.encode(b)
        rows, index, _ = gather(offsets, group)
        q = order[index]
        items = queries['item'][q].astype(np.int64) - 1
        raw = model.decode(code, b['features'], torch.from_numpy(rows).to(model.config.device),
                           torch.from_numpy(items).to(model.config.device))
        result[q] = model.expected(raw).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite predictions')
    return result


def fit(d, c, epochs, prefix, select=False, patience=3, max_batches=0):
    torch.manual_seed(c.seed)
    model = HistoryModel(d['items'], c).to(c.device)
    model.initialize_bias(d)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    active = np.flatnonzero(np.diff(d['train_offsets']) > 0)
    records, best, best_epoch = [], float('inf'), 0
    selected_path = pathlib.Path(str(prefix) + '.pt')
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        rng = np.random.default_rng(c.seed + epoch)
        group_order = rng.permutation(active)
        model.train()
        losses, sse, targets, examples = 0., 0., 0, 0
        for batch_no, group in enumerate(batches(d, group_order, c), 1):
            b = prepare(d, group, c, rng=rng)
            if b['target_ratings'].numel() == 0:
                continue
            raw = model.decode(model.encode(b), b['features'], b['target_rows'], b['target_items'])
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
            if batch_no % 100 == 0 or (max_batches and batch_no == max_batches):
                print(json.dumps(dict(phase=str(prefix), epoch=epoch, batch=batch_no,
                                      users=examples, targets=targets, train_rmse=(sse / targets)**.5,
                                      seconds=time.monotonic() - start)), flush=True)
            if max_batches and batch_no >= max_batches:
                break
        rec = dict(epoch=epoch, loss=losses / max(targets, 1), train_rmse=(sse / max(targets, 1))**.5,
                   users=examples, targets=targets, train_seconds=time.monotonic() - start)
        if select:
            pred = predict(model, d)
            score = float(np.sqrt(np.mean((np.clip(pred.astype(np.float64), 1, 5) - d['probe']['rating'])**2)))
            rec['validation_rmse'] = score
        else:
            score = 0.
        if score < best or not select:
            best, best_epoch = score, epoch
            temporary = pathlib.Path(str(selected_path) + '.tmp')
            torch.save(dict(config=dataclasses.asdict(c), items=d['items'], epoch=epoch,
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
    if not 0 < c.input_drop < 1 or min(c.rank, c.hidden, c.batch_users, c.nnz_budget) < 1:
        raise ValueError('invalid model dimensions or input drop')
    output = pathlib.Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = pathlib.Path(str(output) + '.json')
    if any(pathlib.Path(str(output) + suffix).exists() for suffix in ('.json', '.pt', '.select.pt', '.refit.pt', '.f32')):
        raise FileExistsError('choose a new output prefix')
    if a.selection_data and pathlib.Path(a.selection_data).resolve() == pathlib.Path(a.data).resolve():
        raise ValueError('selection data must be a separate training-only holdout')
    start = time.monotonic()
    report = dict(config=dataclasses.asdict(c), arguments=vars(a), torch_version=str(torch.__version__),
                  numpy_version=np.__version__, source_sha256=digest(__file__),
                  loader_sha256=digest(pathlib.Path(__file__).with_name('gbdt_blend.py')),
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
    model, records, _ = fit(d, c, epochs, str(output) + '.refit', max_batches=a.max_batches)
    pred = predict(model, d)
    pred_path = pathlib.Path(str(output) + '.f32')
    tmp = pathlib.Path(str(pred_path) + '.tmp')
    pred.astype('<f4').tofile(tmp)
    os.replace(tmp, pred_path)
    report.update(refit=records, selected_epoch=epochs, predictions_sha256=digest(pred_path),
                  checkpoint_sha256=digest(str(output) + '.refit.pt'),
                  probe_rmse=float(np.sqrt(np.mean((np.clip(pred.astype(np.float64), 1, 5) - d['probe']['rating'])**2))),
                  elapsed_seconds=time.monotonic() - start)
    atomic_json(report_path, report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--selection-data')
    p.add_argument('--selection-only', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-batches', type=int, default=0, help='throughput/smoke pilot only; 0 means complete epochs')
    choices = dict(representation=['scalar', 'categorical'], head=['scalar', 'distribution'],
                   loss=['mse', 'ce', 'ordinal'], objective=['masked', 'reconstruct'])
    for field in dataclasses.fields(Config):
        p.add_argument('--' + field.name.replace('_', '-'), default=field.default,
                       type=type(field.default), **({'choices': choices[field.name]} if field.name in choices else {}))
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    print(json.dumps({k: result[k] for k in ('selected_epoch', 'probe_rmse', 'elapsed_seconds') if k in result}), flush=True)
