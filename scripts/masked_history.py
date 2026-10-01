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
    # Opt-in temporal extensions; defaults reproduce the round-7 pilot exactly.
    context: str = 'none'
    suffix_prob: float = 0.
    time_bins: int = 0
    week_days: int = 0


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


def suffix_mask(keep, rows, days, offsets, users, c, rng):
    """For a random subset of users, hide the latest ratings by date, as the
    probe does. Uses dates and random tie-breaks only, never rating values."""
    chosen = rng.random(len(users)) < c.suffix_prob
    fraction = rng.uniform(.1, .4, len(users))
    order = np.lexsort((rng.random(len(rows)), days, rows))
    counts = np.diff(offsets)
    position = np.empty(len(rows), dtype=np.int64)
    position[order] = np.arange(len(rows)) - offsets[rows[order]]
    late = position >= np.ceil(counts * (1 - fraction))[rows]
    keep = keep.copy()
    sel = chosen[rows]
    keep[sel] = ~late[sel]
    return keep


SESSION_KEY = 65536


def session_channel(d, rows, index, keep, explicit_index, implicit_rows, implicit_index,
                    target_rows, target_days, width, prefix):
    """Context for each target from the user's interactions in the same
    width-day period: implicit movies (own movie removed by the model) and
    visible ratings. No hidden rating value enters."""
    imp_days = d['implicit_days'][implicit_index].astype(np.int64)
    imp_key = implicit_rows * SESSION_KEY + imp_days // width
    imp_order = np.argsort(imp_key, kind='stable')
    imp_unique, imp_start = np.unique(imp_key[imp_order], return_index=True)
    ctx_key = rows[keep] * SESSION_KEY + d['train_days'][index][keep].astype(np.int64) // width
    ctx_order = np.argsort(ctx_key, kind='stable')
    ctx_unique, ctx_start = np.unique(ctx_key[ctx_order], return_index=True)
    target_key = target_rows * SESSION_KEY + target_days // width

    def lookup(unique, key):
        if not len(unique):
            return np.zeros(len(key), dtype=np.int64)
        pos = np.minimum(np.searchsorted(unique, key), len(unique) - 1)
        return np.where(unique[pos] == key, pos, len(unique))

    imp_session = lookup(imp_unique, target_key)
    ctx_session = lookup(ctx_unique, target_key)
    imp_counts = np.r_[np.diff(np.r_[imp_start, len(imp_key)]), 0]
    ctx_counts = np.r_[np.diff(np.r_[ctx_start, len(ctx_key)]), 0]
    n_imp = np.maximum(imp_counts[imp_session] - 1, 0)
    n_ctx = ctx_counts[ctx_session]
    out = {prefix + '_imp_index': d['implicit_items'][implicit_index][imp_order].astype(np.int64) - 1,
           prefix + '_imp_offsets': np.r_[imp_start, len(imp_key)],
           prefix + '_exp_index': explicit_index[ctx_order],
           prefix + '_exp_offsets': np.r_[ctx_start, len(ctx_key)],
           prefix + '_imp_session': imp_session, prefix + '_exp_session': ctx_session,
           prefix + '_imp_present': (imp_session < len(imp_unique)).astype(np.float32),
           prefix + '_imp_counts': n_imp.astype(np.float32),
           prefix + '_exp_counts': n_ctx.astype(np.float32)}
    return out, n_imp, n_ctx


def session_inputs(d, rows, index, keep, explicit_index, implicit_rows, implicit_index,
                   target_rows, target_days, users, week_days=0):
    """Same-day (and optionally week_days-period) context plus date features."""
    args = (d, rows, index, keep, explicit_index, implicit_rows, implicit_index, target_rows, target_days)
    out, n_imp, n_ctx = session_channel(*args, 1, 'day')
    columns = [np.log1p(n_imp) / 5, np.log1p(n_ctx) / 4]
    if week_days > 1:
        week, w_imp, w_ctx = session_channel(*args, week_days, 'week')
        out.update(week)
        columns += [np.log1p(w_imp) / 6, np.log1p(w_ctx) / 5]
    imp_days = d['implicit_days'][implicit_index].astype(np.int64)
    first = np.full(len(users), d['max_day'], dtype=np.int64)
    last = np.full(len(users), d['min_day'], dtype=np.int64)
    np.minimum.at(first, implicit_rows, imp_days)
    np.maximum.at(last, implicit_rows, imp_days)
    span = max(d['max_day'] - d['min_day'], 1)
    columns += [(target_days - first[target_rows]) / span,
                (last[target_rows] - target_days) / span,
                (target_days - d['min_day']) / span]
    out.update(target_features=np.column_stack(columns).astype(np.float32),
               target_bins=((target_days - d['min_day']) * max(d.get('time_bins', 1), 1) // (span + 1)))
    return out


def prepare(d, users, c, rng=None, keep=None, queries=None):
    """Mask values before constructing all explicit inputs and summary features.

    queries, when given, is (rows, items, days) for prediction targets."""
    rows, index, offsets = gather(d['train_offsets'], users)
    items = d['train_items'][index].astype(np.int64) - 1
    ratings = d['train_ratings'][index].astype(np.int64) - 1
    if keep is None:
        keep = np.ones(len(index), dtype=bool) if rng is None else rng.random(len(index)) >= c.input_drop
        if rng is not None and c.suffix_prob > 0:
            keep = suffix_mask(keep, rows, d['train_days'][index].astype(np.int64), offsets, users, c, rng)
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
    if queries is None:
        target_rows, target_items = rows[targets], items[targets]
        target_ratings = ratings[targets]
        target_days = d['train_days'][index][targets].astype(np.int64)
    else:
        target_rows, target_items, target_days = (np.asarray(q, dtype=np.int64) for q in queries)
        target_ratings = np.zeros(0, dtype=np.int64)
    data = dict(explicit_index=explicit_index, explicit_offsets=context_offsets,
                explicit_values=values, implicit_index=implicit_items,
                implicit_offsets=implicit_offsets, features=features,
                target_rows=target_rows, target_items=target_items,
                target_ratings=target_ratings, counts=context_counts.astype(np.float32),
                implicit_counts=np.diff(implicit_offsets).astype(np.float32))
    if c.context == 'session':
        visible = (items * 5 + ratings)[keep]
        data.update(session_inputs(dict(d, time_bins=c.time_bins), rows, index, keep, visible,
                                   implicit_rows, implicit_index, target_rows, target_days, users,
                                   c.week_days))
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
        if c.context == 'session':
            self.session_implicit = nn.Parameter(torch.empty(items, c.rank))
            self.session_explicit = nn.Parameter(torch.empty(items * 5, c.rank))
            channels, n_features = 1, 5
            if c.week_days > 1:
                self.week_implicit = nn.Parameter(torch.empty(items, c.rank))
                self.week_explicit = nn.Parameter(torch.empty(items * 5, c.rank))
                channels, n_features = 2, 7
            for name in ('session_implicit', 'session_explicit', 'week_implicit', 'week_explicit'):
                if hasattr(self, name):
                    nn.init.normal_(getattr(self, name), std=.04)
            self.session_net = nn.Sequential(nn.Linear(c.rank * (1 + 2 * channels) + n_features, c.hidden), nn.Tanh(),
                                             nn.Dropout(c.dropout), nn.Linear(c.hidden, c.rank))
            nn.init.zeros_(self.session_net[-1].weight)
            nn.init.zeros_(self.session_net[-1].bias)
        if c.time_bins > 0:
            self.time_bias = nn.Parameter(torch.zeros(items * c.time_bins, classes))

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

    def channel(self, b, prefix, implicit, explicit, items, zero):
        imp = F.embedding_bag(b[prefix + '_imp_index'], implicit, b[prefix + '_imp_offsets'],
                              mode='sum', include_last_offset=True)
        imp = torch.cat([imp, zero])[b[prefix + '_imp_session']]
        imp = imp - implicit[items] * b[prefix + '_imp_present'][:, None]
        imp = imp * b[prefix + '_imp_counts'].clamp(min=1).rsqrt()[:, None]
        # Period ratings come from visible context only.
        exp = F.embedding_bag(b[prefix + '_exp_index'], explicit, b[prefix + '_exp_offsets'],
                              mode='sum', include_last_offset=True)
        exp = torch.cat([exp, zero])[b[prefix + '_exp_session']]
        exp = exp * b[prefix + '_exp_counts'].clamp(min=1).rsqrt()[:, None]
        return [imp, exp]

    def target_code(self, code, b):
        """Shift each target's user code by its same-day (and period) context."""
        rows, items = b['target_rows'], b['target_items']
        zero = torch.zeros(1, self.config.rank, dtype=code.dtype, device=code.device)
        parts = [code[rows]] + self.channel(b, 'day', self.session_implicit, self.session_explicit, items, zero)
        if self.config.week_days > 1:
            parts += self.channel(b, 'week', self.week_implicit, self.week_explicit, items, zero)
        shift = self.session_net(torch.cat(parts + [b['target_features']], 1))
        return code[rows] + shift

    def decode(self, code, features, rows, items, b=None):
        if b is not None and self.config.context == 'session':
            vector = self.target_code(code, b)
        else:
            vector = code[rows]
        raw = (self.output[items] * vector[:, None, :]).sum(2) + self.bias[items]
        if b is not None and self.config.time_bins > 0:
            raw = raw + self.time_bias[items * self.config.time_bins + b['target_bins']]
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
        rows, index, _ = gather(offsets, group)
        q = order[index]
        items = queries['item'][q].astype(np.int64) - 1
        b = prepare(d, group, model.config, queries=(rows, items, queries['day'][q]))
        code = model.encode(b)
        raw = model.decode(code, b['features'], b['target_rows'], b['target_items'], b)
        result[q] = model.expected(raw).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite predictions')
    return result


def fit(d, c, epochs, prefix, select=False, patience=3, max_batches=0, save_epochs=False):
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
            raw = model.decode(model.encode(b), b['features'], b['target_rows'], b['target_items'], b)
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
            if save_epochs:
                # Selection never reads these; the holdout-selected epoch's
                # column is picked afterwards, so refit can run concurrently.
                write_f32(str(prefix) + f'.e{epoch}.f32', predict(model, d))
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


def write_f32(path, values):
    tmp = pathlib.Path(str(path) + '.tmp')
    values.astype('<f4').tofile(tmp)
    os.replace(tmp, path)


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
    model, records, _ = fit(d, c, epochs, str(output) + '.refit', max_batches=a.max_batches,
                            save_epochs=a.save_epochs)
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
    p.add_argument('--save-epochs', action='store_true',
                   help='write each refit epoch\'s probe predictions (choose later by holdout selection)')
    p.add_argument('--max-batches', type=int, default=0, help='throughput/smoke pilot only; 0 means complete epochs')
    choices = dict(representation=['scalar', 'categorical'], head=['scalar', 'distribution'],
                   loss=['mse', 'ce', 'ordinal'], objective=['masked', 'reconstruct'],
                   context=['none', 'session'])
    for field in dataclasses.fields(Config):
        p.add_argument('--' + field.name.replace('_', '-'), default=field.default,
                       type=type(field.default), **({'choices': choices[field.name]} if field.name in choices else {}))
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    print(json.dumps({k: result[k] for k in ('selected_epoch', 'probe_rmse', 'elapsed_seconds') if k in result}), flush=True)
