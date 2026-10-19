#!/usr/bin/env python3
"""Matched A2 fine-tuning with an auxiliary observed-movie reconstruction loss.

The auxiliary pass removes its positive movie IDs from every encoder input.
Uniform unobserved movies are negative observation indicators, never ratings.
Both arms run that pass, including weight=0, to match dropout RNG consumption.
This is a balanced discrimination surrogate, not a calibrated propensity model.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import masked_history as mh
import masked_sequence as ms
from gbdt_blend import load


def sample_choices(d, users, rng, positives=8, negatives=32):
    """Uniform unique observed positives; uniform complement negatives with replacement."""
    if min(positives, negatives) < 1:
        raise ValueError('positive sample counts required')
    rows, items, labels, weights = [], [], [], []
    valid_users = 0
    for row, user in enumerate(users):
        begin, end = map(int, d['implicit_offsets'][user:user + 2])
        known = np.unique(d['implicit_items'][begin:end].astype(np.int64) - 1)
        if not len(known) or len(known) == d['items']:
            continue
        if known.min() < 0 or known.max() >= d['items']:
            raise ValueError('invalid implicit movie ID')
        pos = rng.choice(known, size=min(positives, len(known)), replace=False)
        neg = []
        for _ in range(32):
            draw = rng.integers(d['items'], size=max(2 * (negatives - len(neg)), 8))
            neg.extend(draw[~np.isin(draw, known)].tolist())
            if len(neg) >= negatives:
                break
        if len(neg) < negatives:
            complement = np.setdiff1d(np.arange(d['items']), known, assume_unique=True)
            neg.extend(rng.choice(complement, size=negatives - len(neg), replace=True).tolist())
        neg = np.asarray(neg[:negatives], dtype=np.int64)
        rows.extend([row] * (len(pos) + negatives))
        items.extend(np.r_[pos, neg].tolist())
        labels.extend([1.] * len(pos) + [0.] * negatives)
        weights.extend([.5 / len(pos)] * len(pos) + [.5 / negatives] * negatives)
        valid_users += 1
    return dict(rows=np.asarray(rows, dtype=np.int64), items=np.asarray(items, dtype=np.int64),
                labels=np.asarray(labels, dtype=np.float32),
                weights=np.asarray(weights, dtype=np.float32) / max(valid_users, 1))


def auxiliary_batch(d, users, c, rng, positives=8, negatives=32):
    targets = sample_choices(d, users, rng, positives, negatives)
    positive = targets['labels'] == 1
    stride = d['items'] + 1
    excluded = targets['rows'][positive] * stride + targets['items'][positive] + 1
    # Build batch-local histories without held-out IDs, including their dates and ratings.
    local = {key: d[key] for key in ('items', 'mean', 'min_day', 'max_day')}
    local['users'] = len(users)
    for family, fields in [('train', ('items', 'days', 'ratings')), ('implicit', ('items', 'days'))]:
        rows, index, _ = mh.gather(d[family + '_offsets'], users)
        keep = ~np.isin(rows * stride + d[family + '_items'][index], excluded)
        local[family + '_offsets'] = np.r_[0, np.cumsum(np.bincount(rows[keep], minlength=len(users)))]
        for field in fields:
            local[family + '_' + field] = d[family + '_' + field][index[keep]]
    ids = np.arange(len(users), dtype=np.int64)
    rows, index, _ = mh.gather(local['train_offsets'], ids)
    keep = rng.random(len(index)) >= c.input_drop
    empty = np.zeros(0, dtype=np.int64)
    host = dataclasses.replace(c, context='none', time_bins=0, week_days=0, device='cpu')
    b = mh.prepare(local, ids, host, keep=keep, queries=(empty, empty, empty))
    if keep.any():
        seq, width = ms.sequence_inputs(local, ids, rows, index, keep, empty, empty, empty, host, rng)
    else:
        # The existing sequence builder assumes at least one token. USER-only
        # batches are legitimate for cold users and heavily masked fixtures.
        width = c.pad_width
        seq = {key: torch.empty(0, dtype=torch.int64) for key in
               ('seq_flat', 'seq_items', 'seq_item_star', 'seq_stars', 'seq_weeks')}
        seq['seq_features'] = torch.empty((0, ms.TIME_FEATURES), dtype=torch.float32)
        seq['seq_lengths'] = torch.ones(len(users), dtype=torch.int64)
    seq['seq_target_flat'] = torch.arange(len(users), dtype=torch.int64) * width
    b.update(seq, seq_width=width)
    b.update({f'choice_{key}': torch.from_numpy(value) for key, value in targets.items()})
    return b


class SelectionModel(ms.SequenceModel):
    def __init__(self, items, c, weeks=400):
        super().__init__(items, c, weeks)
        self.choice_output = nn.Parameter(torch.empty(items, c.rank))
        self.choice_bias = nn.Parameter(torch.zeros(items))
        nn.init.normal_(self.choice_output, std=.04)

    def choice_loss(self, b):
        code = self.encode(b)  # USER slots; no candidate IDs enter the encoder.
        rows, items = b['choice_rows'], b['choice_items']
        logits = (code[rows] * self.choice_output[items]).sum(1) + self.choice_bias[items]
        loss = F.binary_cross_entropy_with_logits(logits, b['choice_labels'], reduction='none')
        return (loss * b['choice_weights']).sum()


def warm_model(checkpoint, device='cpu', seed=20261019, lr=.0003):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    c = ms.Config(**saved['config'])
    if c.retrieval_table or c.loss != 'mse' or c.objective != 'masked':
        raise ValueError('requires an ordinary masked-rating MSE checkpoint without retrieval')
    c = dataclasses.replace(c, device=device, lr=lr)
    torch.manual_seed(seed)
    model = SelectionModel(saved['items'], c, saved['weeks'])
    missing, unexpected = model.load_state_dict(saved['state'], strict=False)
    if set(missing) != {'choice_output', 'choice_bias'} or unexpected:
        raise ValueError('warm checkpoint must contain exactly the rating-only model')
    return model.to(device), saved['epoch']


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_epoch(model, d, optimizer, epoch, weight, seed, positives=8, negatives=32, max_batches=0):
    c = model.config
    host = dataclasses.replace(c, device='cpu')
    counts = np.diff(d['train_offsets']).astype(np.int64)
    active = np.flatnonzero(counts > 0)
    repeats = np.minimum(np.maximum(-(-counts[active] // c.seq_len), 1), max(c.windows, 1))
    samples = np.repeat(active, repeats)
    rng = np.random.default_rng(seed + epoch)
    auxiliary_rng = np.random.default_rng(seed + epoch + 1000000)
    groups = ms.length_batches(ms.capped_lengths(d, c), rng.permutation(samples), c, rng)
    def prepare(group):
        rating = ms.prepare_batch(d, group, host, rng=rng)
        auxiliary = auxiliary_batch(d, group, host, auxiliary_rng, positives, negatives)
        return dict(rating=rating, auxiliary=auxiliary)
    start = time.monotonic()
    sse, targets, total_rating, total_choice, batch_count = 0., 0, 0., 0., 0
    gradient_samples, rating_gradient, auxiliary_gradient = 0, 0., 0.
    model.train()
    stream = ms.prefetched(prepare, groups, 'cpu')
    for batch_no, (_, pair) in enumerate(stream, 1):
        rating = move(pair['rating'], c.device)
        optimizer.zero_grad(set_to_none=True)
        raw = model.forward_batch(rating)
        rating_loss = mh.loss_value(model, raw, rating['target_ratings'])
        if not torch.isfinite(rating_loss):
            raise FloatingPointError('nonfinite rating loss')
        rating_loss.backward()
        inspect_gradient = batch_no == 1 or batch_no % 100 == 0
        shared = (model.code_out.weight, model.blocks[0].qkv.weight)
        if inspect_gradient:
            before = [p.grad.detach().clone() for p in shared]
        with torch.no_grad():
            sse += float(((model.expected(raw) - rating['target_ratings'] - 1) ** 2).sum())
        count = rating['target_ratings'].numel()
        targets += count
        total_rating += float(rating_loss.detach()) * count
        # Free the rating graph before the second forward to bound peak memory.
        del raw, rating_loss, rating
        auxiliary = move(pair['auxiliary'], c.device)
        choice_loss = model.choice_loss(auxiliary)
        if not torch.isfinite(choice_loss):
            raise FloatingPointError('nonfinite choice loss')
        (weight * choice_loss).backward()
        if inspect_gradient:
            gradient_samples += 1
            rating_gradient += math.sqrt(sum(float(g.square().sum()) for g in before))
            auxiliary_gradient += math.sqrt(sum(float((p.grad - g).square().sum()) for p, g in zip(shared, before)))
        total_choice += float(choice_loss.detach())
        batch_count += 1
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if batch_no % 100 == 0:
            print(json.dumps(dict(epoch=epoch, batch=batch_no, train_rmse=math.sqrt(sse / targets),
                                  choice_loss=total_choice / batch_count,
                                  shared_rating_gradient_norm=rating_gradient / gradient_samples,
                                  shared_weighted_choice_gradient_norm=auxiliary_gradient / gradient_samples,
                                  seconds=time.monotonic() - start)), flush=True)
        if max_batches and batch_no >= max_batches:
            break
    stream.close()
    return dict(epoch=epoch, batches=batch_count, targets=targets, train_rmse=math.sqrt(sse / targets),
                rating_loss=total_rating / targets, choice_loss=total_choice / batch_count,
                shared_gradient_samples=gradient_samples,
                shared_rating_gradient_norm=rating_gradient / gradient_samples,
                shared_weighted_choice_gradient_norm=auxiliary_gradient / gradient_samples,
                train_seconds=time.monotonic() - start)


def run(a):
    if not math.isfinite(a.weight) or a.weight < 0 or min(a.epochs, a.threads, a.positives, a.negatives) < 1:
        raise ValueError('invalid objective or training budget')
    if not math.isfinite(a.lr) or a.lr <= 0 or a.max_batches < 0:
        raise ValueError('invalid learning rate or batch cap')
    torch.set_num_threads(a.threads)
    output = Path(a.output)
    if list(output.parent.glob(output.name + '.*')):
        raise FileExistsError('choose a new output prefix')
    output.parent.mkdir(parents=True, exist_ok=True)
    training = json.loads(Path(a.warm_report).read_text())
    data_sha = mh.digest(a.data)
    data_field = 'selection_sha256' if a.warm_stage == 'selection' else 'data_sha256'
    if training.get(data_field) != data_sha:
        raise ValueError('fine-tuning dataset must match the recorded warm-start selection dataset')
    model, start_epoch = warm_model(a.warm_checkpoint, a.device, a.seed, a.lr)
    if start_epoch != training['selected_epoch']:
        raise ValueError('warm checkpoint epoch does not match its training report')
    expected = dict(training['config'], retrieval_table='', retrieval_frac=.5)
    actual = dataclasses.asdict(model.config)
    if {k: v for k, v in actual.items() if k not in ('device', 'lr')} != {
            k: v for k, v in expected.items() if k not in ('device', 'lr')}:
        raise ValueError('warm configuration differs from its report')
    d = load(a.data)
    if d['items'] != len(model.choice_bias):
        raise ValueError('checkpoint movie count differs')
    report = dict(arguments=vars(a), config=actual, data_sha256=data_sha,
                  warm_checkpoint_sha256=mh.digest(a.warm_checkpoint), warm_epoch=start_epoch,
                  warm_report_sha256=mh.digest(a.warm_report), sources={
                      p.name: mh.digest(p) for p in [Path(__file__), Path(ms.__file__), Path(mh.__file__),
                                                     Path(__file__).with_name('gbdt_blend.py')]},
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  metric='P2 development predictions; fixed fine-tuning epochs, no hidden-test claim', records=[])
    mh.atomic_json(str(output) + '.json', report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=model.config.weight_decay)
    for epoch in range(1, a.epochs + 1):
        rec = train_epoch(model, d, optimizer, epoch, a.weight, a.seed, a.positives, a.negatives, a.max_batches)
        pred = ms.predict(model, d)
        pred_path = str(output) + f'.e{epoch}.f32'
        mh.write_f32(pred_path, pred)
        rec.update(prediction_sha256=mh.digest(pred_path),
                   validation_rmse=float(np.sqrt(np.mean((np.clip(pred.astype(float), 1, 5) - d['probe']['rating']) ** 2))))
        checkpoint = str(output) + f'.e{epoch}.pt'
        torch.save(dict(config=actual, items=d['items'], weeks=ms.weeks(d), epoch=epoch,
                        warm_epoch=start_epoch, state=model.state_dict()), checkpoint)
        rec['checkpoint_sha256'] = mh.digest(checkpoint)
        report['records'].append(rec)
        mh.atomic_json(str(output) + '.json', report)
        print(json.dumps(rec), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('data', 'warm-checkpoint', 'warm-report', 'output'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--weight', type=float, required=True)
    p.add_argument('--warm-stage', choices=('selection', 'refit'), default='selection')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--positives', type=int, default=8)
    p.add_argument('--negatives', type=int, default=32)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    p.add_argument('--max-batches', type=int, default=0, help='non-scoring smoke test only')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
