#!/usr/bin/env python3
"""Matched pooled, neighbor-structured, and shuffled-context P2 encoders.

Only neighbor grouping changes between pooled and structured. All arms use
the same path tokens, parameter count, base predictions and training budgets.
This is development screening; it does not score the original probe ensemble.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from gbdt_blend import blend_fold, load
from local_context_chain import paired
from local_structure_data import FIELDS
from train_holdout import digest


class NeighborEncoder(nn.Module):
    def __init__(self, global_width, width=16, pooled=False, local_normalization='none', disable_local=False):
        super().__init__()
        if local_normalization not in ('none', 'batch'):
            raise ValueError('unknown local normalization')
        self.pooled = pooled
        self.disable_local = disable_local
        self.edge = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.relation = nn.Sequential(nn.Linear(width * 3, width), nn.GELU())
        self.neighbor = nn.Sequential(nn.Linear(width + 3, width * 2), nn.GELU(),
                                      nn.Linear(width * 2, width), nn.GELU())
        # Across-query normalization addresses the measured coordinate-scale
        # imbalance. No learned affine parameters; evaluation uses fit-only
        # running statistics and cannot update them.
        self.local_norm = (nn.BatchNorm1d(width, affine=False, eps=1e-4, momentum=.1)
                           if local_normalization == 'batch' else nn.Identity())
        self.head = nn.Sequential(nn.Linear(global_width + width, 64), nn.GELU(), nn.Dropout(.1),
                                  nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def local_summary(self, tokens, mask, neighbor_meta):
        encoded = self.edge(tokens) * mask.unsqueeze(-1)
        counts = mask.sum(dim=2, keepdim=True)
        denominator = mask.sum(dim=(1, 2)).clamp_min(1).unsqueeze(-1)
        if self.pooled:
            mean = encoded.sum(dim=(1, 2)) / denominator
            row_context = mean[:, None, None, :].expand_as(encoded)
            movie_context = row_context
        else:
            row_context = (encoded.sum(dim=2) / counts.clamp_min(1)).unsqueeze(2).expand_as(encoded)
            movie_counts = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            movie_context = (encoded.sum(dim=1) / movie_counts).unsqueeze(1).expand_as(encoded)
        # One message-passing step preserves both common-neighbor and
        # common-movie correspondence in the structured arm.
        updated = self.relation(torch.cat([encoded, row_context, movie_context], dim=-1)) * mask.unsqueeze(-1)
        if self.pooled:
            summary = (updated.sum(dim=(1, 2)) / denominator).unsqueeze(1).expand(-1, mask.shape[1], -1)
        else:
            summary = updated.sum(dim=2) / counts.clamp_min(1)
        local = self.neighbor(torch.cat([summary, neighbor_meta], dim=-1))
        present = (counts > 0).float()
        return (local * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)

    def forward(self, inputs, global_features, baseline):
        if self.disable_local:
            local = global_features.new_zeros((len(global_features), self.neighbor[-1].out_features))
            return baseline + self.head(torch.cat([global_features, local], dim=1)).squeeze(1)
        mask = (inputs['cross_rating'] > 0).float()
        ur = (inputs['history_rating'].float() - 3) / 2
        vr = (inputs['cross_rating'].float() - 3) / 2
        nr = (inputs['neighbor_rating'].float() - 3) / 2
        ug = (inputs['history_gap'].float() / 365).clamp(-10, 10)
        cg = (inputs['cross_gap'].float() / 365).clamp(-10, 10)
        ng = (inputs['neighbor_gap'].float() / 365).clamp(-10, 10)
        ur, ug = ur.unsqueeze(1).expand_as(vr), ug.unsqueeze(1).expand_as(vr)
        nrr, ngg = nr.unsqueeze(2).expand_as(vr), ng.unsqueeze(2).expand_as(vr)
        tokens = torch.stack([ur, vr, nrr, ur - vr, (ur - vr).square(),
                              ug, cg, ngg, (ug - cg).abs()], dim=-1)
        meta = torch.stack([nr, ng, torch.log1p(mask.sum(dim=2)) / math.log(33)], dim=-1)
        local = self.local_norm(self.local_summary(tokens, mask, meta))
        return baseline + self.head(torch.cat([global_features, local], dim=1)).squeeze(1)


def read_inputs(spec_path, context_path):
    spec, context_path = json.loads(Path(spec_path).read_text()), Path(context_path)
    report = json.loads((context_path / 'manifest.json').read_text())
    if digest(spec_path) != report['spec_sha256'] or report['data_sha256'] != spec['data_sha256']:
        raise ValueError('context/spec mismatch')
    if digest(spec['data']) != spec['data_sha256']:
        raise ValueError('P2 dataset changed')
    for name, expected in report['outputs'].items():
        if digest(context_path / name) != expected:
            raise ValueError(f'context changed: {name}')
    arrays = {name: np.load(context_path / (name + '.npy'), mmap_mode='r') for name in FIELDS}
    queries = np.load(context_path / 'queries.npy')
    d = load(spec['data'])
    n = len(d['probe'])
    if len(queries) != report['rows'] or len(np.unique(queries)) != len(queries) or np.any((queries < 0) | (queries >= n)):
        raise ValueError('invalid query indices')
    columns, support = [], []
    for destination, sources in [(columns, spec['columns']), (support, spec.get('support', []))]:
        for item in sources:
            if digest(item['path']) != item['sha256']:
                raise ValueError(f"base column changed: {item['path']}")
            values = np.fromfile(item['path'], dtype='<f4')
            if len(values) != n or not np.isfinite(values).all():
                raise ValueError('invalid base column')
            destination.append(values[queries])
    predictions = np.column_stack(columns).astype(np.float64)
    global_features = np.column_stack(columns + support + [arrays['pooled']]).astype(np.float32)
    for name, (_, shape) in FIELDS.items():
        if arrays[name].shape != (len(queries), *shape):
            raise ValueError(f'context shape mismatch: {name}')
    if not np.isfinite(global_features).all():
        raise ValueError('nonfinite global inputs')
    return arrays, predictions, global_features, d['probe']['rating'][queries].astype(float), d['probe']['user'][queries], report


def fit_preprocessing(predictions, global_features, y, rows):
    """All supervision and normalization are restricted to the supplied fit rows."""
    design = np.column_stack([np.ones(len(y)), predictions - 3.6])
    x = design[rows]
    penalty = np.eye(x.shape[1]) * len(rows) * 1e-5
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ (y[rows] - 3.6))
    mean = global_features[rows].mean(axis=0, dtype=np.float64)
    scale = np.maximum(global_features[rows].std(axis=0, dtype=np.float64), 1e-4)
    features = np.clip((global_features - mean) / scale, -8, 8).astype(np.float32)
    return features, (3.6 + design @ weights).astype(np.float32), dict(weights=weights, mean=mean, scale=scale)


def batch(arrays, rows, device, shuffled):
    result = {}
    for name in ('history_rating', 'history_gap', 'neighbor_rating', 'neighbor_gap', 'cross_rating', 'cross_gap'):
        source = name.replace('cross_', 'shuffled_') if shuffled and name.startswith('cross_') else name
        result[name] = torch.from_numpy(np.asarray(arrays[source][rows])).to(device)
    return result


def predict(model, arrays, features, baseline, rows, a, shuffled):
    model.eval()
    out = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), a.batch):
            indices = rows[start:start + a.batch]
            value = model({} if model.disable_local else batch(arrays, indices, a.device, shuffled),
                          torch.from_numpy(features[indices]).to(a.device),
                          torch.from_numpy(baseline[indices]).to(a.device))
            out[start:start + len(indices)] = value.cpu().numpy()
    return out


def rmse(y, prediction):
    return float(np.sqrt(np.mean((y - np.clip(prediction, 1, 5)) ** 2)))


def train(arrays, features, baseline, y, rows, *, a, arm, seed, epochs, validation=None):
    if getattr(a, 'local_normalization', 'none') == 'batch' and (a.batch < 2 or len(rows) % a.batch == 1):
        raise ValueError('batch normalization requires at least two training rows per batch')
    torch.manual_seed(seed)
    model = NeighborEncoder(features.shape[1], a.width, pooled=(arm == 'pooled'),
                            local_normalization=getattr(a, 'local_normalization', 'none'),
                            disable_local=getattr(a, 'disable_local', False)).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    rng = np.random.default_rng(seed)
    best = rmse(y[validation], baseline[validation]) if validation is not None else None
    chosen, history = 0, []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled_rows = rng.permutation(rows)
        loss_sum = 0.
        for offset in range(0, len(rows), a.batch):
            indices = shuffled_rows[offset:offset + a.batch]
            optimizer.zero_grad(set_to_none=True)
            prediction = model({} if model.disable_local else batch(arrays, indices, a.device, arm == 'shuffled'),
                               torch.from_numpy(features[indices]).to(a.device),
                               torch.from_numpy(baseline[indices]).to(a.device))
            loss = (prediction - torch.from_numpy(y[indices].astype(np.float32)).to(a.device)).square().mean()
            if not torch.isfinite(loss).item():
                raise ValueError('nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
        record = dict(epoch=epoch, train_rmse=float(np.sqrt(loss_sum / len(rows))),
                      elapsed_seconds=time.monotonic() - started)
        if validation is not None:
            value = predict(model, arrays, features, baseline, validation, a, arm == 'shuffled')
            record['selection_rmse'] = rmse(y[validation], value)
            if record['selection_rmse'] < best:
                best, chosen = record['selection_rmse'], epoch
        history.append(record)
        print(json.dumps(dict(arm=arm, seed=seed, phase='select' if validation is not None else 'refit', **record)), flush=True)
        if validation is not None and epoch - chosen >= a.patience:
            break
    return model, chosen, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--context', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--summary', help='also save the completed report at this path')
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--seeds', nargs='+', type=int, default=[2026, 7])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--benchmark-steps', type=int, default=0)
    p.add_argument('--local-normalization', choices=['none', 'batch'], default='none')
    a = p.parse_args()
    if min(a.width, a.batch, a.epochs, a.patience, a.threads) < 1 or a.benchmark_steps < 0:
        p.error('positive training settings required')
    if a.device == 'mps' and not torch.backends.mps.is_available():
        p.error('MPS unavailable; choose CPU explicitly')
    output = Path(a.output)
    if a.summary and Path(a.summary).exists():
        p.error('summary already exists')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.threads)
    arrays, predictions, global_features, y, users, context_report = read_inputs(a.spec, a.context)
    folds = blend_fold(users)
    fit, selection, held, refit = [np.flatnonzero(mask) for mask in
                                  (folds >= 2, folds == 1, folds == 0, folds != 0)]
    if min(len(fit), len(selection), len(held)) < 1:
        raise ValueError('empty partition')
    print(json.dumps(dict(phase='ready', config=vars(a),
                          rows=dict(fit=len(fit), selection=len(selection), evaluation=len(held)),
                          global_features=global_features.shape[1],
                          parameters=sum(p.numel() for p in NeighborEncoder(global_features.shape[1], a.width).parameters()))), flush=True)
    select_features, select_base, _ = fit_preprocessing(predictions, global_features, y, fit)
    if a.benchmark_steps:
        rows = np.resize(fit, a.benchmark_steps * a.batch)
        started = time.monotonic()
        _, _, history = train(arrays, select_features, select_base, y, rows, a=a,
                              arm='structured', seed=a.seeds[0], epochs=1)
        seconds = time.monotonic() - started
        report = dict(config=vars(a), seconds=seconds, steps=a.benchmark_steps,
                      examples_per_second=len(rows) / seconds,
                      projected_fit_epoch_seconds=len(fit) / len(rows) * seconds,
                      note='Training throughput only; no selection or evaluation scores read.')
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return
    started = time.monotonic()
    refit_features, refit_base, preprocessing = fit_preprocessing(predictions, global_features, y, refit)
    np.savez(output / 'preprocessing.npz', **preprocessing)
    records, outputs = {}, {'baseline': refit_base[held].astype(float)}
    for arm in ('pooled', 'structured', 'shuffled'):
        average, records[arm] = np.zeros(len(held)), []
        for seed in a.seeds:
            _, epochs, selection_history = train(arrays, select_features, select_base, y, fit,
                a=a, arm=arm, seed=seed, epochs=a.epochs, validation=selection)
            model, _, refit_history = train(arrays, refit_features, refit_base, y, refit,
                a=a, arm=arm, seed=seed, epochs=epochs)
            value = predict(model, arrays, refit_features, refit_base, held, a, arm == 'shuffled')
            average += value / len(a.seeds)
            value.astype('<f4').tofile(output / f'{arm}-seed{seed}.f32')
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, output / f'{arm}-seed{seed}.pt')
            records[arm].append(dict(seed=seed, chosen_epochs=epochs, selection=selection_history, refit=refit_history,
                                     parameters=sum(parameter.numel() for parameter in model.parameters())))
        outputs[arm] = average
    # Held labels are accessed for scoring only after all arms and choices are fixed.
    comparisons = {}
    for left, right in [('baseline', 'pooled'), ('pooled', 'structured'),
                         ('shuffled', 'structured'), ('baseline', 'structured')]:
        comparisons[f'{right}_vs_{left}'] = paired(y[held], outputs[left], outputs[right], users[held])
    for name, value in outputs.items():
        value.astype('<f4').tofile(output / (name + '.f32'))
    np.save(output / 'held_indices.npy', held)
    per_seed = {}
    for seed in a.seeds:
        pooled = np.fromfile(output / f'pooled-seed{seed}.f32', dtype='<f4').astype(float)
        structured = np.fromfile(output / f'structured-seed{seed}.f32', dtype='<f4').astype(float)
        shuffled = np.fromfile(output / f'shuffled-seed{seed}.f32', dtype='<f4').astype(float)
        per_seed[str(seed)] = dict(structured_vs_pooled=paired(y[held], pooled, structured, users[held]),
                                   structured_vs_shuffled=paired(y[held], shuffled, structured, users[held]))
    report = dict(config=vars(a), seconds=time.monotonic() - started, torch_version=torch.__version__,
                  partition_rows=dict(fit=len(fit), selection=len(selection), evaluation=len(held)),
                  rmse={k: rmse(y[held], v) for k, v in outputs.items()}, comparisons=comparisons,
                  training=records, per_seed_comparisons=per_seed, shuffle=context_report['shuffle'],
                  spec_sha256=digest(a.spec), context_manifest_sha256=digest(Path(a.context) / 'manifest.json'),
                  outputs={path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()},
                  metric='P2 user-disjoint development screening; no original-probe ensemble improvement claim',
                  limitations=['P2 and base-model epochs have already been used for development.',
                               'Base bank has 11 rehearsal columns, not the full 79-column ensemble.',
                               'Shuffle changes neighbor rating distributions as well as correspondence.',
                               'Common pooled features retain original kNN signal in all three arms.'])
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if a.summary:
        with Path(a.summary).open('x') as stream:
            stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('seconds', 'rmse', 'comparisons', 'metric')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
