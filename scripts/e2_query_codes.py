#!/usr/bin/env python3
"""Bounded CPU audit of frozen A2h co-rater codes at their observed movie/day.

One query per user forward; that rating value is removed from every explicit
encoder input. Implicit identities/dates remain allowed, as in probe inference.
This measures exact query-conditioned codes, not a static USER-slot proxy.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import masked_history as mh
import masked_sequence as ms
from gbdt_blend import load
from train_holdout import digest


def parameter_hash(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def masked_query_batch(d, user, item, day, config):
    users = np.array([user], dtype=np.int64)
    rows, index, _ = mh.gather(d['train_offsets'], users)
    keep = d['train_items'][index].astype(np.int64) != item
    if np.count_nonzero(~keep) != 1:
        raise ValueError('co-rater code requires exactly one observed training pair')
    target_rows = np.zeros(1, dtype=np.int64)
    target_items = np.array([item-1], dtype=np.int64)
    target_days = np.array([day], dtype=np.int64)
    batch = mh.prepare(d, users, config, keep=keep,
        queries=(target_rows, target_items, target_days))
    sequence, width = ms.sequence_inputs(d, users, rows, index, keep,
        target_rows, target_items, target_days, config)
    batch.update(sequence, seq_width=width)
    return batch


@torch.inference_mode()
def query_code(model, batch):
    model.eval()
    raw_code = model.encode(batch)
    vector = raw_code
    if model.config.context == 'session':
        zero = torch.zeros(1, model.config.rank, dtype=raw_code.dtype, device=raw_code.device)
        parts = [raw_code]+model.channel(batch, 'day', model.session_implicit,
                                        model.session_explicit, batch['target_items'], zero)
        if model.config.week_days > 1:
            parts += model.channel(batch, 'week', model.week_implicit,
                                   model.week_explicit, batch['target_items'], zero)
        vector = raw_code+model.session_net(torch.cat(parts+[batch['target_features']], 1))
    items = batch['target_items']
    reconstructed = (model.output[items]*vector[:, None, :]).sum(2)+model.bias[items]
    if model.config.time_bins > 0:
        reconstructed += model.time_bias[items*model.config.time_bins+batch['target_bins']]
    stars = torch.arange(1, 6, dtype=raw_code.dtype, device=raw_code.device)
    shift = batch['features'][batch['target_rows'], :5]@stars-3.65
    reconstructed += shift[:, None]*(stars-3)[None, :]/1.2
    direct = model.decode(raw_code, batch['features'], batch['target_rows'], items, batch)
    error = float((reconstructed-direct).abs().max())
    if error > 1e-6:
        raise ValueError('captured code does not reconstruct the frozen decoder')
    return raw_code.cpu().numpy()[0], vector.cpu().numpy()[0], error


def select_rows(d, sample, per_stratum, seed):
    candidates = np.unique(sample['rows'][sample['mask']])
    owners = np.searchsorted(d['train_offsets'], candidates, side='right')-1
    lengths = np.diff(d['train_offsets'])[owners]
    bins = np.searchsorted(np.array([16, 64, 256, 1024]), lengths, side='left')
    rng = np.random.default_rng(seed)
    selected = []
    for group in range(5):
        available = np.flatnonzero(bins == group)
        if len(available):
            selected.extend(rng.choice(available, min(per_stratum, len(available)), replace=False))
    selected = np.asarray(selected, dtype=np.int64)
    return candidates[selected], owners[selected], lengths[selected], bins[selected]


def run(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    for path, expected in plan['frozen'].items():
        if digest(path) != expected:
            raise ValueError('frozen input changed: '+path)
    torch.set_num_threads(1)
    saved = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
    warm = json.loads(Path(plan['warm_report']).read_text())
    if saved['epoch'] != warm['selected_epoch'] or saved['config'] != warm['config']:
        raise ValueError('checkpoint differs from warm report')
    d = load(plan['data'])
    if plan['frozen'][plan['data']] != warm['selection_sha256']:
        raise ValueError('P2 code audit must use the warm selection context')
    original = ms.Config(**saved['config'])
    if original.head != 'distribution' or original.loss != 'mse' or original.retrieval_table:
        raise ValueError('requires the retained ordinary A2h rating encoder')
    config = dataclasses.replace(original, device='cpu', pad_rows=1)
    model = ms.SequenceModel(saved['items'], config, saved['weeks'])
    model.load_state_dict(saved['state'], strict=True)
    model.eval()
    before = parameter_hash(model)
    sample = np.load(plan['sample'])
    source_rows, owners, lengths, strata = select_rows(d, sample, plan['per_stratum'], plan['seed'])
    if not len(source_rows):
        raise ValueError('no actual co-rater records')
    output.mkdir(parents=True)
    raw, vectors, records = [], [], []
    for row, user, length, group in zip(source_rows, owners, lengths, strata):
        row, user = int(row), int(user)
        item, day = int(d['train_items'][row]), int(d['train_days'][row])
        started = time.monotonic()
        batch = masked_query_batch(d, user, item, day, config)
        prepared = time.monotonic()
        code, vector, error = query_code(model, batch)
        seconds = time.monotonic()-prepared
        raw.append(code); vectors.append(vector)
        records.append(dict(source_row=row, user=user, item=item, day=day,
            training_history=int(length), stratum=int(group), sequence_width=batch['seq_width'],
            prepare_seconds=prepared-started, forward_seconds=seconds, decoder_max_abs_error=error))
        print(json.dumps(records[-1]), flush=True)
    after = parameter_hash(model)
    if before != after:
        raise ValueError('frozen model parameters changed')
    np.savez(output/'codes.npz', source_rows=source_rows, users=owners,
             raw_codes=np.asarray(raw), session_adjusted_codes=np.asarray(vectors))
    for path, expected in plan['frozen'].items():
        if digest(path) != expected:
            raise ValueError('input changed during audit: '+path)
    report = dict(plan_sha256=digest(plan_path), records=records, model_sha256_before=before,
        model_sha256_after=after, output_sha256=digest(output/'codes.npz'),
        device='cpu', pad_rows=1, source_sha256=digest(__file__), torch_version=torch.__version__,
        definition='Exact one-target A2h query code for co-rater/movie/observed-day, with its rating value removed from all explicit channels.',
        limitation='Bounded CPU inference audit, not an E2 model result, full cache, GPU throughput or padding-64/MPS equivalence claim.')
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
