#!/usr/bin/env python3
"""Measure SO1 batch shapes on CPU without a model, gradients or GPU work."""
import argparse
import dataclasses
import json
from pathlib import Path
import time

import numpy as np
import torch

import selection_objective as so
from selection_objective_queue import verify
from train_holdout import digest


def summary(records):
    result = {}
    for arm in ('rating', 'auxiliary'):
        groups = [r[arm] for r in records]
        active_linear = sum(r['users']*r['width'] for r in groups)
        active_attention = sum(r['users']*r['width']**2 for r in groups)
        alternatives = {}
        for pad in (1, 8, 16, 32, 64):
            dimensions = [(-(-r['users']//pad)*pad, r['width']) for r in groups]
            linear = sum(n*w for n, w in dimensions)
            attention = sum(n*w*w for n, w in dimensions)
            alternatives[str(pad)] = dict(dense_token_slots=linear, dense_attention_cells=attention,
                row_padding_linear_multiplier=linear/active_linear,
                row_padding_attention_multiplier=attention/active_attention,
                distinct_batch_shapes=len(set(dimensions)))
        result[arm] = dict(batches=len(groups), real_users=sum(r['users'] for r in groups),
            real_tokens=sum(r['real_tokens'] for r in groups),
            min_users=min(r['users'] for r in groups), max_users=max(r['users'] for r in groups),
            min_width=min(r['width'] for r in groups), max_width=max(r['width'] for r in groups),
            padding_options=alternatives)
    return result


def run(a):
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(a.plan).read_text())
    verify(plan['frozen'])
    torch.set_num_threads(1)
    settings = plan['settings']
    warm = json.loads(Path(plan['warm_report']).read_text())
    c = so.ms.Config(**dict(warm['config'], device='cpu', lr=settings['lr']))
    d = so.load(plan['data'])
    counts = np.diff(d['train_offsets']).astype(np.int64)
    active = np.flatnonzero(counts > 0)
    repeats = np.minimum(np.maximum(-(-counts[active]//c.seq_len), 1), max(c.windows, 1))
    rng = np.random.default_rng(settings['seed']+1)
    auxiliary_rng = np.random.default_rng(settings['seed']+1+1000000)
    groups = so.ms.length_batches(so.ms.capped_lengths(d, c), rng.permutation(np.repeat(active, repeats)), c, rng)
    if not 0 < a.batches <= len(groups):
        raise ValueError('invalid batch count')
    records = []
    start = time.monotonic()
    for number, users in enumerate(groups[:a.batches], 1):
        rating = so.ms.prepare_batch(d, users, c, rng=rng)
        auxiliary = so.auxiliary_batch(d, users, c, auxiliary_rng, settings['positives'], settings['negatives'])
        rec = {'batch': number}
        for name, batch in (('rating', rating), ('auxiliary', auxiliary)):
            n, width = len(users), batch['seq_width']
            lengths = batch['seq_lengths'].numpy()
            assert len(lengths) == n and int(lengths.max()) <= width
            assert len(batch['seq_items'])+n == int(lengths.sum())
            rec[name] = dict(users=n, width=width, real_tokens=int(lengths.sum()))
        records.append(rec)
        if number % 32 == 0:
            print(json.dumps(dict(prepared_batches=number, total_batches=len(groups), seconds=time.monotonic()-start)), flush=True)
        del rating, auxiliary
    verify(plan['frozen'])
    result = dict(plan_sha256=digest(a.plan), source_sha256=digest(__file__),
        state='read_only_shape_audit', sampled_batches=a.batches, full_epoch_batches=len(groups),
        sampler='First consecutive batches of the frozen SO1 epoch-1 shuffled order; includes both passes with exact CPU preparation RNG streams.',
        config=dataclasses.asdict(c), summary=summary(records), records=records,
        seconds=time.monotonic()-start,
        scope='Actual CPU input preparation only; no model, gradients, optimizer, validation scoring, or GPU tensor. Dense cell/slot counts are analytical workload proxies, not measured kernel operations, speedups or memory use.',
        compatibility='Reducing row padding changes training tensor shapes and dropout RNG consumption. Any future training comparison needs a separate fixed matched experiment; do not edit the queued SO1 recipe.',
        verification_needed='Benchmark representative actual batches on idle MPS, including compile/warm-up cost, sustained wall time, memory and shape cache behavior; verify inference and matched-control learning before adopting a new shape policy.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='experiments/SO1-plan.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=128)
    run(parser.parse_args())
