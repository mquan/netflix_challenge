#!/usr/bin/env python3
"""Export exact frozen E2 peer codes under a hash-bound plan; no training.

The caller owns GPU scheduling. A partial cache is resumable but cannot serve
training. Each committed chunk is flushed and hashed before advancing progress.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from e2_joint_context import peer_code_batch, frozen_peer_vectors
from e2_query_codes import parameter_hash
from gbdt_blend import load
import masked_sequence as ms
from train_holdout import digest


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class CacheWriter:
    def __init__(self, output, binding, count, rank, chunk_rows, *, rows=None, resume=False):
        self.output = Path(output)
        if min(count, rank, chunk_rows) < 1:
            raise ValueError('positive cache dimensions required')
        if rows is not None:
            rows = np.asarray(rows, dtype=np.int64)
            if rows.shape != (count,) or rows[0] < 0 or np.any(rows[1:] <= rows[:-1]):
                raise ValueError('sparse source rows must be sorted, unique, nonnegative')
        self.identity = dict(binding=binding, count=count, rank=rank, chunk_rows=chunk_rows,
                             source_rows_sha256=None if rows is None else array_hash(rows))
        self.rows = rows
        self.path = self.output/'manifest.json'
        if resume:
            self.state = json.loads(self.path.read_text())
            if self.state['identity'] != self.identity:
                raise ValueError('cache plan or source rows changed')
            self.vectors = np.load(self.output/'vectors.npy', mmap_mode='r+')
            self.validate()
            if rows is not None and not np.array_equal(np.load(self.output/'rows.npy'), rows):
                raise ValueError('cached source row map changed')
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            self.vectors = np.lib.format.open_memmap(self.output/'vectors.npy', mode='w+',
                                                    dtype='<f4', shape=(count, rank))
            if rows is not None:
                np.save(self.output/'rows.npy', rows)
            self.state = dict(identity=self.identity, completed=0, chunks=[], complete=False)
            atomic_json(self.path, self.state)

    def validate(self):
        count, rank = self.identity['count'], self.identity['rank']
        if self.vectors.shape != (count, rank) or self.vectors.dtype != np.dtype('<f4'):
            raise ValueError('invalid vector payload shape or dtype')
        end = 0
        for record in self.state['chunks']:
            start = end
            end = min(start+self.identity['chunk_rows'], count)
            if record != dict(end=end, sha256=array_hash(self.vectors[start:end])):
                raise ValueError('committed cache chunk changed')
        if end != self.state['completed'] or (self.state['complete'] and end != count):
            raise ValueError('invalid cache progress')

    def append(self, values):
        start = self.state['completed']
        end = min(start+self.identity['chunk_rows'], self.identity['count'])
        values = np.asarray(values, dtype='<f4')
        if self.state['complete'] or start == end or values.shape != (end-start, self.identity['rank']):
            raise ValueError('next complete chunk required')
        if not np.isfinite(values).all():
            raise ValueError('nonfinite peer vectors')
        self.vectors[start:end] = values
        self.vectors.flush()
        with (self.output/'vectors.npy').open('rb') as stream:
            os.fsync(stream.fileno())
        self.state['chunks'].append(dict(end=end, sha256=array_hash(values)))
        self.state['completed'] = end
        atomic_json(self.path, self.state)

    def seal(self):
        self.validate()
        if self.state['completed'] != self.identity['count']:
            raise ValueError('cache is incomplete')
        self.state['complete'] = True
        atomic_json(self.path, self.state)


class CodeCache:
    def __init__(self, output, *, binding):
        output = Path(output)
        state = json.loads((output/'manifest.json').read_text())
        if not state['complete']:
            raise ValueError('cache is incomplete')
        identity = state['identity']
        rows = None if identity['source_rows_sha256'] is None else np.load(output/'rows.npy')
        if identity['binding'] != binding:
            raise ValueError('cache binding differs from requested producer')
        # Read-only validation never repairs or advances a cache.
        writer = object.__new__(CacheWriter)
        writer.identity, writer.state = identity, state
        writer.vectors = np.load(output/'vectors.npy', mmap_mode='r')
        writer.validate()
        if rows is not None and (rows.shape != (identity['count'],)
                or rows.dtype != np.dtype('int64') or array_hash(rows) != identity['source_rows_sha256']
                or rows[0] < 0 or np.any(rows[1:] <= rows[:-1])):
            raise ValueError('cached source row map changed')
        self.rows, self.vectors = rows, writer.vectors

    def __call__(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError('source row vector required')
        positions = rows if self.rows is None else np.searchsorted(self.rows, rows)
        if np.any((positions < 0) | (positions >= len(self.vectors))):
            raise ValueError('uncached peer source row')
        if self.rows is not None and np.any(self.rows[positions] != rows):
            raise ValueError('uncached peer source row')
        return np.asarray(self.vectors[positions])


def export_chunks(writer, d, model, batch_pairs, *, max_chunks=0):
    if batch_pairs < 1 or max_chunks < 0:
        raise ValueError('invalid export batch/chunk limit')
    exported = 0
    while writer.state['completed'] < writer.identity['count']:
        start = writer.state['completed']
        end = min(start+writer.identity['chunk_rows'], writer.identity['count'])
        pieces = []
        started = time.monotonic()
        for begin in range(start, end, batch_pairs):
            stop = min(begin+batch_pairs, end)
            rows = np.arange(begin, stop) if writer.rows is None else writer.rows[begin:stop]
            batch = peer_code_batch(d, rows, model.config)
            vectors, _ = frozen_peer_vectors(model, batch)
            pieces.append(vectors.cpu().numpy())
        writer.append(np.concatenate(pieces))
        print(json.dumps(dict(completed=end, count=writer.identity['count'],
                              seconds=time.monotonic()-started)), flush=True)
        exported += 1
        if max_chunks and exported >= max_chunks:
            break


def run(plan_path, output, *, resume=False, max_chunks=0):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    required = [plan['data'], plan['checkpoint'], plan['warm_report']]
    required += [str(Path(__file__).resolve().with_name(name)) for name in
        ('e2_code_cache.py', 'e2_joint_context.py', 'e2_query_codes.py', 'masked_sequence.py',
         'masked_history.py', 'gbdt_blend.py', 'train_holdout.py')]
    if plan.get('source_rows'):
        required.append(plan['source_rows'])
    if not set(required) <= set(plan['frozen']):
        raise ValueError('cache plan must bind dataset, checkpoint, report and all producer sources')
    def verify():
        for path, expected in plan['frozen'].items():
            if digest(path) != expected:
                raise ValueError('frozen input changed: '+path)
    verify()
    if plan['device'] not in ('cpu', 'mps') or plan['cpu_threads'] < 1:
        raise ValueError('invalid export device or thread count')
    torch.set_num_threads(plan['cpu_threads'])
    saved = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
    report = json.loads(Path(plan['warm_report']).read_text())
    if saved['epoch'] != report['selected_epoch'] or saved['config'] != report['config']:
        raise ValueError('checkpoint differs from warm report')
    if plan['frozen'][plan['data']] != report['selection_sha256']:
        raise ValueError('cache dataset differs from warm selection context')
    original = ms.Config(**saved['config'])
    if original.head != 'distribution' or original.loss != 'mse' or original.retrieval_table:
        raise ValueError('requires ordinary A2h rating encoder')
    config = dataclasses.replace(original, device=plan['device'], pad_rows=plan['pad_rows'])
    d = load(plan['data'])
    model = ms.SequenceModel(saved['items'], config, saved['weeks']).to(config.device)
    model.load_state_dict(saved['state'], strict=True)
    model.eval()
    before = parameter_hash(model)
    rows = None if not plan.get('source_rows') else np.load(plan['source_rows'])
    count = len(d['train_items']) if rows is None else len(rows)
    if rows is not None and (rows.dtype != np.dtype('int64') or np.any(rows >= len(d['train_items']))):
        raise ValueError('invalid sparse source rows')
    binding = dict(plan_sha256=digest(plan_path), model_sha256=before,
                   torch_version=str(torch.__version__))
    writer = CacheWriter(output, binding, count, config.rank, plan['chunk_rows'],
                         rows=rows, resume=resume)
    export_chunks(writer, d, model, plan['batch_pairs'], max_chunks=max_chunks)
    if parameter_hash(model) != before:
        raise ValueError('frozen producer parameters changed')
    verify()
    if writer.state['completed'] == count:
        writer.seal()
    return writer.state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-chunks', type=int, default=0)
    args = parser.parse_args()
    run(args.plan, args.output, resume=args.resume, max_chunks=args.max_chunks)
