#!/usr/bin/env python3
"""Prepare train+probe and unlabeled qualifying queries, without training models.

Dense users follow the original ingest rule: first appearance while scanning
movie files in increasing movie order. Query rows remain in qualifying-file
order. A zero query rating is an explicit unknown-label sentinel, not a target.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from gbdt_blend import HEADER, PROBE, load
from train_holdout import digest

QUERY_ORDER = np.dtype([('raw_user', '<u4'), ('item', '<u2'), ('day', '<u2')])


def recover_users(training_dir, movies):
    mapping = {}
    fingerprint = hashlib.sha256()
    count = 0
    for movie in range(1, movies + 1):
        name = f'mv_{movie:07d}.txt'
        content = (Path(training_dir) / name).read_bytes()
        fingerprint.update(name.encode() + b'\0' + hashlib.sha256(content).digest())
        lines = content.splitlines()
        if not lines or lines[0] != f'{movie}:'.encode():
            raise ValueError(f'invalid movie header: {name}')
        for line in lines[1:]:
            raw = int(line.split(b',', 1)[0])
            if not 0 < raw < 2**32:
                raise ValueError('raw user outside uint32 range')
            if raw not in mapping:
                mapping[raw] = len(mapping)
            count += 1
        if movie % 1000 == 0:
            print(json.dumps(dict(stage='recover_users', movies=movie, users=len(mapping))), flush=True)
    return mapping, count, fingerprint.hexdigest()


def qualifying_rows(path, mapping, movies):
    rows, order = [], []
    movie = 0
    dates = {}
    epoch = datetime.date(1999, 1, 1)
    content = Path(path).read_bytes()
    for line in content.splitlines():
        if not line:
            continue
        if line.endswith(b':'):
            movie = int(line[:-1])
            if not 1 <= movie <= movies:
                raise ValueError('invalid qualifying movie')
            continue
        if not movie:
            raise ValueError('qualifying query before movie header')
        raw_text, date_text = line.split(b',')
        raw = int(raw_text)
        if raw not in mapping:
            raise ValueError('qualifying user absent from training')
        if date_text not in dates:
            day = (datetime.date.fromisoformat(date_text.decode()) - epoch).days
            if not 0 <= day < 65536:
                raise ValueError('qualifying date outside uint16 range')
            dates[date_text] = day
        day = dates[date_text]
        rows.append((mapping[raw], movie, day, 0, b'\0\0\0'))
        order.append((raw, movie, day))
    if not rows:
        raise ValueError('empty qualifying file')
    return np.array(rows, dtype=PROBE), np.array(order, dtype=QUERY_ORDER), hashlib.sha256(content).hexdigest()


def keys(users, items, days):
    return (np.asarray(users, dtype=np.uint64) << np.uint64(32)) | (
        np.asarray(days, dtype=np.uint64) << np.uint64(16)) | np.asarray(items, dtype=np.uint64)


def create(source, training_dir, qualifying, output_dir):
    source, training_dir, qualifying, output_dir = map(Path, (source, training_dir, qualifying, output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    source_sha = digest(source)
    d = load(source)
    header = np.fromfile(source, dtype=HEADER, count=1)
    if header[0]['magic'] != b'NFPDAT2' or d['items'] >= 65536:
        raise ValueError('unsupported source format')
    for ratings in (d['train_ratings'], d['probe']['rating']):
        if not np.all((ratings >= 1) & (ratings <= 5)):
            raise ValueError('source must contain labeled development training and probe')
    mapping, raw_count, raw_sha = recover_users(training_dir, d['items'])
    if len(mapping) != d['users'] or raw_count != len(d['train_ratings']) + len(d['probe']):
        raise ValueError('raw training dimensions differ from source')
    queries, query_order, qualifying_sha = qualifying_rows(qualifying, mapping, d['items'])
    inverse = np.empty(d['users'], dtype='<u4')
    for raw, dense in mapping.items():
        inverse[dense] = raw
    if not np.array_equal(inverse[queries['user']], query_order['raw_user']):
        raise ValueError('query mapping does not round trip')

    print(json.dumps(dict(stage='merge_and_verify', training=raw_count, queries=len(queries))), flush=True)
    old_users = np.repeat(np.arange(d['users'], dtype=np.uint32), np.diff(d['train_offsets']).astype(np.int64))
    labeled_keys = np.concatenate((keys(old_users, d['train_items'], d['train_days']),
                                   keys(d['probe']['user'], d['probe']['item'], d['probe']['day'])))
    del old_users
    permutation = np.argsort(labeled_keys, kind='stable')
    labeled_keys = labeled_keys[permutation]
    ratings = np.concatenate((d['train_ratings'], d['probe']['rating']))[permutation]
    del permutation
    implicit_users = np.repeat(np.arange(d['users'], dtype=np.uint32), np.diff(d['implicit_offsets']).astype(np.int64))
    implicit_keys = keys(implicit_users, d['implicit_items'], d['implicit_days'])
    del implicit_users
    if np.any(implicit_keys[1:] <= implicit_keys[:-1]):
        raise ValueError('implicit interactions must be unique and sorted by user/day/movie')
    query_keys = keys(queries['user'], queries['item'], queries['day'])
    positions = np.searchsorted(implicit_keys, query_keys)
    if np.any(positions >= len(implicit_keys)) or not np.array_equal(implicit_keys[positions], query_keys):
        raise ValueError('raw qualifying mapping disagrees with source implicit interactions')
    if len(np.unique(positions)) != len(queries):
        raise ValueError('duplicate qualifying query')
    labeled_mask = np.ones(len(implicit_keys), dtype=bool)
    labeled_mask[positions] = False
    if not np.array_equal(implicit_keys[labeled_mask], labeled_keys):
        raise ValueError('train + probe + qualifying do not exactly partition implicit interactions')
    del implicit_keys, labeled_mask, query_keys, positions
    counts = np.diff(d['train_offsets']) + np.bincount(d['probe']['user'], minlength=d['users']).astype(np.uint64)
    offsets = np.r_[np.uint64(0), np.cumsum(counts, dtype=np.uint64)]
    header[0]['train_count'] = len(ratings)
    header[0]['probe_count'] = len(queries)
    header[0]['mean'] = float(np.sum(ratings, dtype=np.uint64) / len(ratings))
    output = output_dir / 'train-probe-qualifying.bin'
    with output.open('xb') as stream:
        for part in (header, offsets, (labeled_keys & 65535).astype('<u2'),
                     ((labeled_keys >> np.uint64(16)) & 65535).astype('<u2'), ratings, queries,
                     d['implicit_offsets'], d['implicit_items'], d['implicit_days']):
            np.asarray(part).tofile(stream)
    del labeled_keys
    np.save(output_dir / 'dense-to-raw-user.npy', inverse, allow_pickle=False)
    np.save(output_dir / 'qualifying-order.npy', query_order, allow_pickle=False)
    check = load(output)
    if not np.array_equal(check['probe'], queries) or not np.array_equal(check['train_ratings'], ratings):
        raise ValueError('written dataset verification failed')
    for key in ('implicit_offsets', 'implicit_items', 'implicit_days'):
        if not np.array_equal(d[key], check[key]):
            raise ValueError(f'changed {key}')
    if digest(source) != source_sha or digest(qualifying) != qualifying_sha:
        raise ValueError('source changed during preparation')
    outputs = {p.name: digest(p) for p in (output, output_dir / 'dense-to-raw-user.npy', output_dir / 'qualifying-order.npy')}
    report = dict(state='complete', purpose='production_only_unlabeled_queries', source=str(source.resolve()),
        source_sha256=source_sha, training_dir=str(training_dir.resolve()),
        raw_training_fingerprint=raw_sha, raw_training_fingerprint_method='SHA256 of ascending filename, NUL, and binary SHA256 of each movie file',
        qualifying=str(qualifying.resolve()), qualifying_sha256=qualifying_sha,
        script_sha256=digest(__file__), users=d['users'], items=d['items'], train_count=len(ratings),
        qualifying_count=len(queries), original_train_count=len(d['train_ratings']), original_probe_count=len(d['probe']),
        implicit_count=len(d['implicit_items']), mean=float(header[0]['mean']), outputs=outputs,
        checks=['raw first-appearance user mapping round trips', 'labeled train+probe and qualifying exactly partition original implicit interactions',
                'qualifying file order preserved', 'original implicit arrays unchanged', 'original source digest unchanged'],
        limitation='Query ratings are zero sentinels. Never use this dataset to select epochs, fit a blender on query labels, or report query RMSE. Fixed-budget production trainers remain required.')
    (output_dir / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--training-dir', required=True)
    parser.add_argument('--qualifying', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.source, args.training_dir, args.qualifying, args.output_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
