"""Verify qualifying row identity and serialize already verified float32 ratings.

This low-level writer does not prove the fitted blender/inference provenance.
The final pipeline must establish those bindings before calling it.
"""
import datetime
import os
from pathlib import Path
import json
import struct

import numpy as np

from production_rbm_audit import digest, verify

ORDER = np.dtype([('raw_user', '<u4'), ('item', '<u2'), ('day', '<u2')])
QUERY = np.dtype([('user', '<u4'), ('item', '<u2'), ('day', '<u2'), ('rating', 'u1'), ('padding', 'S3')])


def raw_rows(path):
    """Yield original movie headers and query identities without reordering."""
    epoch = datetime.date(1999, 1, 1)
    movie, dates = 0, {}
    with Path(path).open('rb') as stream:
        for raw in stream:
            line = raw.rstrip(b'\r\n')
            if not line:
                continue
            if line.endswith(b':'):
                movie = int(line[:-1])
                if movie <= 0:
                    raise ValueError('Invalid raw qualifying movie')
                yield ('header', movie)
                continue
            if not movie:
                raise ValueError('Qualifying query before movie header')
            user, date = line.split(b',')
            if date not in dates:
                dates[date] = (datetime.date.fromisoformat(date.decode('ascii')) - epoch).days
            yield ('query', (int(user), movie, dates[date]))


def verify_order(manifest_path):
    path = Path(manifest_path).resolve()
    sha = digest(path)
    manifest = json.loads(path.read_text())
    if manifest['state'] != 'complete' or manifest['purpose'] != 'production_only_unlabeled_queries':
        raise ValueError('Complete unlabeled production manifest required')
    root = path.parent
    data = root / 'train-probe-qualifying.bin'
    inverse_path, order_path = root / 'dense-to-raw-user.npy', root / 'qualifying-order.npy'
    bound = {str(path): sha, manifest['qualifying']: manifest['qualifying_sha256']}
    for target in (data, inverse_path, order_path):
        bound[str(target)] = manifest['outputs'][target.name]
    verify(bound)
    with data.open('rb') as stream:
        header = stream.read(56)
    magic, users, items, train, query, implicit, first, last, mean = struct.unpack('<8sIIQQQHH4xd', header)
    if (magic != b'NFPDAT2\0' or (users, items, query) !=
            (manifest['users'], manifest['items'], manifest['qualifying_count'])):
        raise ValueError('Production query header differs')
    queries = np.memmap(data, dtype=QUERY, mode='r', offset=56 + 8 * (users + 1) + 5 * train, shape=(query,))
    inverse = np.load(inverse_path, mmap_mode='r', allow_pickle=False)
    order = np.load(order_path, mmap_mode='r', allow_pickle=False)
    if inverse.dtype != np.dtype('<u4') or inverse.shape != (users,) or order.dtype != ORDER or order.shape != (query,):
        raise ValueError('Qualifying mapping dimensions or types differ')
    for start in range(0, query, 65536):
        q, o = queries[start:start+65536], order[start:start+65536]
        if (np.any(q['user'] >= users) or np.any(q['rating'] != 0) or
                not np.array_equal(q['item'], o['item']) or not np.array_equal(q['day'], o['day']) or
                not np.array_equal(inverse[q['user']], o['raw_user'])):
            raise ValueError('Dense/raw qualifying identity or unknown labels differ')
    count, headers = 0, 0
    for kind, value in raw_rows(manifest['qualifying']):
        if kind == 'header':
            if value > items:
                raise ValueError('Qualifying movie exceeds dataset dimensions')
            headers += 1
        else:
            if count >= query or tuple(map(int, order[count])) != value:
                raise ValueError('Raw qualifying row order differs')
            count += 1
    if count != query or headers == 0:
        raise ValueError('Raw qualifying row count differs')
    verify(bound)
    return dict(state='verified_raw_qualifying_order', manifest=str(path), rows=count,
                movie_headers=headers, bindings=bound, metric=None)


def write_submission(manifest_path, prediction_path, output_path):
    output, prediction_path = Path(output_path), Path(prediction_path).resolve()
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Submission output or partial output already exists')
    order = verify_order(manifest_path)
    rows = order['rows']
    if prediction_path.stat().st_size != rows * 4:
        raise ValueError('Submission prediction count differs')
    predictions = np.memmap(prediction_path, dtype='<f4', mode='r')
    if not np.isfinite(predictions).all() or np.any(predictions < 1) or np.any(predictions > 5):
        raise ValueError('Submission ratings must be finite and within [1,5]')
    bound = dict(order['bindings'], **{str(prediction_path): digest(prediction_path)})
    manifest = json.loads(Path(manifest_path).read_text())
    count = 0
    with partial.open('xb') as stream:
        for kind, value in raw_rows(manifest['qualifying']):
            if kind == 'header':
                stream.write(f'{value}:\n'.encode('ascii'))
            else:
                stream.write(f'{float(predictions[count]):.8f}\n'.encode('ascii'))
                count += 1
    if count != rows:
        raise ValueError('Submission row count changed during writing')
    verify(bound)
    # Link is atomic and refuses to overwrite an output created concurrently.
    os.link(partial, output)
    partial.unlink()
    return dict(state='serialized_verified_raw_order_ratings', rows=rows, movie_headers=order['movie_headers'],
                bindings=bound, output=str(output.resolve()), output_sha256=digest(output), metric=None,
                scope='Serialization and raw row identity only. Fitted blender, audited bank and inference '
                'provenance must be verified separately before declaring a production export complete.')
