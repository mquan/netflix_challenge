#!/usr/bin/env python3
"""Derive a smaller native neighbor graph by preserving each row's sorted prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

MAGIC = 0x4E46475241504831


def prefix_graph(body, neighbors):
    if len(body) < 24:
        raise ValueError('truncated graph header')
    magic, fingerprint, items, count = struct.unpack('<QQII', body[:24])
    if magic != MAGIC or not 1 <= neighbors <= count < items <= 65536:
        raise ValueError('invalid graph header or requested neighbor count')
    if len(body) != 24 + items * count * 6:
        raise ValueError('graph payload length mismatch')
    ids = np.frombuffer(body, dtype='<u2', offset=24, count=items * count).reshape(items, count)
    scores = np.frombuffer(body, dtype='<f4', offset=24 + 2 * items * count).reshape(items, count)
    if np.any(ids >= items) or np.any(ids == np.arange(items)[:, None]):
        raise ValueError('invalid neighbor IDs')
    if np.any(np.diff(np.sort(ids, axis=1).astype(np.int64), axis=1) == 0):
        raise ValueError('duplicate neighbor IDs')
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('invalid graph similarities')
    if np.any((scores[:, 1:] > scores[:, :-1]) |
              ((scores[:, 1:] == scores[:, :-1]) & (ids[:, 1:] > ids[:, :-1]))):
        raise ValueError('graph rows must be sorted by descending similarity then movie ID')
    output = (struct.pack('<QQII', magic, fingerprint, items, neighbors) +
              ids[:, :neighbors].tobytes() + scores[:, :neighbors].tobytes())
    return output, dict(items=items, source_neighbors=count, neighbors=neighbors,
                        dataset_fingerprint_hex=f'{fingerprint:016x}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--neighbors', required=True, type=int)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('choose a new output directory')
    body = args.source.read_bytes()
    result, info = prefix_graph(body, args.neighbors)
    args.output_dir.mkdir(parents=True)
    destination = args.output_dir / f'neighbors{args.neighbors}.bin'
    destination.write_bytes(result)
    manifest = dict(**info, source=str(args.source.resolve()),
                    source_sha256=hashlib.sha256(body).hexdigest(), output=str(destination.resolve()),
                    output_sha256=hashlib.sha256(result).hexdigest(),
                    transformation='Per-movie prefix of the native descending (similarity, movie-ID) ordering; fingerprint preserved.')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
