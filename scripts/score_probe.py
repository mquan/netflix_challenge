#!/usr/bin/env python3
"""Independently score raw float32 columns in NFPDAT2 Probe row order."""
import argparse
import array
import hashlib
import json
import math
import mmap
import pathlib
import struct
import sys


def score(data, paths):
    with data.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        header = struct.Struct('<8sIIQQQHH4xd')
        magic, users, items, train, probe, implicit, lo, hi, mean = header.unpack_from(mm)
        if magic != b'NFPDAT2\0' or not probe:
            raise ValueError('expected a nonempty NFPDAT2 Probe set')
        offset = header.size + 8*(users+1) + 5*train
        row = struct.Struct('<IHHB3x')
        if offset+row.size*probe > len(mm):
            raise ValueError('truncated Probe data')
        predictions = []
        for path in paths:
            raw = path.read_bytes()
            if len(raw) != probe*4:
                raise ValueError(f'{path}: expected {probe*4} bytes, got {len(raw)}')
            column = array.array('f'); column.frombytes(raw)
            if sys.byteorder != 'little':
                column.byteswap()
            sse = raw_sse = 0.
            for n, value in enumerate(column):
                if not math.isfinite(value):
                    raise ValueError(f'{path}: nonfinite prediction at row {n}')
                u, i, day, rating = row.unpack_from(mm, offset+row.size*n)
                if u >= users or not 1 <= i <= items or not 1 <= rating <= 5:
                    raise ValueError(f'invalid Probe record at row {n}')
                sse += (rating-min(5, max(1, value)))**2
                raw_sse += (rating-value)**2
            predictions.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                    clipped_rmse=math.sqrt(sse/probe),
                                    unclipped_rmse=math.sqrt(raw_sse/probe)))
    return dict(data=str(data), probe_count=probe, predictions=predictions)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', required=True, type=pathlib.Path)
    parser.add_argument('--prediction', required=True, action='append', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    result = json.dumps(score(args.data, args.prediction), indent=2)+'\n'
    if args.output:
        args.output.write_text(result)
    print(result, end='')


if __name__ == '__main__':
    main()
