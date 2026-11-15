#!/usr/bin/env python3
"""Build deterministic, low-dimensional title/year movie vectors for a pilot.

This is a cheap lexical control, not a semantic model. The downstream scorer
also accepts richer precomputed movie vectors with the same float32 layout.
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import re

import numpy as np


STOP = {'a', 'an', 'and', 'at', 'by', 'for', 'from', 'in', 'of', 'on', 'or',
        'the', 'to', 'with', 'part', 'disc', 'disk', 'volume', 'vol', 'dvd'}


def read_titles(path):
    rows = {}
    for line in path.read_text(encoding='latin-1').splitlines():
        movie, year, title = line.split(',', 2)
        movie = int(movie)
        if movie in rows:
            raise ValueError(f'duplicate movie {movie}')
        rows[movie] = (None if year == 'NULL' else int(year), title)
    if set(rows) != set(range(1, len(rows) + 1)):
        raise ValueError('movie IDs must be consecutive from 1')
    return rows


def tokens(title):
    return {w for w in re.findall(r'[a-z0-9]+', title.casefold())
            if len(w) >= 2 and w not in STOP}


def build(rows, width=64):
    n = len(rows)
    docs = [tokens(rows[i][1]) for i in range(1, n + 1)]
    df = collections.Counter(w for doc in docs for w in doc)
    out = np.zeros((n, width + 8), dtype='<f4')
    for i, doc in enumerate(docs):
        for word in doc:
            # Independent fixed hash per word; no Python hash randomization.
            h = hashlib.blake2b(word.encode(), digest_size=8).digest()
            code = int.from_bytes(h, 'little')
            weight = math.log1p(n / (1 + df[word]))
            out[i, code % width] += weight * (1 if code >> 63 else -1)
        norm = np.linalg.norm(out[i, :width])
        if norm:
            out[i, :width] /= norm
        year = rows[i + 1][0]
        if year is not None:
            for j, center in enumerate(range(1935, 2006, 10)):
                out[i, width + j] = math.exp(-0.5 * ((year - center) / 12) ** 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--titles', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    args = ap.parse_args()
    rows = read_titles(args.titles)
    matrix = build(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.tofile(args.output)
    h = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps({
        'kind': 'title_year_lexical_control', 'items': len(rows),
        'dimensions': matrix.shape[1], 'sha256': h,
        'titles_sha256': hashlib.sha256(args.titles.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print(f'{matrix.shape} {h}')


if __name__ == '__main__':
    main()
