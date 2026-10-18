#!/usr/bin/env python3
"""Audit local metadata and item support without using any rating values.

Standard library only. Netflix titles are unquoted, so split at two commas.
DVD package markers are heuristics, not verified media types or franchise IDs.
"""
import argparse
import array
import collections
import hashlib
import json
import pathlib
import re
import struct
import sys
import unicodedata

HEADER = struct.Struct('<8sIIQQQHH4xd')
PROBE = struct.Struct('<IH6x')  # item ID only; skip date, rating and padding


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(title):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', title).casefold()))


def read_titles(path):
    titles = {}
    # Original title file contains Latin-1 bytes; preserve them without replacement.
    for line in pathlib.Path(path).read_text(encoding='latin-1').splitlines():
        movie, year, title = line.split(',', 2)
        movie = int(movie)
        if movie < 1 or movie in titles:
            raise ValueError('invalid or duplicate movie ID')
        titles[movie] = (None if year == 'NULL' else int(year), title)
    return titles


def audit(titles_path, data_path):
    titles = read_titles(titles_path)
    with open(data_path, 'rb') as f:
        magic, users, items, n, p, implicit, *_ = HEADER.unpack(f.read(HEADER.size))
        expected = HEADER.size + 16 * (users + 1) + 5 * n + 12 * p + 4 * implicit
        if magic != b'NFPDAT2\0' or pathlib.Path(data_path).stat().st_size != expected:
            raise ValueError('unsupported or truncated dataset')
        if set(titles) != set(range(1, items + 1)):
            raise ValueError('title IDs do not match processed item IDs')
        f.seek(HEADER.size + 8 * (users + 1))
        counts = collections.Counter()
        for start in range(0, n, 1_000_000):
            values = array.array('H')
            values.frombytes(f.read(2 * min(1_000_000, n - start)))
            if sys.byteorder != 'little':
                values.byteswap()
            counts.update(values)
        if not set(counts) <= set(titles):
            raise ValueError('invalid training item ID')
        f.seek(HEADER.size + 8 * (users + 1) + 5 * n)
        probe_counts = collections.Counter(item for _, item in PROBE.iter_unpack(f.read(12 * p)))
        if not set(probe_counts) <= set(titles):
            raise ValueError('invalid probe item ID')
    def coverage(ids):
        ids = set(ids)
        rows = sum(probe_counts[i] for i in ids)
        return dict(items=len(ids), probe_rows=rows, probe_fraction=rows / p if p else 0)
    keys = collections.defaultdict(list)
    for i, (year, title) in titles.items():
        keys[(normalize(title), year)].append(i)
    duplicates = [v for v in keys.values() if len(v) > 1]
    markers = r'\b(?:season|vol(?:ume)?\.?|disc|collection|box\s*set)\b'
    years = [year for year, _ in titles.values() if year is not None]
    return dict(
        inputs={str(path): digest(path) for path in (titles_path, data_path)},
        users=users, items=items, train_rows=n, probe_rows=p,
        counted_train_rows=sum(counts.values()), counted_probe_rows=sum(probe_counts.values()),
        release_year_range=[min(years), max(years)] if years else None,
        missing_year=coverage(i for i, (y, _) in titles.items() if y is None),
        package_marker=coverage(i for i, (_, t) in titles.items() if re.search(markers, t, re.I)),
        duplicate_normalized_title_year_groups=duplicates,
        duplicate_normalized_title_year=coverage(i for group in duplicates for i in group),
        item_support={str(limit): coverage(i for i in titles if counts[i] < limit)
                      for limit in (1, 10, 20, 50, 100, 500, 1000)},
        note='Support thresholds are exclusive; package markers are only lexical heuristics. No rating values used.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--titles', default='data/raw/extracted/download/movie_titles.txt')
    ap.add_argument('--data', default='data/processed/netflix.bin')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = pathlib.Path(args.output)
    if output.exists():
        ap.error('output exists; use a new report name')
    report = audit(args.titles, args.data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
