#!/usr/bin/env python3
"""Bounded, exact operators for the predeclared complete-trajectory study."""
import fcntl
import hashlib
import os
from pathlib import Path
import re
import numpy as np

COUNTS = {'probex': 1128164, 'qual': 2817131}
RAW391 = list(range(62)) + list(range(137, 466))


def sha(path):
    """Full immutable-file SHA; the cache hint affects only this descriptor."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        fcntl.fcntl(stream.fileno(), 48, 1)  # Darwin F_NOCACHE, previously fixture-verified.
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            digest.update(block)
        after, current = os.fstat(stream.fileno()), path.stat()
    def identity(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                stat.st_mode, stat.st_nlink, stat.st_flags)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError('Input changed during complete hashing: ' + str(path))
    return digest.hexdigest()


def relative(path):
    path = Path(path)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Repository-relative immutable binding required')
    return str(path)


def merge_bindings(target, incoming):
    for name, digest in incoming.items():
        name = relative(name)
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Full original SHA256 required')
        if name in target and target[name] != digest:
            raise ValueError('Conflicting original source expectation: ' + name)
        target[name] = digest


def native_rows(outputs, prefix, epochs, components=(), training=False):
    """Declare ALL native pairs in numeric epoch/terminal/component order."""
    prefix = relative(prefix)
    if epochs not in (2, 23) or len(set(components)) != len(components):
        raise ValueError('Full declared native trajectory required')
    stems = [prefix + '_ep%02d' % epoch for epoch in range(1, epochs + 1)]
    stems += [prefix] + [prefix + '-' + component for component in components]
    required = {stem + '.' + query + '.npy' for stem in stems for query in COUNTS}
    required.add(prefix + '.out')
    if training:
        required.update(prefix + '.' + split + '.npy' for split in ['trainx', 'fulltrain'])
    if set(outputs) != required:
        raise ValueError('Every original native output, including training evidence, required')
    merge_bindings({}, outputs)
    return [dict(id='native:' + Path(stem).name,
                 paired_files={q: stem + '.' + q + '.npy' for q in COUNTS})
            for stem in stems]


def partition(existing_hashes, candidates, candidate_hashes):
    """First complete-pair occurrence wins; a known-only alias never suffices."""
    if set(existing_hashes) != set(COUNTS) or set(candidate_hashes) != set(COUNTS):
        raise ValueError('Both full query hash lists required')
    if any(len(existing_hashes[q]) != 391 or len(candidate_hashes[q]) != len(candidates) for q in COUNTS):
        raise ValueError('All391 original pairs and every candidate required')
    if len({row['id'] for row in candidates}) != len(candidates):
        raise ValueError('Unique original candidate identities required')
    for query in COUNTS:
        if any(not re.fullmatch(r'[0-9a-f]{64}', h) for h in existing_hashes[query] + candidate_hashes[query]):
            raise ValueError('Full complete-vector SHA256 required')
    seen = {}
    for column in range(391):
        key = tuple(existing_hashes[q][column] for q in COUNTS)
        seen.setdefault(key, 'original391:%03d' % column)
    kept, aliases = [], []
    for column, row in enumerate(candidates):
        key = tuple(candidate_hashes[q][column] for q in COUNTS)
        if key in seen:
            aliases.append(dict(id=row['id'], alias_of=seen[key], clipped_pair_sha256=list(key)))
        else:
            seen[key] = row['id']
            kept.append(row)
    return kept, aliases


def scalar(root, row, query, lo, hi):
    if query not in COUNTS or not 0 <= lo <= hi <= COUNTS[query]:
        raise ValueError('Canonical query and complete-row bounds required')
    array = np.load(Path(root) / relative(row['paired_files'][query]), mmap_mode='r', allow_pickle=False)
    if array.shape != (COUNTS[query],) or array.dtype != np.dtype('<f4'):
        raise ValueError('Full original float32 query array required')
    values = np.clip(array[lo:hi], 0, 6).astype('<f4')
    # Reject nonfinite raw values even when clipping would turn infinity finite.
    if not np.isfinite(array[lo:hi]).all():
        raise ValueError('Finite original numerical predictions required')
    return values


def extend(z, f, columns):
    """Retain the old intercept and every467/617 parent bit in original order."""
    if (z.ndim != 2 or z.shape[1] != 467 or z.dtype != np.dtype('float64') or
        f.shape != (len(z), 617) or f.dtype != np.dtype('<f4')):
        raise ValueError('Original complete pairedB391 operators required')
    extra = np.column_stack(columns) if columns else np.empty((len(z), 0), dtype='<f4')
    if extra.shape != (len(z), len(columns)) or extra.dtype != np.dtype('<f4'):
        raise ValueError('Every full float32 addition required')
    if not np.isfinite(z).all() or not np.isfinite(f).all() or not np.isfinite(extra).all():
        raise ValueError('Finite complete operators required')
    return (np.column_stack([z[:, :-1], extra, z[:, -1:]]),
            np.column_stack([f, extra - np.float32(3.6)]))
