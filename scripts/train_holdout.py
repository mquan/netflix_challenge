#!/usr/bin/env python3
"""Create a deterministic training-only probe in the existing Netflix binary format.

The original probe is omitted from the output, but its user/movie/date pairs
remain in the unchanged implicit-feedback arrays. Selected training ratings
likewise remain implicit and move from the training arrays to the new probe.
No rating value participates in split assignment.
"""
import argparse
import bisect
import hashlib
import json
import pathlib

import numpy as np

from gbdt_blend import HEADER, PROBE, load


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select(n, fraction, seed):
    """SplitMix64 hash of source row position; stable across NumPy RNG versions."""
    x = np.arange(n, dtype=np.uint64) + np.uint64(seed)
    with np.errstate(over='ignore'):
        x += np.uint64(0x9e3779b97f4a7c15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return x < np.uint64(int(fraction * (1 << 64)))


def select_matched(d, seed):
    """For each original probe pair, hold the nearest dated training row of its user.

    Original probe ratings are never read. A user keeps at least one labeled row.
    The item identity is not matched, so its distribution remains a diagnostic.
    """
    offsets = d['train_offsets']
    probe = d['probe']
    order = np.argsort(probe['user'], kind='stable')
    counts = np.bincount(probe['user'], minlength=d['users'])
    po = np.r_[0, np.cumsum(counts)]
    held = np.zeros(len(d['train_ratings']), dtype=bool)
    for u in np.flatnonzero(counts):
        lo, hi = int(offsets[u]), int(offsets[u+1])
        budget = min(int(counts[u]), hi-lo-1)
        if budget <= 0:
            continue
        days = d['train_days'][lo:hi]
        used = np.zeros(hi-lo, dtype=bool)
        for q in order[po[u]:po[u+1]][:budget]:
            day = int(probe['day'][q])
            right = bisect.bisect_left(days, day)
            left = right-1
            while left >= 0 and used[left]:
                left -= 1
            while right < len(days) and used[right]:
                right += 1
            if left < 0:
                pick = right
            elif right == len(days):
                pick = left
            else:
                dl, dr = day-int(days[left]), int(days[right])-day
                pick = left if dl < dr or (dl == dr and (u+int(q)+seed) % 2 == 0) else right
            used[pick] = True
            held[lo+pick] = True
    return held


def create(source, output, fraction, seed, strategy='matched'):
    if not 0 < fraction < 0.5:
        raise ValueError('fraction must be between 0 and 0.5')
    source, output = pathlib.Path(source), pathlib.Path(output)
    if source.resolve() == output.resolve() or output.exists() or pathlib.Path(str(output) + '.json').exists():
        raise FileExistsError('choose a new output path')
    d = load(source)
    n = len(d['train_ratings'])
    if strategy not in ('matched', 'random'):
        raise ValueError('strategy must be matched or random')
    held = select_matched(d, seed) if strategy == 'matched' else select(n, fraction, seed)
    old_offsets = d['train_offsets'].astype(np.int64)
    old_counts = np.diff(old_offsets)
    cumulative = np.r_[0, np.cumsum(held, dtype=np.int64)]
    held_counts = cumulative[old_offsets[1:]] - cumulative[old_offsets[:-1]]
    # Preserve one labeled training rating for users who had one or more.
    emptied = np.flatnonzero((held_counts == old_counts) & (old_counts > 0))
    held[old_offsets[emptied]] = False
    cumulative = np.r_[0, np.cumsum(held, dtype=np.int64)]
    held_counts = cumulative[old_offsets[1:]] - cumulative[old_offsets[:-1]]
    kept_counts = old_counts - held_counts
    kept_offsets = np.r_[0, np.cumsum(kept_counts, dtype=np.uint64)]
    probe = np.empty(int(held.sum()), dtype=PROBE)
    probe['user'] = np.repeat(np.arange(d['users'], dtype=np.uint32), held_counts)
    probe['item'] = d['train_items'][held]
    probe['day'] = d['train_days'][held]
    probe['rating'] = d['train_ratings'][held]
    probe['pad'] = b'\0\0\0'
    kept_ratings = d['train_ratings'][~held]
    header = np.empty(1, dtype=HEADER)
    header[0] = (b'NFPDAT2\0', d['users'], d['items'], len(kept_ratings), len(probe),
                 len(d['implicit_items']), int(d['min_day']), int(d['max_day']), 0,
                 float(kept_ratings.astype(np.float64).mean()))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(str(output) + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open('xb') as stream:
            for part in (header, kept_offsets, d['train_items'][~held], d['train_days'][~held],
                         kept_ratings, probe, d['implicit_offsets'], d['implicit_items'], d['implicit_days']):
                part.tofile(stream)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = dict(source=str(source), source_sha256=digest(source), output=str(output),
                  output_sha256=digest(output), strategy=strategy, fraction=fraction if strategy == 'random' else None,
                  seed=seed,
                  train_count=int(len(kept_ratings)), holdout_count=int(len(probe)),
                  source_probe_count=int(len(d['probe'])), implicit_count=int(len(d['implicit_items'])),
                  users_left_without_train=int(np.sum(kept_counts == 0)),
                  holdout_rating_histogram=np.bincount(probe['rating'], minlength=6).tolist())
    pathlib.Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fraction', type=float, default=0.014)
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--strategy', choices=('matched', 'random'), default='matched')
    a = p.parse_args()
    print(json.dumps(create(a.data, a.output, a.fraction, a.seed, a.strategy), indent=2))


if __name__ == '__main__':
    main()
