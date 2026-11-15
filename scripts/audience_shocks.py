#!/usr/bin/env python3
"""AS1 bounded, training-only audience-shock pilot (NumPy only).

A weak, freshly fit bias baseline and fixed random item geometry deliberately
avoid unverified checkpoint provenance. This cannot establish incremental R9 gain.
All rating-dependent upstream features use only the upstream partition; residual
fit, inner selection and outer scoring rows are disjoint from that partition.
"""
import argparse
import dataclasses
import json
import pathlib
import subprocess
import time

import numpy as np

from experiment import digest
from gbdt_blend import blend_fold, load


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 711
    dimensions: int = 4
    window: int = 4
    min_users: int = 12
    profile_ridge: float = 10.
    permutations: int = 19
    null_quantile: float = .99
    ridge: tuple = (10., 100., 1000.)
    bootstrap: int = 300


def rows(user, item, day, rating):
    return dict(user=np.asarray(user, dtype=int), item=np.asarray(item, dtype=int),
                week=np.asarray(day, dtype=int) // 7, rating=np.asarray(rating, dtype=float))


def subset(r, mask):
    return {k: v[mask] for k, v in r.items()}


class Upstream:
    def __init__(self, r, users, items, cfg):
        self.cfg = cfg
        self.mean = float(r['rating'].mean())
        self.ub = np.bincount(r['user'], r['rating'] - self.mean, minlength=users)
        self.ub /= np.bincount(r['user'], minlength=users) + 20
        self.ib = np.bincount(r['item'], r['rating'] - self.mean - self.ub[r['user']], minlength=items)
        self.ib /= np.bincount(r['item'], minlength=items) + 40
        self.q = np.random.default_rng(cfg.seed).normal(size=(items, cfg.dimensions)) / np.sqrt(cfg.dimensions)
        self.sums = np.zeros((users, cfg.dimensions))
        residual = r['rating'] - 3.6
        np.add.at(self.sums, r['user'], residual[:, None] * self.q[r['item']])
        self.count = np.bincount(r['user'], minlength=users)
        # Netflix has one rating per user/movie. Reject duplicates rather than
        # silently failing to exclude all contributions for the candidate movie.
        key = r['user'].astype(np.int64) * items + r['item']
        order = np.argsort(key)
        self.keys = key[order]
        if np.any(np.diff(self.keys) == 0):
            raise ValueError('duplicate user/movie upstream ratings')
        self.contributions = residual[order]
        self.items = items

    def base(self, r):
        return self.mean + self.ub[r['user']] + self.ib[r['item']]

    def profile(self, r):
        key = r['user'].astype(np.int64) * self.items + r['item']
        at = np.searchsorted(self.keys, key)
        match = (at < len(self.keys)) & (self.keys[np.minimum(at, len(self.keys)-1)] == key)
        contribution = np.zeros(len(key))
        contribution[match] = self.contributions[at[match]]
        # Fixed centering keeps the other contributions independent of the
        # candidate rating; fitted base biases would leak through the profile.
        return (self.sums[r['user']] - contribution[:, None] * self.q[r['item']]) / (
            self.count[r['user']] - match + self.cfg.profile_ridge)[:, None]


def candidates(r, profiles, cfg):
    """Adjacent full windows, unique users, contemporaneous audience centering."""
    first, last = int(r['week'].min()), int(r['week'].max()) + 1
    output = []
    # Platform moments are interaction weighted; movie support is distinct users.
    platform = {}
    for start in range(first, last - 2 * cfg.window + 1):
        a = (r['week'] >= start) & (r['week'] < start + cfg.window)
        b = (r['week'] >= start + cfg.window) & (r['week'] < start + 2*cfg.window)
        if a.any() and b.any():
            platform[start] = (int(a.sum()), int(b.sum()), profiles[b].mean(0)-profiles[a].mean(0))
    variance = profiles.var(0) + .01
    for movie in np.unique(r['item']):
        own = np.flatnonzero(r['item'] == movie)
        if len(own) < 2*cfg.min_users or len(np.unique(r['user'][own])) < cfg.min_users:
            continue
        weeks = r['week'][own]
        for start, (a, b, drift) in platform.items():
            ia = own[(weeks >= start) & (weeks < start+cfg.window)]
            ib = own[(weeks >= start+cfg.window) & (weeks < start+2*cfg.window)]
            na, nb = len(ia), len(ib)
            support = [len(np.unique(r['user'][ia])), len(np.unique(r['user'][ib]))]
            if min(support) < cfg.min_users:
                continue
            # Difference in activity shares with finite-count regularization.
            pa, pb = (na+.5)/(a+1), (nb+.5)/(b+1)
            activity = abs(pb-pa)/np.sqrt(pa/a+pb/b+1e-12)
            delta = profiles[ib].mean(0)-profiles[ia].mean(0)-drift
            composition = np.sqrt(np.sum(delta**2 / variance) / (1/na+1/nb+1/cfg.min_users))
            output.append(dict(movie=int(movie), start=int(start+cfg.window), end=int(start+2*cfg.window),
                               before_users=support[0], after_users=support[1], before_count=na,
                               after_count=nb, activity=float(activity), composition=float(composition),
                               score=float(max(activity, composition))))
    return output


def detect(r, upstream, cfg):
    profiles = upstream.profile(r)
    observed = candidates(r, profiles, cfg)
    rng = np.random.default_rng(cfg.seed)
    maxima = []
    for _ in range(cfg.permutations):
        shuffled = {k: v.copy() for k, v in r.items()}
        # Permute movie assignments within 16-week strata. Preserves each
        # stratum's movie totals, dated user events and broad activity; destroys
        # within-stratum movie timing and user/movie association. Profiles must
        # be recomputed to exclude the newly assigned candidate movie.
        strata = r['week'] // 16
        for stratum in np.unique(strata):
            ix = np.flatnonzero(strata == stratum)
            shuffled['item'][ix] = rng.permutation(r['item'][ix])
        null = candidates(shuffled, upstream.profile(shuffled), cfg)
        maxima.append(max((e['score'] for e in null), default=0.))
    cutoff = float(np.quantile(maxima, cfg.null_quantile)) if maxima else float('inf')
    flagged = [e for e in observed if e['score'] > cutoff]
    merged = []
    for e in sorted(flagged, key=lambda x: (x['movie'], x['start'])):
        e = dict(e, confidence=float(min(e['before_users'], e['after_users']) /
                                    (min(e['before_users'], e['after_users'])+cfg.min_users)))
        if merged and merged[-1]['movie'] == e['movie'] and e['start'] <= merged[-1]['end']:
            old = merged[-1]
            old['end'] = max(old['end'], e['end'])
            # Retain representative statistics of the strongest flagged window.
            if e['score'] > old['score']:
                lo, hi = old['start'], old['end']
                old.update(e); old.update(start=lo, end=hi)
        else:
            merged.append(e)
    return merged, dict(candidates=len(observed), cutoff=cutoff, null_maxima=maxima,
                        calibration='maximum score per permutation; familywise empirical cutoff')


def design(r, upstream, events, arm):
    p = upstream.profile(r)
    width = 1 if arm == 'bias' else p.shape[1]+1
    x = np.zeros((len(r['user']), len(events)*width))
    for j, e in enumerate(events):
        own = r['item'] == e['movie']
        if arm == 'smooth':
            center = (e['start']+e['end'])/2
            weight = np.maximum(0, 1-abs(r['week']-center)/max(e['end']-e['start'], 1))
        else:
            weight = ((r['week'] >= e['start']) & (r['week'] < e['end'])).astype(float)
        weight *= own * e['confidence']
        x[:, j*width] = weight
        if width > 1:
            x[:, j*width+1:(j+1)*width] = weight[:, None]*p
    # Normalize overlapping event responses, including placebo and smooth arms.
    active = np.count_nonzero(x[:, ::width], axis=1)
    x /= np.maximum(active, 1)[:, None]
    return x


def fit(x, residual, ridge, width):
    if x.shape[1] == 0:
        return np.zeros(0)
    # Hierarchical ridge: event parameters shrink toward a shared response,
    # itself penalized toward zero. Eliminate the shared parameter analytically.
    count = x.shape[1] // width
    penalty = np.eye(x.shape[1])
    for k in range(width):
        ix = np.arange(k, x.shape[1], width)
        penalty[np.ix_(ix, ix)] -= 1/(count+1)
    return np.linalg.solve(x.T@x + ridge*penalty, x.T@residual)


def rmse(y, p):
    return float(np.sqrt(np.mean((y-p)**2)))


def evaluate(r, base, predictions, cfg):
    result = {}
    users, inv = np.unique(r['user'], return_inverse=True)
    counts = np.bincount(inv)
    rng = np.random.default_rng(cfg.seed)
    for arm, p in predictions.items():
        delta = (r['rating']-p)**2-(r['rating']-base)**2
        sums = np.bincount(inv, delta)
        boot = []
        for _ in range(cfg.bootstrap):
            ix = rng.integers(len(users), size=len(users))
            boot.append(float(sums[ix].sum()/counts[ix].sum()))
        result[arm] = dict(rmse=rmse(r['rating'], p), folds=[
            rmse(r['rating'][blend_fold(r['user']) == f], p[blend_fold(r['user']) == f])
            if np.any(blend_fold(r['user']) == f) else None for f in range(5)],
            paired_user_mse_difference=sums.tolist(), users=users.tolist(),
            paired_mse_ci95=np.quantile(boot, [.025, .975]).tolist())
    return result


def pilot(r, users, items, cfg):
    rng = np.random.default_rng(cfg.seed)
    split = rng.choice(4, len(r['user']), p=[.6, .2, .1, .1])
    if any(not np.any(split == k) for k in range(4)):
        raise ValueError('insufficient rows for four partitions')
    train, residual, inner, outer = [subset(r, split == k) for k in range(4)]
    upstream = Upstream(train, users, items, cfg)
    events, detection = detect(train, upstream, cfg)
    placebo = [dict(e, start=e['start']+8, end=e['end']+8) for e in events]
    predictions = {'base': upstream.base(outer)}
    models, selection = {}, {}
    for arm in ('smooth', 'bias', 'full', 'placebo'):
        ev = placebo if arm == 'placebo' else events
        kind = 'full' if arm == 'placebo' else arm
        xf, xi = [design(part, upstream, ev, kind) for part in (residual, inner)]
        width = 1 if arm == 'bias' else cfg.dimensions+1
        choices = []
        for ridge in cfg.ridge:
            beta = fit(xf, residual['rating']-upstream.base(residual), ridge, width)
            choices.append((rmse(inner['rating'], upstream.base(inner)+xi@beta), ridge, beta))
        _, ridge, beta = min(choices, key=lambda t: t[0])
        # Refit downstream only; upstream and detection stay fixed and clean.
        xr = np.concatenate([xf, xi])
        targets = np.r_[residual['rating']-upstream.base(residual), inner['rating']-upstream.base(inner)]
        beta = fit(xr, targets, ridge, width)
        models[arm] = beta
        selection[arm] = ridge
        predictions[arm] = upstream.base(outer)+design(outer, upstream, ev, kind)@beta
    scores = evaluate(outer, predictions['base'], predictions, cfg)
    best_control = min(scores[a]['rmse'] for a in ('base', 'smooth', 'bias', 'placebo'))
    passed = scores['full']['rmse'] <= best_control-.0001
    covered = np.any(design(outer, upstream, events, 'bias') != 0, axis=1)
    report = dict(events=events, detection=detection, selected_ridge=selection, scores=scores,
                  rows_by_partition=[int((split == k).sum()) for k in range(4)],
                  coverage=float(covered.mean()), required_event_mse_gain=.003897/max(covered.mean(), 1e-12),
                  pilot_gate_passed=passed, promotion_gate_passed=False,
                  status='pilot_pass_requires_R9_comparison' if passed else 'rejected_pilot_gate',
                  limitations=['Weak bias baseline; no claim of gain over R9.',
                               'Fixed random geometry, not learned K1 taste factors.',
                               'Retrospective training-only interactions; no Probe labels or identities.',
                               'Small permutation budget; shifted placebo support may differ.',
                               'No private-test labels; 10% target unverified.'])
    for arm, p in predictions.items():
        report['scores'][arm]['event_rmse'] = rmse(outer['rating'][covered], p[covered]) if covered.any() else None
    return report, dict(upstream=upstream, events=events, placebo=placebo, models=models,
                        split=split, predictions=predictions, outer=outer)


def export(r, state, output):
    """Full ordered column; accepts IDs/dates only. No sorting or clipping."""
    x = design(r, state['upstream'], state['events'], 'full')
    correction = x@state['models']['full']
    base = state['upstream'].base(r)
    np.asarray(base+correction, dtype='<f4').tofile(str(output)+'.f32')
    np.asarray(correction, dtype='<f4').tofile(str(output)+'.correction.f32')
    support = design(r, state['upstream'], state['events'], 'bias')
    confidence = support.max(1) if support.shape[1] else np.zeros(len(base))
    np.asarray(confidence, dtype='<f4').tofile(str(output)+'.confidence.f32')
    counts = np.zeros(len(base))
    for e in state['events']:
        mask = (r['item'] == e['movie']) & (r['week'] >= e['start']) & (r['week'] < e['end'])
        counts[mask] = np.maximum(counts[mask], min(e['before_users'], e['after_users']))
    np.asarray(counts, dtype='<f4').tofile(str(output)+'.support.f32')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', default='data/processed/netflix.bin')
    parser.add_argument('--output', required=True)
    parser.add_argument('--users', type=int, default=500)
    args = parser.parse_args()
    if args.users < 1:
        parser.error('--users must be positive')
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cfg = Config()
    source = pathlib.Path(args.data)
    before = source.stat()
    source_hash = digest(source)
    d = load(source)
    # Sample complete histories by deterministic user ID, not observed errors.
    selected = np.sort(np.random.default_rng(cfg.seed).choice(d['users'], min(args.users, d['users']), replace=False))
    ix = np.concatenate([np.arange(d['train_offsets'][u], d['train_offsets'][u+1], dtype=np.int64) for u in selected])
    owner = np.repeat(np.arange(len(selected)), np.diff(d['train_offsets'])[selected].astype(int))
    r = rows(owner, d['train_items'][ix].astype(int)-1, d['train_days'][ix], d['train_ratings'][ix])
    manifest = dict(config=dataclasses.asdict(cfg), input=str(source.resolve()), sha256=source_hash,
                    source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_hash=digest(__file__), numpy=np.__version__, selected_users=selected.tolist(),
                    baseline='fresh regularized user/item biases; weaker than R9-LAT-A2h',
                    split='seeded row split 60% upstream / 20% residual / 10% inner / 10% outer',
                    geometry='fixed seeded random item vectors; rating independent',
                    gate='full RMSE beats base, smooth, bias and placebo by >=0.0001',
                    budget='500 users by default, 19 permutations; no upstream checkpoint training')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    report, state = pilot(r, len(selected), d['items'], cfg)
    if (source.stat().st_size, source.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise RuntimeError('input changed during pilot; discard run')
    np.savez_compressed(out/'state.npz', split=state['split'], source_rows=ix,
                        **{f'beta_{k}': v for k, v in state['models'].items()})
    export(state['outer'], state, out/'outer')
    report['elapsed_seconds'] = time.monotonic()-started
    report['rows_per_second'] = len(ix)/report['elapsed_seconds']
    report['output_hashes'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'coverage', 'elapsed_seconds', 'rows_per_second')}))


if __name__ == '__main__':
    main()
