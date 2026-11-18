#!/usr/bin/env python3
"""Diagnostic released-base additions to the frozen fresh context control.

Released columns never become fresh model features. Each single/bundle refits
only its final regression in the fixed user folds; ranking is exploratory.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
import re
import numpy as np
import upstream_repro as u
import upstream_exchange as x
import upstream_expansion as e
import upstream_snapshot_blend as b

REPORT = u.ROOT/'experiments/upstream-missing-screen'
SETUP = {'download', 'ingest', 'newsplit'}


def producer_for(name, jobs):
    """Prefer native declared outputs; accept only known auxiliary suffixes."""
    path = 'preds_new/'+name+'.probex.npy'
    explicit = [key for key, job in jobs.items() if path in job['outputs']]
    if len(explicit) > 1:
        raise ValueError('Ambiguous native producer: '+name)
    if explicit:
        return explicit[0], 'declared native output'
    matches = [key for key in jobs if name.startswith(key) and
               re.fullmatch(r'(?:_ep\d+|-(?:bias|mf|nsvd1|p4))', name[len(key):])]
    if not matches:
        return None, 'unresolved; do not infer a historical recipe from a similar name'
    key = max(matches, key=len)
    return key, 'auxiliary epoch/component suffix; verify dispatcher emission before launch'


def solve_subsets(grams, rhs, base_width, arms):
    """Penalty keeps the baseline intercept unpenalized after adding columns."""
    total, target = grams.sum(axis=0), rhs.sum(axis=0)
    width = rhs.shape[1]
    result = np.zeros((len(grams), width, len(arms)))
    for j, columns in enumerate(arms.values()):
        ix = list(range(base_width))+columns
        penalty = np.eye(len(ix))*1000
        penalty[base_width-1, base_width-1] = 0
        for f in range(len(grams)):
            result[f, ix, j] = np.linalg.solve((total-grams[f])[np.ix_(ix, ix)]+penalty,
                                               (target-rhs[f])[ix])
    return result


def score_arms(block, labels, folds, base_width, arms):
    width = block(0, 1).shape[1]
    grams, rhs = np.zeros((5, width, width)), np.zeros((5, width))
    for lo in range(0, len(labels), 8192):
        hi = min(lo+8192, len(labels)); z = block(lo, hi)
        for f in range(5):
            mask = folds[lo:hi] == f; a = z[mask]
            grams[f] += a.T@a
            rhs[f] += a.T@labels[lo:hi][mask]
    weights = solve_subsets(grams, rhs, base_width, arms)
    sse = np.zeros((5, len(arms)))
    for lo in range(0, len(labels), 8192):
        hi = min(lo+8192, len(labels)); z = block(lo, hi)
        for f in range(5):
            mask = folds[lo:hi] == f
            pred = np.clip(z[mask]@weights[f], 1.05, 4.95)
            sse[f] += np.sum((pred-labels[lo:hi][mask, None])**2, axis=0)
    counts = np.bincount(folds, minlength=5)
    fold_scores = np.sqrt(sse/counts[:, None])
    scores = np.sqrt(sse.sum(axis=0)/len(labels))
    return [dict(arm=name, released_columns=len(columns), rmse=float(scores[j]),
                 fold_rmse=fold_scores[:, j].tolist(), delta_vs_fresh=float(scores[j]-scores[0]),
                 fold_delta_vs_fresh=(fold_scores[:, j]-fold_scores[:, 0]).tolist())
            for j, (name, columns) in enumerate(arms.items())]


def main():
    os.nice(10)
    labels, users, fresh = e.collect_expansion()
    folds = x.user_folds(users)
    frozen = u.read(b.REPORT/'manifest.json')
    # Regenerate and check all receipt-backed saved output entries.
    current = b.freeze(labels, users, fresh)
    if any(current[k] != frozen[k] for k in current):
        raise ValueError('Fresh bank differs from frozen snapshot control')
    for path, digest in frozen['context_inputs'].items():
        if u.sha(u.ROOT/path) != digest:
            raise ValueError('Context input changed: '+path)
    context = b.context_features(*[np.load(u.ROOT/p, mmap_mode='r', allow_pickle=False)
                                   for p in frozen['context_inputs']])
    if x.array_sha(context) != frozen['context_content_sha256']:
        raise ValueError('Context features differ')
    selected = [r for r in frozen['entries'] if r['duplicate_of'] is None]
    arrays = [fresh[r['name']] for r in selected]
    clip = [r['clip'] for r in selected]
    baseline, base_width = b.evaluate(arrays, clip, labels, folds, context, tuple(range(14)))
    if abs(x.metric(baseline, labels)-0.8663200739003126) > 1e-9:
        raise ValueError('Snapshot/context control does not reproduce')
    print('Fresh control reproduced', x.metric(baseline, labels), flush=True)
    registry = u.read(u.BASE/'registry.json'); jobs = registry['jobs']
    reference = u.read(u.BASE/'reference-manifest.json')
    if reference['source_commit'] != u.PIN:
        raise ValueError('Wrong released source commit')
    base_indices = sorted({i for group in ['tx', 'rbm', 'nn', 'other'] for i in reference['groups'][group]})
    entries, released = [], []
    bundles = {}
    present = {r['name'] for r in frozen['entries']}
    for i in base_indices:
        entry = reference['entries'][i]
        if entry['name'] in present:
            continue
        path = u.REFERENCE/entry['path']
        if u.sha(path) != entry['sha256']:
            raise ValueError('Released fixture changed: '+entry['name'])
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if value.shape != labels.shape or not np.isfinite(value).all():
            raise ValueError('Invalid released vector')
        owner, reason = producer_for(entry['name'], jobs)
        root = owner.split('__')[0] if owner else 'unresolved:'+entry['name']
        record = dict(**entry, clip=reference['clip'][i], producer=owner, producer_basis=reason,
                      bundle=root, full_dependency_order=u.dependency_order(jobs, owner) if owner else [],
                      historical_epoch_coefficients=bool(owner and owner.endswith('__epochs')))
        index = base_width+len(entries)
        bundles.setdefault(root, []).append(index)
        entries.append(record); released.append(value)
    arms = {'fresh-control': []}
    arms.update({'single:'+r['name']: [base_width+i] for i, r in enumerate(entries)})
    arms.update({'bundle:'+name: indices for name, indices in bundles.items() if len(indices)>1})
    manifest = dict(source_commit=u.PIN, baseline_manifest_sha256=u.sha(b.REPORT/'manifest.json'),
                    reference_manifest_sha256=u.sha(u.BASE/'reference-manifest.json'),
                    registry_sha256=u.sha(u.BASE/'registry.json'),
                    pipeline_sha256=u.sha(u.SOURCE/'pipeline-new.toml'),
                    base_width=base_width, entries=entries, arms=arms,
                    recipe='Fixed context interactions on fresh fourteen only; each released column enters as a global linear feature, with native per-column clipping. Lambda 1000; intercept unpenalized; five fixed user folds.',
                    scope='Exploratory conditional released-vector screen; NOT independently achieved performance. Ranking and group construction use the same development set; no selection-adjusted performance claim.')
    path = REPORT/'manifest.json'
    if path.exists() and u.read(path) != manifest:
        raise ValueError('Frozen diagnostic screen changed')
    u.write(path, manifest)
    def block(lo, hi):
        known = b.design(np.column_stack([a[lo:hi] for a in arrays]), clip,
                         context[lo:hi], tuple(range(14)))
        missing = np.column_stack([a[lo:hi] for a in released]).astype(np.float64)
        mask = [r['clip'] for r in entries]
        missing[:, mask] = np.clip(missing[:, mask], 0, 6)
        return np.column_stack([known, missing])
    records = score_arms(block, labels, folds, base_width, arms)
    if abs(records[0]['rmse']-x.metric(baseline, labels)) > 1e-9:
        raise ValueError('Diagnostic control differs from fresh evaluation')
    result = dict(scope=manifest['scope'], rows=len(labels), manifest_sha256=u.sha(path),
                  records=records, ranking=sorted(records[1:], key=lambda r:r['rmse']))
    u.write(REPORT/'evaluation.json', result)
    for r in result['ranking'][:20]:
        print(r['arm'], r['rmse'], r['delta_vs_fresh'], flush=True)


if __name__ == '__main__':
    main()
