#!/usr/bin/env python3
"""Fresh saved-output ablations and one-layer, user-fold-safe context blends."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
import numpy as np
import upstream_repro as u
import upstream_expansion as e
import upstream_exchange as x

REPORT = u.ROOT/'experiments/upstream-snapshot-blend'
RUN = u.ROOT/'artifacts/upstream-snapshot-blend'
CONTEXT = ['user_log_count', 'movie_log_count', 'query_date', 'user_relative_date', 'movie_relative_date']


def aggregate_dates(ids, dates, size):
    count = np.bincount(ids, minlength=size)
    total = np.bincount(ids, weights=dates, minlength=size)
    return count, np.divide(total, count, out=np.zeros(size), where=count > 0)


def context_features(train_users, train_items, train_dates, users, items, dates):
    """Only trainx identities/dates and query metadata; no outcome argument."""
    columns = []
    for source, query in [(train_users, users), (train_items, items)]:
        count, mean = aggregate_dates(source, train_dates, int(max(source.max(), query.max()))+1)
        columns.append((count[query], mean[query]))
    (uc, ud), (ic, idate) = columns
    result = np.column_stack([np.log1p(uc)/7-1, np.log1p(ic)/10-1,
                              dates/3000-0.5, (dates-ud)/1000, (dates-idate)/1000])
    if not np.isfinite(result).all():
        raise ValueError('Non-finite metadata context')
    return result


def design(matrix, clip, context=None, interaction_columns=()):
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    matrix[:, clip] = np.clip(matrix[:, clip], 0, 6)
    parts = [matrix]
    if context is not None:
        parts.append(context)
        # Fixed centering changes the penalty parameterization, not label use.
        parts.extend([(matrix[:, interaction_columns]-3.6)*context[:, j:j+1]
                      for j in range(context.shape[1])])
    parts.append(np.ones((len(matrix), 1)))
    return np.column_stack(parts)


def fit_folds(grams, rhs, penalty=1000):
    regularizer = np.eye(rhs.shape[1])*penalty
    regularizer[-1, -1] = 0
    total, target = grams.sum(axis=0), rhs.sum(axis=0)
    return [np.linalg.solve(total-grams[f]+regularizer, target-rhs[f]) for f in range(len(grams))]


def evaluate(arrays, clip, labels, folds, context=None, interaction_columns=()):
    def block(lo, hi):
        return design(np.column_stack([a[lo:hi] for a in arrays]), clip,
                      None if context is None else context[lo:hi], interaction_columns)
    width = block(0, 1).shape[1]
    grams, rhs = np.zeros((5, width, width)), np.zeros((5, width))
    for lo in range(0, len(labels), 8192):
        hi = min(lo+8192, len(labels)); z = block(lo, hi)
        for f in range(5):
            mask = folds[lo:hi] == f; a = z[mask]
            grams[f] += a.T@a
            rhs[f] += a.T@labels[lo:hi][mask]
    weights = fit_folds(grams, rhs)
    prediction = np.empty(len(labels))
    for lo in range(0, len(labels), 8192):
        hi = min(lo+8192, len(labels)); z = block(lo, hi)
        for f in range(5):
            mask = folds[lo:hi] == f
            prediction[lo:hi][mask] = np.clip(z[mask]@weights[f], 1.05, 4.95)
    return prediction, width


def freeze(labels, users, fresh):
    entries = []
    for name in e.ARMS['all-fourteen']:
        entries.append(dict(name=name, kind='base', clip=True, content_sha256=x.array_sha(fresh[name]),
                            provenance='collect_expansion validates frozen original/expansion artifacts and receipts'))
    for parent, epochs, components in [('tsvdx4-60', 20, []),
            ('tsvdx4-400lm', 20, ['bias', 'mf', 'nsvd1']), ('dnn-64', 5, ['base', 'net'])]:
        receipt_path = u.BASE/'fresh-receipts'/(parent+'.json')
        receipt = u.read(receipt_path)
        if receipt['source_commit'] != u.PIN or receipt['job'] != parent:
            raise ValueError('Incorrect parent receipt')
        for name, kind in ([(f'{parent}_ep{i:02}', 'snapshot') for i in range(1, epochs+1)] +
                           [(parent+'-'+c, 'component') for c in components]):
            path = u.FRESH/'preds_new'/(name+'.probex.npy')
            if u.sha(path) != receipt['outputs'][str(path.relative_to(u.FRESH))]:
                raise ValueError('Saved output changed: '+name)
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            if a.shape != labels.shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('Invalid saved output: '+name)
            fresh[name] = a
            entries.append(dict(name=name, kind=kind, clip=kind!='component', parent=parent,
                                path=str(path.relative_to(u.ROOT)), sha256=u.sha(path),
                                receipt_sha256=u.sha(receipt_path), content_sha256=x.array_sha(a)))
    seen = {}
    for entry in entries:
        # Preserve semantic clipping even if raw values happen to match.
        key = (entry['content_sha256'], entry['clip'])
        entry['duplicate_of'] = seen.get(key)
        seen.setdefault(key, entry['name'])
    manifest = dict(source_commit=u.PIN, entries=entries, rows=len(labels),
                    labels_sha256=x.array_sha(labels), users_sha256=x.array_sha(users),
                    folds_sha256=x.array_sha(x.user_folds(users)),
                    ridge_lambda=1000, output_clip=[1.05, 4.95], fold_seed=20261116,
                    context_names=CONTEXT, interactions='fourteen original rating columns only; centered at fixed 3.6',
                    normalization='fixed: log1p(user count)/7-1, log1p(movie count)/10-1, date/3000-.5, query-minus-training-mean-date/1000',
                    scope='Meta-level user CV on historically selected development base recipes; not untouched test evidence')
    return manifest


def main():
    os.nice(10)
    labels, users, fresh = e.collect_expansion()
    folds = x.user_folds(users)
    RUN.mkdir(parents=True, exist_ok=True)
    # Control must pass before accessing the additional features.
    names = e.ARMS['all-fourteen']
    control, _ = evaluate([fresh[n] for n in names], [True]*14, labels, folds)
    if abs(x.metric(control, labels)-0.8677119362283443) > 1e-9:
        raise ValueError('Fourteen-bank control failed')
    print('Fourteen-bank control reproduced', x.metric(control, labels), flush=True)
    manifest = freeze(labels, users, fresh)
    train, query = u.FRESH/'data/trainx', u.FRESH/'data/probex'
    paths = [train/(n+'.npy') for n in ['user_idxs','item_idxs','dates']]
    paths += [query/(n+'.npy') for n in ['user_idxs','item_idxs','dates']]
    manifest['context_inputs'] = {str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
    data = [np.load(p, mmap_mode='r', allow_pickle=False) for p in paths]
    if not np.array_equal(data[3], users):
        raise ValueError('Query user ordering differs')
    context = context_features(*data)
    manifest['context_content_sha256'] = x.array_sha(context)
    frozen = REPORT/'manifest.json'
    if frozen.exists() and u.read(frozen) != manifest:
        raise ValueError('Frozen experiment inputs/recipe changed')
    u.write(frozen, manifest)
    arms = {'fourteen': ['base'], 'snapshots': ['base','snapshot'],
            'components': ['base','component'], 'combined': ['base','snapshot','component'],
            'fourteen-context': ['base'], 'combined-context': ['base','snapshot','component']}
    records = []
    for arm, kinds in arms.items():
        selected = [r for r in manifest['entries'] if r['kind'] in kinds and r['duplicate_of'] is None]
        is_context = arm.endswith('-context')
        prediction, width = evaluate([fresh[r['name']] for r in selected], [r['clip'] for r in selected],
                                    labels, folds, context if is_context else None, tuple(range(14)))
        dest = RUN/(arm+'.f32'); prediction.astype('<f4').tofile(dest)
        record = dict(arm=arm, prediction_columns=len(selected), design_columns=width,
                      columns=[r['name'] for r in selected], rmse=x.metric(prediction, labels),
                      fold_rmse=[x.metric(prediction[folds==f], labels[folds==f]) for f in range(5)],
                      prediction_sha256=u.sha(dest), delta_vs_fourteen=x.metric(prediction, labels)-x.metric(control, labels),
                      fold_delta_vs_fourteen=[x.metric(prediction[folds==f], labels[folds==f])-x.metric(control[folds==f], labels[folds==f]) for f in range(5)])
        records.append(record)
        u.write(REPORT/'evaluation.json', dict(manifest_sha256=u.sha(frozen), records=records,
                   complete=len(records)==len(arms), scope=manifest['scope'],
                   isolation='Single learned layer; fold weights exclude all scoring-user ProbeX labels. Context uses trainx identities/dates and query metadata only. No ProbeX/qualifying outcomes or released prediction features. No target-dependent preprocessing or published epoch coefficients.'))
        print(arm, record['rmse'], record['fold_delta_vs_fourteen'], flush=True)


if __name__ == '__main__':
    main()
