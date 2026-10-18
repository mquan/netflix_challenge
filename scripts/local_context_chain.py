#!/usr/bin/env python3
"""Matched pooled/path additions to the frozen R9 ensemble, with paired scoring.

No weights are fit to saved out-of-fold blend outputs. Each arm repeats the
original nested ridge/LightGBM and neural recipes on the original raw bank.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from gbdt_blend import blend_fold, load
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path('/Users/quan/workspace/netflix_round9')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def resolve(name, config, root):
    for candidate in (root / name, root / config['bank_root'] / name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(name)


def paired(y, control, treatment, users):
    if not (y.shape == control.shape == treatment.shape == users.shape) or not len(y):
        raise ValueError('paired scores require nonempty matching one-dimensional arrays')
    if y.ndim != 1 or not all(np.isfinite(x).all() for x in (y, control, treatment)):
        raise ValueError('nonfinite or non-vector predictions')
    control, treatment = np.clip(control, 1, 5), np.clip(treatment, 1, 5)
    ce, te = (y - control) ** 2, (y - treatment) ** 2
    delta = te - ce
    _, cluster = np.unique(users, return_inverse=True)
    counts = np.bincount(cluster)
    centered = np.bincount(cluster, delta) - delta.mean() * counts
    groups = len(counts)
    interval = None
    if groups > 1:
        se = np.sqrt(groups / (groups - 1) * np.sum(centered ** 2)) / len(delta)
        interval = [float(delta.mean() - 1.96 * se), float(delta.mean() + 1.96 * se)]
    folds = blend_fold(users)
    changes = [float(np.sqrt(te[folds == f].mean()) - np.sqrt(ce[folds == f].mean()))
               if np.any(folds == f) else None for f in range(5)]
    return dict(rows=len(y), users=groups, control_rmse=float(np.sqrt(ce.mean())),
                treatment_rmse=float(np.sqrt(te.mean())),
                delta_rmse=float(np.sqrt(te.mean()) - np.sqrt(ce.mean())),
                fold_deltas=changes, delta_mse=float(delta.mean()),
                cluster_normal_95_mse_interval=interval,
                all_folds_improve=all(x is not None and x < 0 for x in changes))


def validate_features(manifest, data_sha, rows):
    if manifest['data_sha256'] != data_sha:
        raise ValueError('features belong to a different dataset')
    if manifest['query_rows'] != rows or manifest['full_query_rows'] != rows:
        raise ValueError('features do not cover the full probe')
    if manifest['settings'] != dict(history=32, neighbors=32, candidates=256, seed=20261013):
        raise ValueError('retrieval recipe differs from the predeclared LP2 settings')
    features = manifest['features']
    if len(features['pooled']) != 27 or len(features['paths']) != 133:
        raise ValueError('unexpected feature dimensions')
    listed = features['pooled'] + features['paths']
    if len(set(listed)) != 160 or set(listed) != set(manifest['outputs']):
        raise ValueError('feature manifest has missing or duplicate columns')
    for name in listed:
        path = Path(name)
        if path.stat().st_size != rows * 4 or digest(path) != manifest['outputs'][name]:
            raise ValueError(f'feature changed: {name}')
        if not np.isfinite(np.fromfile(path, dtype='<f4')).all():
            raise ValueError(f'nonfinite feature: {name}')


def preflight(a):
    import lightgbm
    import torch
    root = Path(a.reference_root).resolve()
    base_path, control_path = Path(a.base).resolve(), Path(a.control).resolve()
    base, control = read(base_path), read(control_path)
    if (root / control['config']['like']).resolve() != base_path:
        raise ValueError('neural control does not refer to the supplied tree control')
    if base['inputs'] != control['inputs']:
        raise ValueError('reference tree and neural inputs differ')
    if base['config']['no_union']:
        raise ValueError('this experiment requires the promoted full reference bank')
    if lightgbm.__version__ != base['lightgbm_version'] or torch.__version__ != control['torch_version']:
        raise ValueError('blender package versions differ from the reference')
    config = base['config']
    frozen = {}
    for name, expected in base['inputs'].items():
        path = resolve(name, config, root)
        if digest(path) != expected:
            raise ValueError(f'reference input changed: {name}')
        frozen[str(path)] = expected
    data = (root / config['data']).resolve()
    data_sha = digest(data)
    historical = read(root / 'experiments' / (base_path.stem + '.json'))
    if historical['inputs'][config['data']]['sha256'] != data_sha:
        raise ValueError('reference dataset changed since training')
    cache = ROOT / 'artifacts/r4/meta-context.npz'
    if digest(cache) != digest(root / 'artifacts/r4/meta-context.npz'):
        raise ValueError('local meta-context cache differs from reference')
    frozen[str(data)] = data_sha
    frozen[str(cache)] = digest(cache)
    for relative in ('experiments/UNION-round-manifest.json', 'experiments/BOOST-support-inputs.json'):
        path = (root / config['bank_root'] / relative).resolve()
        frozen[str(path)] = digest(path)
    for path in (base_path, control_path):
        frozen[str(path)] = digest(path)
    for suffix in ('.avg.f32', '.nn.f32'):
        path = control_path.with_suffix(suffix)
        frozen[str(path)] = digest(path)
    frozen[str(base_path.with_suffix('.refit.f32'))] = digest(base_path.with_suffix('.refit.f32'))
    d = load(data)
    y, users = d['probe']['rating'].astype(float), d['probe']['user']
    baseline = np.fromfile(control_path.with_suffix('.avg.f32'), dtype='<f4').astype(float)
    score = paired(y, baseline, baseline, users)
    if abs(score['control_rmse'] - control['avg_rmse']) > 1e-8:
        raise ValueError('saved reference predictions do not reproduce the recorded score')
    features_path = Path(a.features).resolve()
    feature_manifest = read(features_path)
    validate_features(feature_manifest, data_sha, len(y))
    frozen[str(features_path)] = digest(features_path)
    frozen.update(feature_manifest['outputs'])
    # Freeze the source used by both treatment arms. experiment.py also archives it.
    for name in ('local_context_chain.py', 'round4_boost.py', 'round8_nn_blend.py',
                 'gbdt_blend.py', 'moonshot_nn_blend.py'):
        path = ROOT / 'scripts' / name
        frozen[str(path)] = digest(path)
    audit = dict(base_report=str(base_path), control_report=str(control_path),
                 data=str(data), data_sha256=data_sha, feature_manifest=str(features_path),
                 reference_rmse=score['control_rmse'], reference_input_count=len(base['inputs']),
                 frozen=frozen, numpy_version=np.__version__, lightgbm_version=lightgbm.__version__,
                 torch_version=torch.__version__,
                 source_note='round4_boost differs from reference only in the inactive no_union branch '
                             'and creation of a missing cache directory; no_union=False and cache exists.')
    return base, control, feature_manifest, audit


def arguments(script, config):
    command = [sys.executable, str(ROOT / 'scripts' / script)]
    for name, value in config.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif isinstance(value, list):
            command += [flag, *map(str, value)]
        else:
            command += [flag, str(value)]
    return command


def stages(a, base, control, features):
    root = Path(a.reference_root).resolve()
    output = Path(a.output).resolve()
    result = []
    for arm, added in [('pooled', features['pooled']),
                       ('paths', features['pooled'] + features['paths'])]:
        config = dict(base['config'])
        config['extra'] = [str(resolve(x, config, root)) for x in config['extra']]
        config['extra_support'] = [str(resolve(x, config, root)) for x in config['extra_support']] + added
        config['data'] = str((root / config['data']).resolve())
        config['bank_root'] = str((root / config['bank_root']).resolve())
        config['processes'] = a.blend_processes
        config['output'] = str(output / (arm + '-s3'))
        result.append((f'{a.run_id}-{arm}-s3', arguments('round4_boost.py', config)))
        neural = dict(control['config'])
        neural.update(data=config['data'], like=config['output'] + '.json',
                      output=str(output / (arm + '-NN2')))
        result.append((f'{a.run_id}-{arm}-NN2', arguments('round8_nn_blend.py', neural)))
    return result


def check_frozen(audit):
    for path, expected in audit['frozen'].items():
        if digest(path) != expected:
            raise ValueError(f'frozen input or source changed: {path}')


def summarize_run(a, audit):
    output = Path(a.output).resolve()
    d = load(audit['data'])
    y, users = d['probe']['rating'].astype(float), d['probe']['user']
    paths = dict(reference=str(Path(audit['control_report']).with_suffix('.avg.f32')),
                 pooled=str(output / 'pooled-NN2.avg.f32'), paths=str(output / 'paths-NN2.avg.f32'))
    predictions = {k: np.fromfile(v, dtype='<f4').astype(float) for k, v in paths.items()}
    counts = np.diff(d['train_offsets'])[users]
    comparisons = {}
    for name, left, right in [('pooled_vs_reference', 'reference', 'pooled'),
                              ('paths_vs_reference', 'reference', 'paths'),
                              ('paths_vs_pooled', 'pooled', 'paths')]:
        comparison = paired(y, predictions[left], predictions[right], users)
        comparison['slices'] = {}
        for slice_name, keep in [('history_le_20', counts <= 20), ('history_gt_20', counts > 20)]:
            if keep.any():
                comparison['slices'][slice_name] = paired(y[keep], predictions[left][keep],
                                                         predictions[right][keep], users[keep])
        comparisons[name] = comparison
    evidence = comparisons['paths_vs_pooled']
    baseline = comparisons['paths_vs_reference']
    report = dict(metric='Original-probe development user-fold CV, not a hidden-test score',
                  comparisons=comparisons,
                  outputs={path: digest(path) for path in paths.values()},
                  sources={str(output / (arm + '-NN2.json')): digest(output / (arm + '-NN2.json'))
                           for arm in ('pooled', 'paths')},
                  path_arm_passes_promotion_rule=baseline['all_folds_improve'],
                  path_information_passes_all_folds=evidence['all_folds_improve'],
                  substantial_investment_gate=(baseline['all_folds_improve']
                      and baseline['delta_rmse'] <= -.0003 and evidence['all_folds_improve']),
                  target=dict(cinematch_test_rmse=.9525, improvement_fraction=.10,
                              target_test_rmse=.85725, estimated_retraining_offset=.0074,
                              approximate_dev_target=.86465,
                              path_arm_estimated_test_rmse=baseline['treatment_rmse'] - .0074,
                              path_arm_dev_gap=baseline['treatment_rmse'] - .86465),
                  limitations=['Probe-guided development selection remains.',
                               'Paired intervals do not adjust for model selection or shared fitted weights.',
                               'Feature additions also change random feature sampling and NN input dimension.',
                               'This tests summaries, not a structure-preserving encoder.',
                               'The calibrated target is not a measured hidden-test result.'])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', default='LC1')
    p.add_argument('--reference-root', default=str(REFERENCE))
    p.add_argument('--base', default=str(REFERENCE / 'artifacts/r9/R9-LAT-A2h-20261010-s3.json'))
    p.add_argument('--control', default=str(REFERENCE / 'artifacts/r9/R9-LAT-A2h-20261010-NN2.json'))
    p.add_argument('--features', default='artifacts/local-context/LC1/features/manifest.json')
    p.add_argument('--output', default='artifacts/local-context/LC1')
    p.add_argument('--blend-processes', type=int, default=3)
    p.add_argument('--check', action='store_true')
    p.add_argument('--resume', action='store_true', help='reuse only verified successful stages of this run')
    a = p.parse_args()
    os.chdir(ROOT)
    if Path(a.run_id).name != a.run_id or a.blend_processes < 1:
        p.error('run-id must be a filename and blend-processes positive')
    base, control, features, audit = preflight(a)
    plan = stages(a, base, control, features['features'])
    print(json.dumps(dict(reference_rmse=audit['reference_rmse'], inputs=audit['reference_input_count'],
                          stages=[name for name, _ in plan])), flush=True)
    if a.check:
        return
    output = Path(a.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_path = Path('experiments') / (a.run_id + '-audit.json')
    state_path = Path('experiments') / (a.run_id + '-chain.status.json')
    if audit_path.exists():
        if not a.resume:
            raise FileExistsError('run exists; use --resume only after its process has stopped')
        previous = read(audit_path)
        if previous != dict(audit=audit, stages=plan):
            # JSON stores the plan tuples as lists.
            if previous != json.loads(json.dumps(dict(audit=audit, stages=plan))):
                raise ValueError('resume audit or commands differ from the recorded run')
    else:
        for name, command in plan:
            prefix = Path(command[command.index('--output') + 1])
            if list(Path('experiments').glob(name + '.*')) or list(prefix.parent.glob(prefix.name + '.*')):
                raise FileExistsError(name)
        with audit_path.open('x') as stream:
            json.dump(dict(audit=audit, stages=plan), stream, indent=2)
            stream.write('\n')
    # O_EXCL lock prevents overlapping launches, including --resume.
    lock = output / 'chain.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    completed = []

    def status(**values):
        write(state_path, dict(pid=os.getpid(), completed=completed,
                               updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values))

    try:
        for name, command in plan:
            check_frozen(audit)
            experiment = Path('experiments') / (name + '.json')
            receipt = output / (name + '-receipt.json')
            prefix = Path(command[command.index('--output') + 1])
            if a.resume and receipt.exists():
                if read(experiment).get('exit_code') != 0 or read(experiment)['command'] != command:
                    raise ValueError(f'invalid completed experiment: {name}')
                for path, expected in read(receipt).items():
                    if digest(path) != expected:
                        raise ValueError(f'completed output changed: {path}')
            else:
                status(state='running', stage=name)
                subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                                '--strategy', 'LC1 matched pooled/path features on frozen R9 raw bank; '
                                'unchanged nested three-seed blending and fixed 50/50 average',
                                '--', *command], check=True)
                files = list(prefix.parent.glob(prefix.name + '.*'))
                if not prefix.with_suffix('.json').exists():
                    raise FileNotFoundError(str(prefix) + '.json')
                write(receipt, {str(path): digest(path) for path in files})
            completed.append(name)
        check_frozen(audit)
        report = summarize_run(a, audit)
        write(output / 'comparison.json', report)
        write(Path('experiments') / (a.run_id + '-comparison.json'), report)
        status(state='complete', comparison=str(output / 'comparison.json'),
               path_arm_passes_promotion_rule=report['path_arm_passes_promotion_rule'],
               comparisons=report['comparisons'])
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        status(state='failed', error=repr(error))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
