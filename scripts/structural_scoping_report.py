"""Audit and summarize every fixed S0 arm; never select or train a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from gbdt_blend import blend_fold, load
from local_context_chain import paired
from selection_objective_full import verify_blend
from selection_objective_queue import verify
from structural_scoping import arm_indices
from train_holdout import digest

SOURCES = ('structural_scoping_report.py', 'structural_scoping.py',
           'selection_objective_full.py', 'selection_objective_queue.py',
           'gbdt_blend.py', 'local_context_chain.py', 'train_holdout.py')


def read(path):
    return json.loads(Path(path).read_text())


def require_complete(arms):
    """Reject partial experiments before loading predictions or creating a report."""
    missing = []
    for arm in arms:
        if [s['kind'] for s in arm['stages']] != ['tree', 'neural']:
            raise ValueError('one tree and one neural stage required')
        for stage, suffixes in zip(arm['stages'],
                [('.json', '.refit.f32', '.ridge.f32'), ('.json', '.nn.f32', '.avg.f32')]):
            missing.extend(stage['config']['output']+s for s in suffixes
                           if not Path(stage['config']['output']+s).is_file())
    if missing:
        raise ValueError('incomplete fixed experiment: '+', '.join(missing))


def audit_arm(arm, plan):
    bound, reports = {}, []
    for stage in arm['stages']:
        bound.update(verify_blend(stage, plan))
        report = read(stage['config']['output']+'.json')
        if sorted(r['fold'] for r in report['folds']) != list(range(5)):
            raise ValueError('all five stage folds required')
        reports.append(report)
    if reports[0]['inputs'] != reports[1]['inputs']:
        raise ValueError('tree and neural input banks differ')
    tree = np.fromfile(arm['stages'][0]['config']['output']+'.refit.f32', '<f4').astype(float)
    prefix = arm['stages'][1]['config']['output']
    neural, average = [np.fromfile(prefix+s, '<f4').astype(float) for s in ('.nn.f32', '.avg.f32')]
    if not np.allclose(average, (tree+neural)/2, atol=1e-6, rtol=0):
        raise ValueError('saved average differs from fixed 50/50 components')
    return average, tree, neural, bound


def summarize(plan, arms, results):
    by_name = {a['name']: a for a in arms}
    if len(by_name) != len(arms) or set(results) != set(by_name):
        raise ValueError('exactly one result for every registered arm required')
    diversity = []
    for count in plan['diversity']['counts']:
        values = []
        for seed in plan['diversity']['permutation_seeds']:
            name = 'all79' if count == len(plan['columns']) else f'diversity-s{seed}-n{count}'
            values.append(dict(seed=seed, arm=name,
                               rmse=results[name]['comparison']['treatment_rmse']))
        scores = [v['rmse'] for v in values]
        diversity.append(dict(count=count, permutations=values, mean_rmse=float(np.mean(scores)),
            min_rmse=min(scores), max_rmse=max(scores),
            permutation_sd=float(np.std(scores)), shared_reference=count == len(plan['columns'])))
    quality = []
    for arm in arms:
        if arm['kind'] == 'quality':
            degradation = arm['standalone_degradation']
            if not np.isfinite(degradation) or degradation < 0:
                raise ValueError('invalid standalone degradation')
            comparison = results[arm['name']]['comparison']
            quality.append(dict(family=arm['family'], best=arm['best'], weaker=arm['weaker'],
                standalone_degradation=degradation, blend_degradation=comparison['delta_rmse'],
                ratio=comparison['delta_rmse']/degradation if degradation else None,
                fold_deltas=comparison['fold_deltas']))
    return dict(diversity=diversity, quality_substitutions=quality)


def run(execution_path, parity_path, binding_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    binding = read(binding_path)
    binding_sha = digest(binding_path)
    required = {str(Path(p).resolve()) for p in (execution_path, parity_path)}
    required.update(str(Path(__file__).resolve().with_name(name)) for name in SOURCES)
    if not required <= set(binding['frozen']):
        raise ValueError('reporter sources, execution and parity proof must be bound')
    verify(binding['frozen'])
    execution = read(execution_path)
    verify(execution['frozen'])
    plan = read(execution['plan'])
    if digest(execution['plan']) != execution['plan_sha256']:
        raise ValueError('protocol changed')
    inventory = read(execution['standalone_inventory'])
    if inventory['plan_sha256'] != execution['plan_sha256']:
        raise ValueError('standalone inventory protocol differs')
    definitions = arm_indices(plan, [c['rmse'] for c in inventory['columns']])
    if len(definitions) != len(execution['arms']):
        raise ValueError('registered arm count differs')
    for expected, arm in zip(definitions, execution['arms']):
        if any(arm[k] != v for k, v in expected.items()):
            raise ValueError('registered arm definition differs')
    if read(parity_path)['state'] != 'verified_full_reference_parity':
        raise ValueError('full reference parity required')
    require_complete(execution['arms'])
    data = load(execution['data'])
    y, users = data['probe']['rating'].astype(float), data['probe']['user']
    if len(y) != execution['rows'] or len(y) != plan['rows']:
        raise ValueError('row count differs')
    if set(blend_fold(users)) != set(range(5)):
        raise ValueError('all five user folds required')
    reference = np.fromfile(execution['reference_prediction'], '<f4').astype(float)
    if reference.shape != y.shape or not np.isfinite(reference).all():
        raise ValueError('invalid reference prediction')
    adapted = dict(execution, base_report=execution['reference_tree'],
                   neural_report=execution['reference_neural'])
    results, outputs = {}, {}
    for arm in execution['arms']:
        average, tree, neural, bound = audit_arm(arm, adapted)
        outputs.update(bound)
        if arm['kind'] == 'reference':
            for values, path in [(average, execution['reference_prediction']),
                    (tree, str(Path(execution['reference_tree']).with_suffix('.refit.f32'))),
                    (neural, str(Path(execution['reference_neural']).with_suffix('.nn.f32')))]:
                if not np.array_equal(values, np.fromfile(path, '<f4').astype(float)):
                    raise ValueError('full reference payload parity lost')
        results[arm['name']] = dict(tree_rmse=float(np.sqrt(np.mean((np.clip(tree, 1, 5)-y)**2))),
            neural_rmse=float(np.sqrt(np.mean((np.clip(neural, 1, 5)-y)**2))),
            comparison=paired(y, reference, average, users))
    summary = summarize(plan, execution['arms'], results)
    verify(execution['frozen']); verify(binding['frozen']); verify(outputs)
    if digest(binding_path) != binding_sha:
        raise ValueError('reporter binding changed')
    result = dict(state='complete_all_fixed_arms_audited', execution_sha256=digest(execution_path),
        binding_sha256=binding_sha, outputs=outputs, arms=results, **summary,
        scope='Reused original-probe development diagnostics, fixed support/context retained. '
              'Observed substitution ratios are not causal derivatives, future-gain forecasts or ceilings. '
              'The all-column reference is shared across permutations, not three independent runs. '
              'No subset selection, extrapolation, training, automatic promotion or hidden-score claim.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('execution', 'parity', 'binding', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(run(args.execution, args.parity, args.binding, args.output)['state'])
