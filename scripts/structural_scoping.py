"""Prepare S0's predeclared arms; never train or select a winning subset.

Each prediction occurrence gets a distinct symlink, so replacing a column
with an existing sibling preserves the duplicate and the original width.
The frozen blenders load these through ordinary union/support manifests.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gbdt_blend import blend_fold, load
from local_context_chain import arguments
from selection_objective_queue import verify
from train_holdout import digest


def arm_indices(plan, scores):
    n = len(plan['columns'])
    if len(scores) != n or not np.isfinite(scores).all():
        raise ValueError('one finite standalone score per column required')
    families = plan['families']
    if sorted(i for group in families.values() for i in group) != list(range(n)):
        raise ValueError('families must partition the bank')
    arms = [dict(name='all79', kind='reference', indices=list(range(n)))]
    for seed in plan['diversity']['permutation_seeds']:
        permutation = np.random.default_rng(seed).permutation(n)
        for count in plan['diversity']['counts']:
            if count < 1 or count > n:
                raise ValueError('invalid subset size')
            if count != n:
                arms.append(dict(name=f'diversity-s{seed}-n{count}', kind='diversity',
                    seed=seed, count=count, indices=sorted(permutation[:count].tolist())))
    for family, group in families.items():
        best = min(group, key=lambda i: (scores[i], i))
        weaker = min(group, key=lambda i: (-scores[i], i))
        indices = list(range(n))
        indices[best] = weaker
        arms.append(dict(name='quality-'+family, kind='quality', family=family,
            best=best, weaker=weaker, standalone_degradation=float(scores[weaker]-scores[best]),
            indices=indices))
    return arms


def write_bank(directory, columns, indices, support):
    directory = Path(directory)
    (directory/'columns').mkdir(parents=True)
    (directory/'support').mkdir()
    (directory/'experiments').mkdir()
    banks = []
    for kind, records in [('columns', [columns[i] for i in indices]), ('support', support)]:
        manifest = {}
        for position, record in enumerate(records):
            link = directory/kind/f'{position:03d}.f32'
            link.symlink_to(Path(record['path']).resolve())
            manifest[str(link)] = record['sha256']
        banks.append(manifest)
    paths = [directory/'experiments/UNION-round-manifest.json',
             directory/'experiments/BOOST-support-inputs.json']
    paths[0].write_text(json.dumps(dict(reports=dict(union=dict(inputs=banks[0]))), indent=2)+'\n')
    paths[1].write_text(json.dumps(banks[1], indent=2)+'\n')
    return {str(p): digest(p) for p in paths}


def prepare(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    verify(plan['frozen'])
    plan_sha = digest(plan_path)
    d = load(plan['data'])
    y, users = d['probe']['rating'].astype(float), d['probe']['user']
    if len(y) != plan['rows']:
        raise ValueError('row count differs from plan')
    folds = blend_fold(users)
    if set(folds) != set(range(5)):
        raise ValueError('all five user folds required')
    inventory = []
    for record in plan['columns']:
        if digest(record['path']) != record['sha256']:
            raise ValueError('column changed: '+record['path'])
        values = np.fromfile(record['path'], '<f4').astype(float)
        if values.shape != y.shape or not np.isfinite(values).all():
            raise ValueError('invalid standalone prediction')
        error = (np.clip(values, 1, 5)-y)**2
        inventory.append(dict(record, rmse=float(np.sqrt(error.mean())),
            fold_rmse=[float(np.sqrt(error[folds == f].mean())) for f in range(5)]))
    definitions = arm_indices(plan, [r['rmse'] for r in inventory])
    output.mkdir(parents=True)
    frozen = dict(plan['frozen'])
    frozen[str(plan_path)] = plan_sha
    frozen[str(Path(__file__).resolve())] = digest(__file__)
    arms = []
    for definition in definitions:
        bank = output/'banks'/definition['name']
        frozen.update(write_bank(bank, plan['columns'], definition['indices'], plan['fixed_tree_support']))
        prefix = output/'results'/definition['name']
        prefix.parent.mkdir(exist_ok=True)
        tree = dict(plan['recipe']['tree'], data=plan['data'], bank_root=str(bank),
            no_union=False, extra=[], extra_support=[], output=str(prefix)+'-s3')
        neural = dict(plan['recipe']['neural'], data=plan['data'],
            like=tree['output']+'.json', output=str(prefix)+'-NN2')
        stages = [dict(kind=kind, config=config, command=arguments(script, config))
                  for kind, config, script in [('tree', tree, 'round4_boost.py'),
                                               ('neural', neural, 'round8_nn_blend.py')]]
        arms.append(dict(definition, bank_root=str(bank), stages=stages))
    verify(plan['frozen'])
    if digest(plan_path) != plan_sha:
        raise ValueError('scoping plan changed during preparation')
    inventory_path = output/'standalone.json'
    inventory_path.write_text(json.dumps(dict(plan_sha256=plan_sha, columns=inventory,
        scope='Original-probe development scores under predeclared S0 family selection; no new blend score.'), indent=2)+'\n')
    frozen[str(inventory_path)] = digest(inventory_path)
    execution = dict(state='prepared_reference_parity_required', plan=str(plan_path),
        plan_sha256=plan_sha, source_sha256=digest(__file__), arms=arms, frozen=frozen,
        standalone_inventory=str(inventory_path), rows=plan['rows'], data=plan['data'],
        reference_prediction=plan['reference_prediction'],
        reference_tree=plan['reference_tree'], reference_neural=plan['reference_neural'],
        policy='Run all79 first and verify retained-reference parity before interpreting other arms. No automatic training or promotion.')
    (output/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    return execution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    result = prepare(a.plan, a.output)
    print(json.dumps(dict(state=result['state'], arms=len(result['arms']),
                         inventory=result['standalone_inventory'])), flush=True)
