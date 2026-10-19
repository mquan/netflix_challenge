#!/usr/bin/env python3
"""Bind every retained prediction column to recovered recipe evidence, by digest."""
from collections import Counter
import json
from pathlib import Path

from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / 'experiments/production-recipe-coverage.json'
    if destination.exists():
        raise FileExistsError(destination)
    inventory_path = ROOT / 'experiments/production-bank-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    sources, matches = {}, {}

    def read(name):
        path = ROOT / 'experiments' / name
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())

    def add(sha, family, source, index, validation):
        entry = dict(family=family, recipe='experiments/' + source, entry=index, validation=validation)
        if sha in matches and matches[sha]['family'] != family:
            raise ValueError('ambiguous recipe family for ' + sha)
        matches[sha] = entry

    for source, family, validation in [
        ('production-neural-recipes.json', 'neural', 'Archived source and fixed epochs recovered; fourteen prepare-only checks. Full production training is outstanding.'),
        ('production-autorec-recipes.json', 'autorec', 'Epoch budgets recovered; fixed-budget fixture compatibility checked. Historical producer digests are incomplete; full retraining is outstanding.'),
        ('production-bayes-recipes.json', 'bayes', 'Stored posterior means match retained columns. Fixed-sweep production interface checked on fixtures; multithreaded retraining is nondeterministic and outstanding.'),
    ]:
        for index, recipe in enumerate(read(source)['recipes']):
            add(recipe['column_sha256'], family, source, index, validation)
    source = 'production-joint-recipes.json'
    for index, stage in enumerate(read(source)['stages']):
        add(stage['prediction_sha256'], 'joint_integrated', source, index,
            'All retained checkpoint inference replays match exactly. J1-J4 full original-data retraining is exact (PR5/PR6); production retraining is outstanding.')
    source = 'production-rbm-recipes-v2.json'
    for index, column in enumerate(read(source)['retained_columns']):
        add(column['sha256'], 'rbm', source, index,
            'Nine retained inference columns match across PR3/PR4 recovery evidence. Fixed training/resumption budgets recovered; full production retraining is outstanding.')
    source = 'production-knn-recipes.json'
    for index, column in enumerate(read(source)['columns']):
        add(column['prediction_sha256'], 'item_knn', source, index,
            'Every retained item-kNN inference column matches exactly. Production base models and graphs must be regenerated.')
    source = 'production-neighborhood-recipe.json'
    recipe = read(source)
    evidence_path = Path(recipe['retraining_evidence'])
    if digest(evidence_path) != recipe['retraining_evidence_sha256']:
        raise ValueError('neighborhood evidence changed')
    evidence = json.loads(evidence_path.read_text())
    add(evidence['predictions']['reference_sha256'], 'learned_neighborhood', source, None,
        'Rebuilt baseline/graphs reproduce retained-checkpoint inference exactly. Fixed-budget retraining has sub-1e-6 prediction drift; production retraining is outstanding.')
    source = 'production-user-knn-recipe.json'
    recipe = read(source)
    evidence_path = Path(recipe['inference_evidence'])
    if digest(evidence_path) != recipe['inference_evidence_sha256']:
        raise ValueError('user-kNN evidence changed')
    evidence = json.loads(evidence_path.read_text())
    add(evidence['comparisons']['full_replay']['.f32']['reference_sha256'], 'user_knn', source, None,
        'Full retained inference is exact; rebuilt source has bounded compatibility coverage. Production K1 training and inference are outstanding.')
    source = 'production-kernel-recipes.json'
    for index, recipe in enumerate(read(source)['recipes']):
        for column in recipe['columns']:
            add(column['sha256'], 'gp' if recipe['family'] == 'gp' else 'local_residual', source, index,
                'Full original-data GP inference or six-epoch local retraining matches retained columns exactly. Production-data runs are outstanding.')
    covered, missing = [], []
    for column in inventory['columns']:
        if digest(column['path']) != column['sha256']:
            raise ValueError('retained bank changed: ' + column['path'])
        if column['sha256'] not in matches:
            missing.append(column['path'])
        else:
            covered.append(dict(column=column['path'], sha256=column['sha256'], **matches[column['sha256']]))
    if missing:
        raise ValueError('missing recipes: ' + ', '.join(missing))
    if len(covered) != 79 or len({c['sha256'] for c in covered}) != 77:
        raise ValueError('unexpected retained bank size')
    result = dict(inventory=str(inventory_path), inventory_sha256=digest(inventory_path), sources=sources,
                  columns=covered, family_counts=dict(Counter(c['family'] for c in covered)),
                  retained_columns=len(covered), unique_columns=len({c['sha256'] for c in covered}),
                  scope='Recipe coverage, not completed production training or a score gain. Validation strength varies and is stated per family. Production latent/context features, blender fitting, qualifying export and the target score remain outstanding.')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('retained_columns', 'unique_columns', 'family_counts')}))


if __name__ == '__main__':
    main()
