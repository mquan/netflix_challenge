#!/usr/bin/env python3
"""Bind retained blend order, fixed full-probe budgets, and available qualifying features.

Untrained production prediction/support columns remain explicit pending entries;
development predictions are never substituted for qualifying predictions.
"""
import argparse
import json
from pathlib import Path

from gbdt_blend import load
from production_blend import budgets
from production_latent import verify
from production_metadata import pinned
from round9_latent import digest


def prepare(reference_root, reference_plan, latent_recipe, metadata_recipe, production_manifest, output):
    root, plan_path, latent_path, metadata_path, prod_path, output = map(
        Path, (reference_root, reference_plan, latent_recipe, metadata_recipe, production_manifest, output))
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    tree_path, neural_path = Path(plan['base_report']), Path(plan['control_report'])
    tree, neural = (json.loads(p.read_text()) for p in (tree_path, neural_path))
    latent, metadata = (json.loads(p.read_text()) for p in (latent_path, metadata_path))
    prod = json.loads(prod_path.read_text())
    sources = {str(p.resolve()): digest(p) for p in (plan_path, latent_path, metadata_path, prod_path)}
    for p in (tree_path, neural_path):
        sources[str(p)] = verify(p, plan['frozen'][str(p)])
    if Path(neural['config']['like']).name != tree_path.name:
        raise ValueError('neural report belongs to a different tree blend')
    for name in ('round4_boost.py', 'round8_nn_blend.py', 'moonshot_nn_blend.py', 'gbdt_blend.py'):
        p = Path(__file__).with_name(name)
        sources[str(p.resolve())] = verify(p, pinned(plan, 'scripts/'+name))
    for name in ('production_blend.py', 'production_blend_plan.py', 'production_blend_tree.py'):
        p = Path(__file__).with_name(name)
        sources[str(p.resolve())] = digest(p)
    cfg = tree['config']
    data = Path(plan['data'])
    original_sha = verify(data, pinned(plan, 'data/processed/netflix.bin'))
    prod_data = prod_path.parent / 'train-probe-qualifying.bin'
    if prod['state'] != 'complete' or prod['source_sha256'] != original_sha:
        raise ValueError('production dataset does not match development source')
    prod_sha = verify(prod_data, prod['outputs'][prod_data.name])
    for recipe in (latent, metadata):
        if recipe['state'] != 'complete':
            raise ValueError('incomplete production feature recipe')
        # The feature recipes bind both datasets, regardless of relative paths.
        bound = {str(Path(p).resolve()): h for p, h in recipe['inputs'].items()}
        if bound.get(str(data.resolve())) != original_sha or bound.get(str(prod_data.resolve())) != prod_sha:
            raise ValueError('feature recipe dataset differs')
    dev = dict(dataset=dict(path=str(data.resolve()), sha256=original_sha), rows=len(load(data)['probe']),
               predictions=[], auxiliary=[], context=[])
    query = dict(dataset=dict(path=str(prod_data.resolve()), sha256=prod_sha), rows=prod['qualifying_count'],
                 predictions=[], auxiliary=[], context=[])
    bank_root = root / cfg['bank_root']
    union_path = bank_root / 'experiments/UNION-round-manifest.json'
    support_path = bank_root / 'experiments/BOOST-support-inputs.json'
    union = {} if cfg['no_union'] else json.loads(union_path.read_text())['reports']['union']['inputs']
    support = {} if cfg['no_union'] else json.loads(support_path.read_text())
    for p in (union_path, support_path):
        sources[str(p)] = verify(p, plan['frozen'][str(p)])

    def entry(name, path, expected, column=None):
        record = dict(name=name, path=str(Path(path).resolve()), sha256=verify(path, expected))
        if column is not None:
            record['column'] = column
        return record

    def pending(name, sha):
        return dict(name=name, pending=True, development_sha256=sha,
                    requirement='Fixed-recipe train+probe regeneration with qualifying row order and provenance')

    for name, path in [(n, bank_root/n) for n in union] + [(n, root/n) for n in cfg['extra']]:
        expected = tree['inputs'][name]
        dev['predictions'].append(entry(name, path, expected))
        query['predictions'].append(pending(name, expected))
    for name, path in [(n, bank_root/n) for n in support] + [(n, root/n) for n in cfg['extra_support']]:
        expected = tree['inputs'][name]
        dev['auxiliary'].append(entry(name, path, expected))
        filename = Path(name).name
        if name.startswith('artifacts/r9/latent/'):
            source = latent['stages']['qualifying'][Path(name).stem]
            query['auxiliary'].append(entry(name, source['path'], source['sha256']))
        elif filename.startswith('meta-'):
            source = metadata['stages']['qualifying']['movie_'+Path(name).stem[len('meta-'):]]
            query['auxiliary'].append(entry(name, source['path'], source['sha256']))
        else:
            query['auxiliary'].append(pending(name, expected))
    for stage, target in (('development_replay', dev), ('qualifying', query)):
        for group in ('meta', 'extra_meta', 'context'):
            if group == 'extra_meta' and not cfg['extra_meta']:
                continue
            source = metadata['stages'][stage][group]
            for i, name in enumerate(source['columns']):
                if group == 'meta':
                    name = 'meta'+str(i)
                target['context' if group == 'context' else 'auxiliary'].append(
                    entry(name, source['path'], source['sha256'], i))
    schema = {key: [c['name'] for c in dev[key]] for key in ('predictions', 'auxiliary', 'context')}
    if schema != {key: [c['name'] for c in query[key]] for key in schema}:
        raise ValueError('development/qualifying feature order mismatch')
    expected_inputs = {v['name']: v['sha256'] for group in ('predictions', 'auxiliary') for v in dev[group] if 'column' not in v}
    if expected_inputs != tree['inputs']:
        raise ValueError('bank entries differ from retained report')
    result = dict(state='development_ready_qualifying_pending', development=dev, qualifying=query, schema=schema,
        settings=dict(tree_config={k: cfg[k] for k in ('learning_rate', 'leaves', 'min_leaf')},
            neural_config={k: v for k, v in neural['config'].items() if k not in ('data', 'like', 'output')},
            budgets=budgets(tree, neural), budget_policy='per-seed median selected budget across the five completed development folds; no qualifying selection',
            lightgbm_version=tree['lightgbm_version'], torch_version=neural['torch_version']),
        sources=sources,
        reference_score=neural['avg_rmse'],
        limitation='Preparation only. Do not start final fitting/retraining until the development bank is frozen after its promotion gate. Pending qualifying entries are not executable inputs.')
    for path, sha in sources.items():
        verify(path, sha)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root', 'reference-plan', 'latent-recipe', 'metadata-recipe', 'production-manifest', 'output'):
        parser.add_argument('--'+name, required=True)
    result = prepare(**vars(parser.parse_args()))
    print(json.dumps(dict(state=result['state'], widths={k: len(v) for k, v in result['schema'].items()},
        budgets=result['settings']['budgets'], pending_qualifying=sum(bool(v.get('pending'))
            for k in result['schema'] for v in result['qualifying'][k]))), flush=True)
