#!/usr/bin/env python3
"""Replay development metadata and export the same formulas on train+probe queries.

Keep all category mappings and fixed context scales unchanged. Statistics use
the dataset's training labels only; query labels never enter feature formulas.
Learned blender normalization must subsequently be fitted on development rows
and reused for qualifying, not fitted here on the qualifying distribution.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from gbdt_blend import load, meta_features
from moonshot_nn_blend import contexts
from production_latent import same_context, verify
from round4_boost import extra_meta
from round9_latent import digest

CONTEXT_NAMES = ['intercept', 'log_training_user_count_scaled', 'log_training_item_count_scaled',
                 'day_minus_implicit_user_mean_scaled', 'log_implicit_session_count_scaled']


def movie_table(path, items):
    """The retained meta_columns.py categorical encoding, with explicit ID checks."""
    with Path(path).open(newline='') as f:
        tags = list(csv.DictReader(f, delimiter='\t'))
    if len(tags) != items or any(int(t['movie_id']) != i+1 for i, t in enumerate(tags)):
        raise ValueError('movie tags must cover ascending dense movie IDs exactly')
    kinds = sorted({t['kind'] for t in tags})
    audiences = ['kids', 'family', 'teen', 'adult', 'mature']
    tones = {'light': 0, 'neutral': 1, 'dark': 2}
    confidence = {'low': 0, 'medium': 1, 'high': 2}
    franchise = collections.Counter(t['franchise'] for t in tags if t['franchise'])
    table = dict(kind=[kinds.index(t['kind']) for t in tags],
                 audience=[audiences.index(t['audience']) for t in tags],
                 tone=[tones[t['tone']] for t in tags],
                 popularity=[int(t['era_popularity']) for t in tags],
                 confidence=[confidence[t['confidence']] for t in tags],
                 franchise_size=[franchise.get(t['franchise'], 0) for t in tags])
    for genre in ('action', 'comedy', 'drama', 'horror', 'romance', 'scifi',
                  'family', 'thriller', 'documentary', 'foreign'):
        table['g_'+genre] = [float(genre in t['genres'].split('|')) for t in tags]
    return {name: np.asarray(values, dtype='<f4') for name, values in table.items()}


def blocks(d):
    """Yield blocks separately so the large historical temporary arrays are released."""
    meta = meta_features(d)
    yield 'meta', np.column_stack(list(meta.values())).astype('<f4'), list(meta)
    del meta
    yield 'context', contexts(d).astype('<f4'), CONTEXT_NAMES
    values, names = extra_meta(d)
    yield 'extra_meta', values.astype('<f4'), names


def pinned(plan, suffix):
    matches = [h for path, h in plan['frozen'].items() if Path(path).as_posix().endswith('/'+suffix)]
    if len(matches) != 1:
        raise ValueError(f'expected one frozen binding for {suffix}')
    return matches[0]


def export(data, reference_root, reference_plan, production_manifest, tags, output):
    data, root, plan_path, prod_path, tags, output = map(
        Path, (data, reference_root, reference_plan, production_manifest, tags, output))
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    prod = json.loads(prod_path.read_text())
    blender_path = Path(plan['base_report'])
    blender = json.loads(blender_path.read_text())
    cache = root / 'artifacts/r4/meta-context.npz'
    inputs = {str(plan_path): digest(plan_path), str(prod_path): digest(prod_path), str(tags): digest(tags)}
    inputs[str(blender_path)] = verify(blender_path, plan['frozen'][str(blender_path)])
    inputs[str(data)] = verify(data, pinned(plan, 'data/processed/netflix.bin'))
    inputs[str(cache)] = verify(cache, pinned(plan, 'artifacts/r4/meta-context.npz'))
    if prod['state'] != 'complete' or prod['source_sha256'] != inputs[str(data)]:
        raise ValueError('production dataset is not bound to original training data')
    prod_data = prod_path.parent / 'train-probe-qualifying.bin'
    inputs[str(prod_data)] = verify(prod_data, prod['outputs'][prod_data.name])
    sources = {}
    for name in ('gbdt_blend.py', 'moonshot_nn_blend.py', 'round4_boost.py'):
        path = Path(__file__).with_name(name)
        sources[str(path.resolve())] = verify(path, pinned(plan, 'scripts/'+name))
    for name in ('production_metadata.py', 'production_latent.py', 'round9_latent.py'):
        path = Path(__file__).with_name(name)
        sources[str(path.resolve())] = digest(path)
    d, q = load(data), load(prod_data)
    same_context(d, q)
    if len(q['probe']) != prod['qualifying_count'] or len(q['train_ratings']) != prod['train_count']:
        raise ValueError('production row counts disagree with manifest')
    table = movie_table(tags, d['items'])
    movie_paths = [p for p in blender['config']['extra_support'] if Path(p).name.startswith('meta-')]
    movie_names = [Path(p).stem[len('meta-'):] for p in movie_paths]
    if set(movie_names) != set(table) or len(movie_names) != len(table):
        raise ValueError('movie feature schema differs from retained bank')
    for p in movie_paths:
        inputs[str(root / p)] = verify(root / p, blender['inputs'][p])
    output.mkdir(parents=True)
    stages = {}
    for stage, dataset in (('development_replay', d), ('qualifying', q)):
        dest = output / stage
        dest.mkdir()
        records = {}
        for name, values, columns in blocks(dataset):
            print(json.dumps(dict(stage=stage, block=name, shape=values.shape)), flush=True)
            if values.shape != (len(dataset['probe']), len(columns)) or not np.isfinite(values).all():
                raise ValueError('invalid metadata block: '+name)
            if stage == 'development_replay' and name in ('meta', 'context'):
                with np.load(cache, allow_pickle=False) as saved:
                    retained = saved[name]
                if values.dtype != retained.dtype or values.shape != retained.shape or values.tobytes() != retained.tobytes():
                    raise ValueError('cached development metadata replay differs: '+name)
            path = dest / (name+'.npy')
            np.save(path, values, allow_pickle=False)
            records[name] = dict(path=str(path.resolve()), sha256=digest(path),
                                 shape=list(values.shape), columns=columns,
                                 validation='cached_array_byte_exact' if stage == 'development_replay' and name in ('meta', 'context') else 'unchanged_source_formula')
        indices = dataset['probe']['item'].astype(np.int64)-1
        for name, retained in zip(movie_names, movie_paths):
            values = table[name][indices]
            path = dest / ('meta-'+name+'.f32')
            values.tofile(path)
            sha = digest(path)
            if stage == 'development_replay' and sha != blender['inputs'][retained]:
                raise ValueError('movie feature replay differs: '+name)
            records['movie_'+name] = dict(path=str(path.resolve()), sha256=sha,
                shape=[len(values)], retained_input=retained,
                validation='retained_column_byte_exact' if stage == 'development_replay' else 'frozen_movie_table_lookup')
        stages[stage] = records
    for path, sha in {**inputs, **sources}.items():
        verify(path, sha)
    report = dict(state='complete', inputs=inputs, sources=sources, stages=stages,
        movie_feature_order=movie_names, statistics_policy='development uses original training; qualifying uses train+probe; all implicit interactions unchanged',
        context_policy='same five formulas and fixed scales; no fitted qualifying normalization',
        movie_policy='unchanged title tags and categorical mappings; qualifying item lookup',
        checks=['original meta and context cache arrays reproduced byte for byte',
                'all 16 retained movie feature columns reproduced byte for byte',
                'extra metadata uses hash-verified original source; no historical cached array exists',
                'all output rows finite and in input query order', 'all frozen inputs and sources rehashed'],
        limitation='Metadata export only. Retrained predictions, regenerated neighborhood supports, and final fitted blender remain required. No score improvement claimed.')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'reference-root', 'reference-plan', 'production-manifest', 'tags', 'output'):
        parser.add_argument('--'+name, required=True)
    report = export(**vars(parser.parse_args()))
    print(json.dumps(dict(state=report['state'], blocks=len(report['stages']['qualifying']))), flush=True)


if __name__ == '__main__':
    main()
