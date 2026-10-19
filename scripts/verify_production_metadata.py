#!/usr/bin/env python3
"""Independent arithmetic checks for PF2, without importing its feature functions."""
import argparse
import ast
import collections
import csv
import json
from pathlib import Path

import numpy as np

from gbdt_blend import load
from round9_latent import digest


def original_tag_table(source, tags):
    # Execute only the historical pure table-construction statements. Exclude
    # its hard-coded dataset load and output writes; the original bank is immutable.
    nodes = ast.parse(Path(source).read_text()).body
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, n in enumerate(nodes) if assigns(n, 'kinds'))
    stop = next(i for i, n in enumerate(nodes) if assigns(n, 'd'))
    with Path(tags).open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    scope = dict(tags=rows, collections=collections)
    exec(compile(ast.Module(body=nodes[start:stop], type_ignores=[]), str(source), 'exec'), scope)
    return {k: np.asarray(v, dtype='<f4') for k, v in scope['table'].items()}


def verify(manifest, data, production_data, tags, original_generator, output):
    manifest, output = Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(manifest.read_text())
    assert report['state'] == 'complete'
    for path, expected in {**report['inputs'], **report['sources']}.items():
        assert digest(path) == expected, path
    for path in (data, production_data, tags):
        assert str(Path(path)) in report['inputs'], path
    verified = {}
    for stage, records in report['stages'].items():
        for name, rec in records.items():
            assert digest(rec['path']) == rec['sha256'], name
            values = (np.load(rec['path'], allow_pickle=False) if rec['path'].endswith('.npy')
                      else np.fromfile(rec['path'], dtype='<f4'))
            assert list(values.shape) == rec['shape'] and np.isfinite(values).all(), name
        verified[stage] = dict(files=len(records), all_hashes_shapes_and_values_finite=True)
    d, q = load(data), load(production_data)
    samples = np.linspace(0, len(q['probe'])-1, min(2048, len(q['probe'])), dtype=np.int64)
    records = report['stages']['qualifying']
    actual = {name: np.load(records[name]['path'], allow_pickle=False)[samples]
              for name in ('meta', 'context', 'extra_meta')}
    item_n = np.bincount(q['train_items'], minlength=q['items']+1)
    ratings = q['train_ratings'].astype(np.float64)
    item_sum = np.bincount(q['train_items'], weights=ratings, minlength=q['items']+1)
    item_sq = np.bincount(q['train_items'], weights=ratings*ratings, minlength=q['items']+1)
    prior1, prior5 = float(np.mean(ratings == 1)), float(np.mean(ratings == 5))
    del ratings
    first_item = np.full(q['items']+1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_item, q['implicit_items'], q['implicit_days'])
    expected = {name: [] for name in actual}
    for row in samples:
        query = q['probe'][row]
        u, i, day = int(query['user']), int(query['item']), int(query['day'])
        a, b = map(int, q['train_offsets'][u:u+2])
        r = q['train_ratings'][a:b].astype(np.float64)
        a, b = map(int, q['implicit_offsets'][u:u+2])
        days = q['implicit_days'][a:b].astype(np.float64)
        n, implicit = len(r), len(days)
        same, mean_day, distinct = int(np.count_nonzero(days == day)), float(days.mean()), len(np.unique(days))
        user_std = np.sqrt(max(float(r@r)/max(n, 1)-(float(r.sum())/max(n, 1))**2, 0))
        item_std = np.sqrt(max(item_sq[i]/max(item_n[i], 1)-(item_sum[i]/max(item_n[i], 1))**2, 0))
        expected['meta'].append([np.log1p(n), np.log1p(item_n[i]), np.log1p(implicit),
            (r.sum()+18)/(n+5), user_std, (item_sum[i]+90)/(item_n[i]+25), day,
            day-mean_day, day-days.min(), days.max()-days.min(), np.log1p(same)])
        expected['context'].append([1, np.log1p(n)/6-1, np.log1p(item_n[i])/10-1,
            (np.float32(day)-np.float32(mean_day))/np.float32(1000), np.log1p(same)/4-.5])
        expected['extra_meta'].append([day-first_item[i], (np.count_nonzero(r == 1)+5*prior1)/(n+5),
            (np.count_nonzero(r == 5)+5*prior5)/(n+5), np.log1p(implicit-n), np.log1p(distinct),
            same/(implicit/max(distinct, 1)), item_std])
    checks = {}
    for name in actual:
        values = np.asarray(expected[name], dtype='<f4')
        assert actual[name].tobytes() == values.tobytes(), name
        checks[name] = dict(columns=values.shape[1], sampled_rows=len(samples), byte_identical=True)
    original_sha = digest(original_generator)
    table = original_tag_table(original_generator, tags)
    for name, values in table.items():
        values = values[q['probe']['item']-1]
        emitted = np.fromfile(records['movie_'+name]['path'], dtype='<f4')
        assert emitted.tobytes() == values.tobytes(), name
    assert digest(original_generator) == original_sha
    unseen = np.setdiff1d(np.unique(q['probe']['item']), np.unique(d['probe']['item']))
    result = dict(passed=True, export_manifest=str(manifest), export_manifest_sha256=digest(manifest),
        stages=verified, numeric_sample=checks,
        sampling='np.linspace(0, qualifying_count-1, min(2048, qualifying_count), dtype=int64)',
        numeric_method='independent direct per-user rating and implicit-history reductions; global integer item moments and priors; same specified float32 output rounding',
        movie_checks=dict(columns=len(table), rows_per_column=len(q['probe']), all_bytes_exact=True,
            qualifying_movies_absent_from_development_probe=len(unseen),
            original_generator=str(original_generator), original_generator_sha256=original_sha,
            method='historical generator pure table statements, then full qualifying item lookup'),
        verifier_source_sha256=digest(__file__),
        scope='All output hashes/shapes/finiteness and all movie values verified. Rating-derived numeric values independently checked on at most 2048 qualifying rows; no score claim.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data', 'production-data', 'tags', 'original-generator', 'output'):
        parser.add_argument('--'+name, required=True)
    result = verify(**vars(parser.parse_args()))
    print(json.dumps(dict(passed=result['passed'], numeric_sample=result['numeric_sample'], movie_checks=result['movie_checks'])), flush=True)
