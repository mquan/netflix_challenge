#!/usr/bin/env python3
"""Read-only, label-free context coverage audit of the archived N1 encoder.

Compare choice_tokens 0/1 in the same archived input builder with the same query
batch. No model construction, prediction, training, or validation-label access.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np

from gbdt_blend import load
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]


def archived_module(audit):
    archive = Path(audit['candidate_source_archive'])
    if digest(archive) != audit['candidate_source_archive_sha256']:
        raise ValueError('archived N1 source changed')
    with zipfile.ZipFile(archive) as z:
        source = z.read('scripts/choice_sequence.py')
        if hashlib.sha256(source).hexdigest() != audit['candidate_source_sha256']:
            raise ValueError('N1 source does not match audit')
        for name in ('masked_history.py', 'gbdt_blend.py'):
            if z.read('scripts/'+name) != (ROOT/'scripts'/name).read_bytes():
                raise ValueError('local dependency differs from N1 archive: '+name)
    module = types.ModuleType('archived_choice_context')
    module.__file__ = str(archive)+'!/scripts/choice_sequence.py'
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def compact(d, users, gather):
    """Copy only selected histories, preserving original user IDs for hash keys."""
    result = {k: d[k] for k in ('users', 'items', 'mean', 'min_day', 'max_day')}
    for prefix, fields in (('train', ('items', 'days', 'ratings')), ('implicit', ('items', 'days'))):
        _, index, _ = gather(d[prefix+'_offsets'], users)
        counts = np.zeros(d['users'], dtype=np.int64)
        counts[users] = np.diff(d[prefix+'_offsets'])[users]
        result[prefix+'_offsets'] = np.r_[0, counts.cumsum()]
        for field in fields:
            result[prefix+'_'+field] = np.asarray(d[prefix+'_'+field][index])
    return result


def summarize(rows):
    if not rows:
        return {'users': 0}
    total = lambda name: sum(r[name] for r in rows)
    before, after = total('rated_without_choices'), total('rated_with_choices')
    return dict(users=len(rows), query_rows=total('queries'),
        users_losing_rated_context=sum(r['rated_with_choices'] < r['rated_without_choices'] for r in rows),
        rated_tokens_without_choices=before, rated_tokens_with_choices=after,
        rated_token_loss=before-after, rated_token_loss_fraction=(before-after)/before if before else 0.,
        choice_tokens_retained=total('choices_retained'),
        choice_tokens_available=total('choices_available'),
        rated_id_removals=total('rated_ids_removed'), rated_id_additions=total('rated_ids_added'),
        max_rated_token_loss=max(r['rated_without_choices']-r['rated_with_choices'] for r in rows))


def run(a):
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(Path(a.audit).read_text())
    module = archived_module(audit)
    module.torch.set_num_threads(1)
    data = Path(audit['selection_dataset'])
    if digest(data) != audit['selection_dataset_sha256']:
        raise ValueError('P2 data differs from N1')
    # Use the fixed launch report, not a live checkpoint or progress file.
    report_path = Path(audit['candidate_record'])
    raw_report = report_path.read_bytes()
    if hashlib.sha256(raw_report).hexdigest() != audit['candidate_record_sha256']:
        raise ValueError('N1 launch report changed; inspect completed state first')
    report = json.loads(raw_report)
    cfg = {f.name: report['arguments'][f.name] for f in dataclasses.fields(module.Config)}
    cfg.update(device='cpu')
    c = module.Config(**cfg)
    d = load(data)
    probe = d['probe']
    if np.any(probe['user'][1:] < probe['user'][:-1]):
        raise ValueError('expected user-sorted P2 queries')
    query_counts = np.bincount(probe['user'], minlength=d['users'])
    query_offsets = np.r_[0, query_counts.cumsum()]
    population = np.flatnonzero(query_counts)
    if not 0 < a.users <= len(population):
        raise ValueError('invalid sample size')
    users = np.sort(np.random.default_rng(a.seed).choice(population, a.users, replace=False))
    small = compact(d, users, module.gather)
    offsets, _, _ = module.choices(small)
    records = []
    for start in range(0, len(users), 32):
        group = users[start:start+32]
        rows, index, _ = module.gather(small['train_offsets'], group)
        t_rows, qindex, _ = module.gather(query_offsets, group)
        t_items, t_days = probe['item'][qindex].astype(np.int64)-1, probe['day'][qindex].astype(np.int64)
        results = []
        for enabled in (0, 1):
            tokens, width = module.sequence_inputs(small, group, rows, index,
                np.ones(len(index), dtype=bool), t_rows, t_items, t_days,
                dataclasses.replace(c, choice_tokens=enabled), predicting=True)
            owner = tokens['seq_flat'].numpy()//width
            stars, items = tokens['seq_stars'].numpy(), tokens['seq_items'].numpy()
            rated = [set(items[(owner == r) & (stars < module.MASK)].tolist()) for r in range(len(group))]
            choice = [set(items[(owner == r) & (stars == module.CHOICE)].tolist()) for r in range(len(group))]
            for r in range(len(group)):
                if choice[r] & set(t_items[t_rows == r].tolist()):
                    raise ValueError('query pair leaked into choice context')
            results.append((rated, choice))
        for r, user in enumerate(group):
            before, after = results[0][0][r], results[1][0][r]
            records.append(dict(user=int(user), training_ratings=int(np.diff(small['train_offsets'])[user]),
                queries=int(query_counts[user]), choices_available=int(offsets[user+1]-offsets[user]),
                rated_without_choices=len(before), rated_with_choices=len(after),
                choices_retained=len(results[1][1][r]), rated_ids_removed=len(before-after), rated_ids_added=len(after-before)))
    if digest(data) != audit['selection_dataset_sha256'] or report_path.read_bytes() != raw_report:
        raise ValueError('audit inputs changed during execution')
    result = dict(audit_sha256=digest(a.audit), source_sha256=digest(__file__),
        source_archive_sha256=audit['candidate_source_archive_sha256'],
        data_sha256=audit['selection_dataset_sha256'], settings=vars(a),
        population_users=len(population), config=dataclasses.asdict(c),
        summary=summarize(records),
        by_training_history={name: summarize([r for r in records if lo <= r['training_ratings'] < hi])
            for name, lo, hi in [('under_384', 0, 384), ('384_to_1535', 384, 1536), ('1536_or_more', 1536, float('inf'))]},
        records=records,
        scope='Uniform sample of P2 query users; deterministic prediction input construction only. Same archived N1 builder with choice_tokens 0/1. This is not historical A2h model inference, training-mask coverage, score attribution, or a promotion gate. No P2 ratings were used; no model was loaded or updated.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('summary', 'by_training_history', 'scope')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='experiments/N1-input-comparison-audit.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--users', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20261019)
    run(parser.parse_args())
