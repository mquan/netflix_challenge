"""Resolve all completed item-kNN outputs into the original retained alias order."""
import argparse
import json
from pathlib import Path

from production_rbm_audit import digest, verify, write_new


def resolve(plan_path):
    plan_path = Path(plan_path)
    plan_sha = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    destination = Path(plan['resolved_manifest'])
    if destination.exists():
        raise FileExistsError(destination)
    verify(plan['frozen'])
    if len(plan['stages']) != 28 or len(plan['column_occurrences']) != 29 or len(plan['support_aliases']) != 6:
        raise ValueError('Complete retained kNN coverage required')
    receipts, outputs = {}, {}
    for index, stage in enumerate(plan['stages']):
        root = Path(stage['output_dir'])
        path = root / 'audit.json'
        receipt = json.loads(path.read_text())
        if (receipt['state'] != 'verified_complete' or receipt['stage'] != index or
                receipt['plan_sha256'] != plan_sha or receipt['metric'] is not None or
                receipt['qualifying_count'] != plan['dataset']['query_count'] or
                receipt['retained_columns'] != stage['retained_columns']):
            raise ValueError('Incomplete or different kNN stage audit')
        for target in (root / 'model.f32', root / 'support.f32'):
            if str(target) not in receipt['outputs']:
                raise ValueError('Unbound qualifying prediction/support output')
        verify(receipt['outputs'])
        outputs[str(path)] = digest(path)
        outputs.update(receipt['outputs'])
        receipts[index] = receipt
    columns = []
    for occurrence in plan['column_occurrences']:
        index = occurrence['producer_stage']
        stage = plan['stages'][index]
        if occurrence['retained'] not in stage['retained_columns']:
            raise ValueError('Retained column points to a different producer')
        target = str(Path(stage['output_dir']) / 'model.f32')
        columns.append(dict(retained=occurrence['retained'], development_sha256=occurrence['development_sha256'],
                            qualifying_path=target, qualifying_sha256=receipts[index]['outputs'][target], producer_stage=index))
    supports, groups = [], {}
    for alias in plan['support_aliases']:
        index = alias['producer_stage']
        target = str(Path(plan['stages'][index]['output_dir']) / 'support.f32')
        if alias['output'] != target:
            raise ValueError('Support alias points to a different output')
        actual = receipts[index]['outputs'][target]
        original = alias['original_sha256']
        if original in groups and groups[original] != actual:
            raise ValueError('Equivalent support aliases have different production outputs')
        groups[original] = actual
        supports.append(dict(retained_input=alias['retained_input'], development_sha256=original,
                             qualifying_path=target, qualifying_sha256=actual, producer_stage=index))
    if len(groups) != 3:
        raise ValueError('Retained support-rule coverage differs')
    verify(plan['frozen'])
    verify(outputs)
    if digest(plan_path) != plan_sha:
        raise ValueError('kNN phase changed')
    result = dict(state='complete_knn_columns_and_support_aliases_resolved', plan_sha256=plan_sha,
                  master_sha256=plan['master_sha256'], qualifying_count=plan['dataset']['query_count'],
                  columns=columns, support_aliases=supports, audit_bindings=outputs, metric=None,
                  scope='Ordered kNN subset only. Other retained families, final blender and final qualifying export remain required.')
    write_new(destination, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    print(resolve(args.plan)['state'])
