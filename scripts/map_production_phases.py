"""Map frozen production phases to the complete ordered retained blender bank."""
import argparse
import json
from pathlib import Path

from production_rbm_audit import digest, write_new

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('GPU', 'CPU-neural', 'RBM', 'DEPS', 'JOINT', 'KNN', 'AUTOREC', 'BAYES', 'KERNEL', 'USERKNN')


def mapping():
    master_path = ROOT / 'experiments/P0-production-20261031-plan.json'
    blender_path = ROOT / 'experiments/production-blender-plan.json'
    inventory_path = ROOT / 'experiments/production-bank-inventory.json'
    master, blender, inventory = [json.loads(p.read_text()) for p in (master_path, blender_path, inventory_path)]
    by_name = {Path(c['path']).name: c for c in inventory['columns']}
    if len(by_name) != 79:
        raise ValueError('Retained names are not unique or bank has wrong size')
    bindings = {str(p): digest(p) for p in (master_path, blender_path, inventory_path, Path(__file__))}
    predictions, supports, phases = {}, {}, {}

    def prediction(phase, plan_path, stage, retained, sha, output):
        name = Path(retained).name
        original = by_name[name]
        if original['sha256'] != sha or name in predictions:
            raise ValueError('Ambiguous or wrong retained prediction producer: ' + name)
        predictions[name] = dict(phase=phase, plan=str(plan_path), plan_sha256=digest(plan_path),
                                 stage=stage, retained=original['path'], development_sha256=sha,
                                 planned_output=str(output))

    for phase in PHASES:
        path = ROOT / f'experiments/P0-{phase}-20261031-plan.json'
        p = json.loads(path.read_text())
        if p['master_sha256'] != digest(master_path) or p.get('data', master['data']) != master['data']:
            raise ValueError('Production phase belongs to different master/data')
        bindings[str(path)] = digest(path)
        phases[phase] = dict(plan=str(path), plan_sha256=digest(path), stages=len(p['stages']))
        for index, stage in enumerate(p['stages']):
            if phase in ('GPU', 'CPU-neural'):
                e = stage['expected']
                prediction(phase, path, index, e['column'], e['column_sha256'], Path(stage['output_dir']) / 'model.f32')
            elif phase == 'RBM':
                for column in stage['retained_columns']:
                    prediction(phase, path, index, column['column'], column['sha256'], Path(stage['output_dir']) / 'model.f32')
            elif phase == 'JOINT':
                for column in stage['retained_columns']:
                    prediction(phase, path, index, column, by_name[Path(column).name]['sha256'],
                               Path(stage['output_dir']) / 'model.f32')
            elif phase == 'DEPS' and stage['kind'] == 'neighborhood':
                column = by_name['neighborhood64.f32']
                prediction(phase, path, index, column['path'], column['sha256'], Path(stage['output_dir']) / 'model.f32')
        for column in p.get('column_occurrences', []):
            index = column['producer_stage']
            stage = p['stages'][index]
            retained = column['column'] if phase == 'AUTOREC' else column['retained']
            output = column.get('path', str(Path(stage['output_dir']) / column.get('prediction', 'model.f32')))
            prediction(phase, path, index, retained, column['development_sha256'], output)
        for alias in p.get('support_aliases', []):
            retained = alias.get('retained', alias.get('retained_input'))
            name = Path(retained).name
            if name in supports:
                raise ValueError('Duplicate support alias')
            supports[name] = dict(phase=phase, plan=str(path), plan_sha256=digest(path),
                                  stage=alias['producer_stage'],
                                  development_sha256=alias.get('development_sha256', alias.get('original_sha256')),
                                  planned_output=alias.get('path', alias.get('output')))
    ordered = []
    for column in blender['development']['predictions']:
        entry = predictions.pop(Path(column['name']).name)
        if entry['development_sha256'] != column['sha256']:
            raise ValueError('Blender prediction hash differs')
        ordered.append(dict(name=column['name'], **entry))
    if predictions or len(ordered) != 79:
        raise ValueError('Incomplete or excess retained prediction coverage')
    ordered_supports = []
    for column in blender['qualifying']['auxiliary']:
        if not column.get('pending'):
            continue
        entry = supports.pop(Path(column['name']).name)
        if entry['development_sha256'] != column['development_sha256']:
            raise ValueError('Blender support alias hash differs')
        ordered_supports.append(dict(name=column['name'], **entry))
    if supports or len(ordered_supports) != 8:
        raise ValueError('Incomplete or excess retained support coverage')
    for entry in ordered + ordered_supports:
        p = json.loads(Path(entry['plan']).read_text())
        entry['required_audit'] = str(Path(p['stages'][entry['stage']]['output_dir']) / 'audit.json')
    # A digest match proves a frozen mapping, never that an output already exists.
    completed = sum(Path(entry['required_audit']).is_file() for entry in ordered)
    return dict(state='complete_planned_production_bank_coverage', frozen=bindings, phases=phases,
                predictions=ordered, support_aliases=ordered_supports, prediction_occurrences=79,
                unique_planned_prediction_files=len({entry['planned_output'] for entry in ordered}),
                support_alias_occurrences=8, required_stage_audits=sum(p['stages'] for p in phases.values()),
                prediction_occurrences_with_audit_file_present=completed,
                scope='Planning coverage only: all retained prediction/support occurrences map to frozen production '
                'phase outputs in original blender order. Audit-file presence is not audit verification. '
                'Production completion, output hashes, final blender fitting and ordered qualifying export remain required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = mapping()
    write_new(args.report, report)
    print(json.dumps({key: report[key] for key in ('state', 'prediction_occurrences', 'support_alias_occurrences',
                                                  'required_stage_audits', 'prediction_occurrences_with_audit_file_present')}))
