"""Resolve every retained item-kNN column and support alias to production inputs."""
import argparse
import json
from pathlib import Path

from production_rbm_audit import digest, verify

ROOT = Path(__file__).resolve().parents[1]


def argument(command, flag):
    return command[command.index(flag) + 1]


def signature(command):
    normalized = list(command)
    for flag in ('--predictions', '--support'):
        normalized[normalized.index(flag) + 1] = '{' + flag + '}'
    return tuple(normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    registry_path = ROOT / 'experiments/production-knn-recipes.json'
    joint_registry_path = ROOT / 'experiments/production-joint-recipes.json'
    joint_path = ROOT / 'experiments/P0-JOINT-20261031-plan.json'
    dep_path = ROOT / 'experiments/P0-DEPS-20261031-plan.json'
    master_path = ROOT / 'experiments/P0-production-20261031-plan.json'
    graph_proof_path = ROOT / 'experiments/PR7-dependency-verification.json'
    registry, joint_registry, joint, dep, master, graph_proof = [json.loads(p.read_text()) for p in
        (registry_path, joint_registry_path, joint_path, dep_path, master_path, graph_proof_path)]
    if joint['master_sha256'] != digest(master_path) or dep['master_sha256'] != digest(master_path):
        raise ValueError('Production master bindings differ')
    joint_by_name = {s['name']: (index, s) for index, s in enumerate(joint['stages'])}
    bases = {}
    for stage in joint_registry['stages']:
        index, production = joint_by_name[stage['name']]
        bases[str(Path(stage['model']).resolve())] = dict(
            phase='joint', stage=index, model=str(Path(production['output_dir']) / 'model.bin'),
            audit=str(Path(production['output_dir']) / 'audit.json'), original_sha256=stage['model_sha256'])
    original_base = str((ROOT / 'artifacts/time-baseline.bin').resolve())
    bases[original_base] = dict(phase='dependencies', stage=0,
        model=str(Path(dep['stages'][0]['output_dir']) / 'model.bin'),
        audit=str(Path(dep['stages'][0]['output_dir']) / 'audit.json'),
        original_sha256=graph_proof['comparisons']['time_baseline']['reference_sha256'])
    graphs = {
        'artifacts/neighbors64.bin': dict(stage=2, graph=str(Path(dep['stages'][2]['output_dir']) / 'derived/neighbors64.bin'),
            original_sha256=graph_proof['comparisons']['graph64']['reference_sha256']),
        'artifacts/r4/neighbors256.bin': dict(stage=1, graph=str(Path(dep['stages'][1]['output_dir']) / 'neighbors256.bin'),
            original_sha256=graph_proof['comparisons']['graph256']['reference_sha256']),
    }
    # Every recorded kNN replay used the same exact binary as retained J1-J4.
    binary = joint['stages'][0]['command'][0]
    binary_sha = digest(binary)
    frozen = {str(p): digest(p) for p in [registry_path, joint_registry_path, joint_path, dep_path,
                                        master_path, graph_proof_path, Path(__file__)]}
    frozen[binary] = binary_sha
    stages, by_prediction, by_signature = [], {}, {}
    root = Path(master['output_root']) / 'knn'
    for column in registry['columns']:
        if column['binary_sha256'] != binary_sha:
            raise ValueError('Different historical kNN binary requires separate recovery')
        command = list(column['command'])
        key = signature(command)
        if column['prediction_sha256'] in by_prediction:
            index = by_prediction[column['prediction_sha256']]
            if key != stages[index]['original_signature']:
                raise ValueError('Equal development predictions do not prove equal recipes')
            stages[index]['retained_columns'].append(column['retained'])
            continue
        index = len(stages)
        flag = '--joint-model' if '--joint-model' in command else '--time-model'
        base_path = argument(command, flag)
        base = bases[str((ROOT / base_path).resolve())]
        if column['inputs'][base_path]['sha256'] != base['original_sha256']:
            raise ValueError('kNN base model provenance differs')
        graph_path = argument(command, '--graph')
        g = graphs[graph_path]
        if column['inputs'][graph_path]['sha256'] != g['original_sha256']:
            raise ValueError('kNN graph provenance differs')
        directory = root / f'stage-{index:02d}'
        if directory.exists():
            raise FileExistsError(directory)
        command[0] = binary
        for option, value in [('--data', master['data']), ('--graph', g['graph']),
                              (flag, base['model']), ('--predictions', str(directory / 'model.f32')),
                              ('--support', str(directory / 'support.f32'))]:
            command[command.index(option) + 1] = value
        stage = dict(index=index, command=command, output_dir=str(directory), original_signature=key,
                     original_prediction_sha256=column['prediction_sha256'], retained_columns=[column['retained']],
                     base_dependency=base, graph_dependency=g, execution_record=column['execution_record'],
                     execution_record_sha256=column['execution_record_sha256'], threads=1)
        stages.append(stage)
        by_prediction[column['prediction_sha256']] = index
        by_signature[key] = index
        frozen[column['execution_record']] = column['execution_record_sha256']
    supports = []
    support_groups = {}
    for support in registry['support_columns']:
        key = signature(support['command'])
        if key not in by_signature:
            raise ValueError('Support alias has no matching complete prediction recipe')
        index = by_signature[key]
        command = support['command']
        support_signature = tuple(argument(command, flag) for flag in ('--graph', '--neighbors', '--power', '--time-decay'))
        expected = support['expected_sha256']
        if support['sha256'] != expected:
            raise ValueError('Historical support digest differs')
        if expected in support_groups and support_groups[expected] != support_signature:
            raise ValueError('Equal support bytes do not justify different production support rules')
        support_groups[expected] = support_signature
        supports.append(dict(retained_input=support['retained_input'], original_sha256=expected,
                             producer_stage=index, output=str(Path(stages[index]['output_dir']) / 'support.f32')))
    if (len(stages), sum(len(s['retained_columns']) for s in stages), len(supports), len(support_groups)) != (28, 29, 6, 3):
        raise ValueError('Retained kNN or support coverage differs')
    verify(frozen)
    result = dict(state='all_knn_columns_and_support_aliases_resolved_not_executable',
                  master_plan=str(master_path), master_plan_sha256=digest(master_path), frozen_bindings=frozen,
                  dependency_plan=str(dep_path), dependency_plan_sha256=digest(dep_path),
                  joint_plan=str(joint_path), joint_plan_sha256=digest(joint_path),
                  stages=stages, support_aliases=supports,
                  unique_prediction_recipes=28, retained_column_occurrences=29, retained_support_aliases=6,
                  unique_support_rules=3,
                  scope='Production input/path resolution only. Duplicate prediction occurrences require identical full recipes; support aliases require matching graph, neighbor count, power and time decay. Native output auditor, guarded execution plan and fixture checks remain to be prepared. No kNN production inference launched.')
    with destination.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(state=result['state'], stages=28, columns=29, support_aliases=6, unique_support_rules=3)))


if __name__ == '__main__':
    main()
