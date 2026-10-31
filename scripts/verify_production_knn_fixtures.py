"""Run every frozen item-kNN recipe on two audited, label-isolated native fixtures."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from prepare_production_dependency_phase import build_stages
from production_dependency_audit import audit as audit_dependency, prepare as prepare_dependency
from production_joint_audit import audit as audit_joint, prepare as prepare_joint
from production_knn_audit import audit as audit_knn, prepare as prepare_knn
from production_rbm_audit import digest, verify, write_new

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    mapping_path, output, report_path = (p.resolve() for p in (args.mapping, args.output_dir, args.report))
    if output.exists() or report_path.exists():
        raise FileExistsError('Choose new fixture outputs')
    mapping = json.loads(mapping_path.read_text())
    verify(mapping['frozen_bindings'])
    binary = mapping['stages'][0]['command'][0]
    native = json.loads((ROOT / 'experiments/P0-dependency-source-recovery-20261031.json').read_text())['binary']
    helper_info = json.loads((ROOT / 'experiments/P0-rbm-dataset-metadata-20261031.json').read_text())
    helper = helper_info['binary']
    if digest(helper) != helper_info['binary_sha256']:
        raise ValueError('Native metadata helper changed')
    source_paths = [ROOT / 'scripts' / name for name in (
        'production_knn_audit.py', 'production_joint_audit.py', 'production_dependency_audit.py',
        'production_rbm_audit.py', 'production_graph.py', 'prepare_production_dependency_phase.py')]
    frozen = {str(p): digest(p) for p in [mapping_path, *source_paths]}
    frozen.update({binary: digest(binary), native: digest(native), helper: digest(helper)})
    output.mkdir(parents=True)
    receipts, phases, commands = [], [], []

    def execute(command, path):
        with Path(path).open('w') as log:
            subprocess.run(['nice', '-n', '10', *command], stdout=log, stderr=subprocess.STDOUT, check=True)
        commands.append(command)

    for label in (0, 5):
        folder = output / str(label)
        folder.mkdir()
        data = folder / 'data.bin'
        users, items = 200, 300
        training = [[1, 2, 3] if u % 2 == 0 else [1, 4, 5] for u in range(users)]
        ratings = [1 + u % 5 for u in range(users) for _ in range(3)]
        mean = sum(ratings) / len(ratings)
        with data.open('wb') as stream:
            stream.write(struct.pack('<8sIIQQQHH4xd', b'NFPDAT2\0', users, items,
                                     users * 3, users * 2, users * 5, 0, 4, mean))
            def put(dtype, values):
                stream.write(np.array(list(values), dtype=dtype).tobytes())
            put('<u8', (u * 3 for u in range(users + 1)))
            put('<u2', (i for row in training for i in row))
            put('<u2', (i - 1 for row in training for i in row))
            put('u1', ratings)
            for u in range(users):
                for item in ([4, 5] if u % 2 == 0 else [2, 3]):
                    stream.write(struct.pack('<IHHB3x', u, item, item - 1, label))
            put('<u8', (u * 5 for u in range(users + 1)))
            put('<u2', list(range(1, 6)) * users)
            put('<u2', list(range(5)) * users)
        metadata = json.loads(subprocess.check_output([helper, str(data)], text=True))
        metadata['mean'] = mean
        local_frozen = dict(frozen)
        local_frozen[str(data)] = digest(data)
        dep_stages = build_stages(folder / 'dependencies', str(data), native, sys.executable)
        dep_path = folder / 'dependency-plan.json'
        write_new(dep_path, dict(data=str(data), dataset=metadata, frozen=local_frozen, stages=dep_stages))
        for index, stage in enumerate(dep_stages):
            prepare_dependency(dep_path, index)
            for command, log in zip(stage['commands'], ('train.log', 'reload.log')):
                execute(command, Path(stage['output_dir']) / log)
            audit_dependency(dep_path, index)
        joint_dir = folder / 'joint'
        joint_command = [binary, 'joint', '--data', str(data), '--model', str(joint_dir / 'model.bin'),
                         '--predictions', str(joint_dir / 'model.f32'), '--factors', '4', '--iterations', '2',
                         '--learning-rate', '.012', '--regularization', '.015', '--decay', '.93',
                         '--patience', '0', '--fixed-epoch', '--threads', '1']
        joint_stage = dict(name='fixture-joint', family='joint', command=joint_command, output_dir=str(joint_dir),
                           rank=4, day_rank=0, model_format=2, seed=2026, start_epoch=1, fixed_final_epoch=2, dependency_index=None)
        joint_path = folder / 'joint-plan.json'
        write_new(joint_path, dict(data=str(data), dataset=metadata, frozen=local_frozen, stages=[joint_stage]))
        prepare_joint(joint_path, 0)
        execute(joint_command, joint_dir / 'train.log')
        execute([binary, 'eval-joint', '--data', str(data), '--model', str(joint_dir / 'model.bin'),
                 '--predictions', str(joint_dir / 'reload.f32'), '--threads', '1'], joint_dir / 'reload.log')
        audit_joint(joint_path, 0)
        source_phases = dict(dependencies=dict(plan=str(dep_path), plan_sha256=digest(dep_path)),
                             joint=dict(plan=str(joint_path), plan_sha256=digest(joint_path)))
        stages = []
        for index, original in enumerate(mapping['stages']):
            root = folder / 'knn' / f'stage-{index:02d}'
            command = list(original['command'])
            kind = original['base_dependency']['phase']
            base = joint_dir / 'model.bin' if kind == 'joint' else Path(dep_stages[0]['output_dir']) / 'model.bin'
            graph_index = original['graph_dependency']['stage']
            graph_root = Path(dep_stages[graph_index]['output_dir'])
            graph = graph_root / ('derived/neighbors64.bin' if graph_index == 2 else 'neighbors256.bin')
            for flag, value in [('--data', str(data)), ('--graph', str(graph)),
                                ('--joint-model' if kind == 'joint' else '--time-model', str(base)),
                                ('--predictions', str(root / 'model.f32')), ('--support', str(root / 'support.f32'))]:
                command[command.index(flag) + 1] = value
            stages.append(dict(command=command, output_dir=str(root), retained_columns=original['retained_columns'],
                               base_dependency=dict(phase=kind, stage=0, model=str(base)),
                               graph_dependency=dict(stage=graph_index, graph=str(graph))))
        knn_path = folder / 'knn-plan.json'
        write_new(knn_path, dict(data=str(data), dataset=metadata, frozen=local_frozen,
                                stages=stages, source_phases=source_phases))
        phases.append(dict(label=label, plan=str(knn_path), sha256=digest(knn_path), source_phases=source_phases))
        for index, stage in enumerate(stages):
            prepare_knn(knn_path, index)
            root = Path(stage['output_dir'])
            execute(stage['command'], root / 'run.log')
            replay = list(stage['command'])
            replay[replay.index('--predictions') + 1] = str(root / 'replay.f32')
            replay[replay.index('--support') + 1] = str(root / 'replay-support.f32')
            execute(replay, root / 'replay.log')
            result = audit_knn(knn_path, index)
            if result['support_min'] <= 0:
                raise ValueError('Fixture failed to exercise nonzero support on every query')
            receipts.append(dict(label=label, stage=index, audit=str(root / 'audit.json'), sha256=digest(root / 'audit.json')))
    isolation = []
    for index in range(28):
        for name in ('model.f32', 'support.f32'):
            a = output / '0/knn' / f'stage-{index:02d}' / name
            b = output / '5/knn' / f'stage-{index:02d}' / name
            if a.read_bytes() != b.read_bytes():
                raise ValueError('Query labels changed kNN predictions/support')
        isolation.append(dict(stage=index, predictions_and_support_byte_identical=True))
    support_groups = {}
    for alias in mapping['support_aliases']:
        path = output / '0/knn' / f"stage-{alias['producer_stage']:02d}" / 'support.f32'
        actual = digest(path)
        if alias['original_sha256'] in support_groups and support_groups[alias['original_sha256']] != actual:
            raise ValueError('Support alias equivalence changed')
        support_groups[alias['original_sha256']] = actual
    if len(set(support_groups.values())) != 3:
        raise ValueError('Fixture did not distinguish the three support rules')
    verify(frozen)
    report = dict(state='all_28_native_knn_recipes_and_support_rules_fixture_verified',
                  verifier_sha256=digest(Path(__file__)), mapping_sha256=digest(mapping_path), frozen_inputs=frozen,
                  phases=phases, receipts=receipts, commands=commands, query_label_isolation=isolation,
                  support_rule_hashes=support_groups,
                  limitations=['Tiny fixtures, reduced joint rank and budgets; all 28 original kNN hyperparameter recipes are preserved.',
                               'All support weights are positive and all three retained support rules differ on this fixture.',
                               'No full production kNN inference or qualifying-score claim.'])
    write_new(report_path, report)
    print(json.dumps(dict(state=report['state'], audited_knn_stages=len(receipts), native_commands=len(commands))))


if __name__ == '__main__':
    main()
