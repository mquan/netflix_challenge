"""Exercise all six retained kernel columns using newly audited tiny J3 models."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from production_joint_audit import audit as audit_joint, prepare as prepare_joint
from production_kernel_audit import audit, prepare, suffixes
from production_rbm_audit import digest, verify, write_new

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_bayes import dataset
from test_native_fixed_epochs import query_labels


def exercise(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    registry_path = ROOT / 'experiments/production-kernel-recipes.json'
    joint_path = ROOT / 'experiments/P0-JOINT-20261031-plan.json'
    helper_path = ROOT / 'experiments/P0-rbm-dataset-metadata-20261031.json'
    registry, joint, helper_info = [json.loads(p.read_text()) for p in (registry_path, joint_path, helper_path)]
    helper = helper_info['binary']
    binary = joint['stages'][2]['command'][0]
    verify({helper: helper_info['binary_sha256'], binary: joint['frozen'][binary]})
    frozen = {str(p): digest(p) for p in [registry_path, joint_path, helper_path, Path(__file__),
              ROOT / 'scripts/production_kernel_audit.py', ROOT / 'scripts/production_joint_audit.py',
              ROOT / 'scripts/production_dependency_audit.py', ROOT / 'scripts/production_rbm_audit.py',
              ROOT / 'tests/test_bayes.py', ROOT / 'tests/test_native_fixed_epochs.py']}
    frozen.update({helper: digest(helper), binary: digest(binary)})
    for recipe in registry['recipes']:
        verify({recipe['binary']: recipe['binary_sha256']})
        frozen[recipe['binary']] = recipe['binary_sha256']
    receipts, plans, commands, proofs = [], [], [], []

    def execute(command, log):
        with log.open('w') as stream:
            subprocess.run(['nice', '-n', '10', *command], stdout=stream, stderr=subprocess.STDOUT, check=True)
        commands.append(command)

    for label in (0, 5):
        folder = output / str(label)
        folder.mkdir()
        data = folder / 'data.bin'
        dataset(data, users=12, items=10, per_user=8)
        query_labels(data, label)
        meta = json.loads(subprocess.check_output([helper, str(data)], text=True))
        base = copy.deepcopy(joint['stages'][2])
        base_root = folder / 'J3'
        base['output_dir'] = str(base_root)
        for flag, value in [('--data', str(data)), ('--model', str(base_root / 'model.bin')),
                            ('--predictions', str(base_root / 'model.f32'))]:
            base['command'][base['command'].index(flag) + 1] = value
        native_plan = folder / 'joint-plan.json'
        write_new(native_plan, dict(data=str(data), dataset=meta, frozen=dict(frozen, **{str(data): digest(data)}),
                                    stages=[base]))
        prepare_joint(native_plan, 0)
        execute(base['command'], base_root / 'train.log')
        execute([binary, 'eval-joint', '--data', str(data), '--model', str(base_root / 'model.bin'),
                 '--predictions', str(base_root / 'reload.f32')], base_root / 'reload.log')
        audit_joint(native_plan, 0)
        receipts.append(base_root / 'audit.json')
        stages = []
        for recipe in registry['recipes']:
            directory = folder / recipe['name']
            prefix = directory / 'model'
            values = dict(VERIFIED_MOONSHOT_GP_INITIAL_BINARY=recipe['binary'],
                          VERIFIED_MOONSHOT_GP_PERSISTENT_BINARY=recipe['binary'],
                          VERIFIED_MOONSHOT_LOCAL_BINARY=recipe['binary'], PRODUCTION_DATASET=str(data),
                          PRODUCTION_J3_MODEL=str(base_root / 'model.bin'), OUTPUT_PREFIX=str(prefix))
            cmd = [token.format(**values) for token in recipe['production_command_template']]
            stages.append(dict(name=recipe['name'], family=recipe['family'], parameters=recipe['parameters'],
                               command=cmd, output_dir=str(directory), threads=1))
        plan_path = folder / 'kernel-plan.json'
        write_new(plan_path, dict(data=str(data), dataset=meta, frozen=dict(frozen, **{str(data): digest(data)}),
                                  source_phase=dict(plan=str(native_plan), plan_sha256=digest(native_plan),
                                                    stage=0, model=str(base_root / 'model.bin')), stages=stages))
        plans.append(plan_path)
        for index, stage in enumerate(stages):
            prepare(plan_path, index)
            directory = Path(stage['output_dir'])
            execute(stage['command'], directory / 'run.log')
            replay = list(stage['command'])
            replay[replay.index('--output') + 1] = str(directory / 'replay')
            execute(replay, directory / 'replay.log')
            audit(plan_path, index)
            receipts.append(directory / 'audit.json')
    left, right = [json.loads(p.read_text()) for p in plans]
    for a, b in zip(left['stages'], right['stages']):
        for suffix in suffixes(a):
            if (Path(a['output_dir']) / ('model' + suffix)).read_bytes() != (Path(b['output_dir']) / ('model' + suffix)).read_bytes():
                raise ValueError('Kernel output depends on query labels')
        folder = Path(a['output_dir'])
        if a['family'] == 'local_residual':
            if (folder / 'model.epoch1.f32').read_bytes() == (folder / 'model.epoch6.f32').read_bytes():
                raise ValueError('Degenerate local budget fixture')
        else:
            variants = [(folder / f'model.{i}.f32').read_bytes() for i in range(3)]
            if len(set(variants)) != 3 or variants[0] == (folder / 'model.base.f32').read_bytes():
                raise ValueError('Degenerate GP noise/correction fixture')
        proofs.append(dict(name=a['name'], label_isolation=True, complete_replay_exact=True, nondegenerate=True))
    if (Path(left['stages'][0]['output_dir']) / 'model.0.f32').read_bytes() == (Path(left['stages'][1]['output_dir']) / 'model.0.f32').read_bytes():
        raise ValueError('Degenerate persistent-kernel fixture')
    verify(frozen)
    for receipt in receipts:
        verify(json.loads(receipt.read_text())['outputs'])
    return dict(state='verified_production_kernel_fixtures', frozen=frozen,
                outputs={str(p): digest(p) for p in [*plans, *receipts]}, commands=commands,
                audited_J3_models=2, audited_kernel_stages=8, retained_columns_per_fixture=6, checks=proofs,
                limitation='Tiny 12-user/10-item fixtures; exact retained ranks, six-epoch local schedules and '
                'kernel parameters, one thread. New J3 rank128 epoch9 models audited separately. '
                'No full production outputs or qualifying score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = exercise(args.output_dir)
    write_new(args.report, result)
    print(json.dumps({k: result[k] for k in ('state', 'audited_J3_models', 'audited_kernel_stages', 'checks')}))
