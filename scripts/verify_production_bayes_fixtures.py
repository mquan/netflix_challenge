"""Exercise retained Bayes sweep budgets on tiny single-thread native fixtures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from production_bayes_audit import audit, prepare
from production_rbm_audit import digest, verify, write_new

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_bayes import dataset
from test_native_fixed_epochs import query_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    output, report = args.output_dir.resolve(), args.report.resolve()
    if output.exists() or report.exists():
        raise FileExistsError('Use new fixture paths')
    registry_path = ROOT / 'experiments/production-bayes-recipes.json'
    registry = json.loads(registry_path.read_text())
    binary = registry['binary']
    helper_record = ROOT / 'experiments/P0-rbm-dataset-metadata-20261031.json'
    helper_info = json.loads(helper_record.read_text())
    helper = helper_info['binary']
    verify({binary: registry['binary_sha256'], helper: helper_info['binary_sha256']})
    paths = [registry_path, helper_record, Path(__file__), ROOT / 'tests/test_bayes.py',
             ROOT / 'tests/test_native_fixed_epochs.py', ROOT / 'scripts/production_bayes_audit.py',
             ROOT / 'scripts/production_rbm_audit.py', ROOT / 'scripts/recover_bayes_recipes.py']
    frozen = {str(path): digest(path) for path in paths}
    frozen.update({binary: digest(binary), helper: digest(helper)})
    output.mkdir(parents=True)
    commands, audits, plans = [], [], []

    def execute(command, log):
        with log.open('w') as stream:
            subprocess.run(['nice', '-n', '10', *command], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        commands.append(command)

    for label in (0, 5):
        folder = output / str(label)
        folder.mkdir()
        data = folder / 'data.bin'
        dataset(data, users=12, items=10, per_user=8)
        query_labels(data, label)
        meta = json.loads(subprocess.check_output([helper, str(data)], text=True))
        sizes = dict(user=12, item=10, **{'user-bag': 10, 'item-bag': 12, 'day': 60,
                                        'item-bin': 300, 'session': meta['sessions'], 'drift': 12})
        stages = []
        for recipe in registry['recipes']:
            parent = None
            for number, original in enumerate(recipe['stages']):
                index = len(stages)
                name = Path(recipe['column']).name + f'.stage{number}'
                directory = folder / name
                values = dict(BINARY=binary, DATASET=str(data), MODEL=str(directory / 'model.bin'),
                              PREDICTIONS=str(directory / 'model.f32'))
                if parent is not None:
                    values['MODEL_SWEEP32'] = str(Path(stages[parent]['output_dir']) / 'model.bin')
                elif len(recipe['stages']) == 2:
                    values['MODEL_SWEEP32'] = values['MODEL']
                cmd = [token.format(**values) for token in original['command_template']]
                original_threads = int(cmd[cmd.index('--threads') + 1])
                cmd[cmd.index('--threads') + 1] = '1'
                final = int(cmd[cmd.index('--samples') + 1])
                groups = {key: dict(mode=value['mode'], size=sizes[key])
                          for key, value in recipe['checkpoint_metadata']['groups'].items()}
                stages.append(dict(name=name, output_dir=str(directory), command=cmd, rank=recipe['checkpoint_metadata']['rank'],
                                   threads=1, historical_threads=original_threads, burn_in=16, groups=groups,
                                   start_sweep=stages[parent]['final_sweep'] if parent is not None else 0,
                                   final_sweep=final, dependency_index=parent))
                parent = index
        plan_path = folder / 'plan.json'
        plan = dict(data=str(data), dataset=meta, stages=stages,
                    frozen=dict(frozen, **{str(data): digest(data)}))
        write_new(plan_path, plan)
        plans.append(plan_path)
        for index, stage in enumerate(stages):
            prepare(plan_path, index)
            directory = Path(stage['output_dir'])
            execute(stage['command'], directory / 'train.log')
            reload = [binary, 'bayes', '--data', str(data), '--init-model', str(directory / 'model.bin'),
                      '--model', str(directory / 'reload.bin'), '--predictions', str(directory / 'reload.f32'),
                      '--samples', str(stage['final_sweep']), '--burn-in', '16', '--threads', '1']
            execute(reload, directory / 'reload.log')
            audit(plan_path, index)
            audits.append(directory / 'audit.json')
    left, right = [json.loads(path.read_text()) for path in plans]
    checks = []
    for a, b in zip(left['stages'], right['stages']):
        for filename in ('model.bin', 'model.f32'):
            first = (Path(a['output_dir']) / filename).read_bytes()
            second = (Path(b['output_dir']) / filename).read_bytes()
            if filename == 'model.bin':
                first, second = first[16:], second[16:]  # Dataset fingerprints include query labels.
            if first != second:
                raise ValueError('Query labels changed Bayes training or posterior averages')
        checks.append(dict(name=a['name'], rank=a['rank'], final_sweep=a['final_sweep'],
                           kept=a['final_sweep'] - 16, query_label_isolation=True))
    verify(frozen)
    outputs = {str(path): digest(path) for path in [*plans, *audits]}
    for receipt in audits:
        verify(json.loads(receipt.read_text())['outputs'])
    write_new(report, dict(state='verified_bayes_budget_fixtures', frozen=frozen, outputs=outputs,
                           checks=checks, audited_stages=len(audits), commands=commands,
                           limitation='Tiny 12-user/10-item dataset; exact retained ranks, sweep budgets and burn-in. '
                           'Host threads reduced to one for deterministic query-label isolation. Historical '
                           'four/five-thread execution and full production data remain unverified here. '
                           'No qualifying score or full retraining reproduction.'))
    print(json.dumps(dict(audited_stages=len(audits), pipeline_commands=len(commands), checks=checks)))


if __name__ == '__main__':
    main()
