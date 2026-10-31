"""Verify full K1 budgets and retained user-kNN exports on tiny native fixtures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from production_rbm_audit import digest, verify, write_new
from production_user_knn_audit import audit, prepare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_user_knn import write_dataset
from test_native_fixed_epochs import query_labels


def exercise(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    recipe_path = ROOT / 'experiments/production-user-knn-recipe.json'
    recipe = json.loads(recipe_path.read_text())
    evidence_path = Path(recipe['inference_evidence'])
    verify({str(evidence_path): recipe['inference_evidence_sha256']})
    evidence = json.loads(evidence_path.read_text())
    k1 = evidence['builds']['K1']['binary']
    rebuilt = evidence['builds']['user_knn']['binary']
    original = str(ROOT / recipe['user_knn']['original_binary'])
    helper_path = ROOT / 'experiments/P0-rbm-dataset-metadata-20261031.json'
    helper_info = json.loads(helper_path.read_text())
    helper = helper_info['binary']
    bindings = {k1: evidence['builds']['K1']['binary_sha256'],
                rebuilt: evidence['builds']['user_knn']['binary_sha256'],
                original: recipe['user_knn']['original_binary_sha256'], helper: helper_info['binary_sha256']}
    verify(bindings)
    frozen = dict(bindings)
    paths = [recipe_path, evidence_path, helper_path, Path(__file__), ROOT / 'tests/test_user_knn.py',
             ROOT / 'tests/test_native_fixed_epochs.py', ROOT / 'scripts/production_user_knn_audit.py',
             ROOT / 'scripts/production_joint_audit.py', ROOT / 'scripts/production_dependency_audit.py',
             ROOT / 'scripts/production_rbm_audit.py']
    frozen.update({str(p): digest(p) for p in paths})
    receipts, plans, commands = [], [], []

    def execute(command, log):
        with log.open('w') as stream:
            subprocess.run(['nice', '-n', '10', *command], stdout=stream, stderr=subprocess.STDOUT, check=True)
        commands.append(command)

    for label in (0, 5):
        folder = output / str(label)
        folder.mkdir()
        data = folder / 'data.bin'
        write_dataset(data)
        query_labels(data, label)
        meta = json.loads(subprocess.check_output([helper, str(data)], text=True))
        k1root, knnroot = folder / 'K1', folder / 'user-knn'
        values = dict(REBUILT_K1_BINARY=k1, PRODUCTION_DATASET=str(data),
                      PRODUCTION_K1_MODEL=str(k1root / 'model.bin'),
                      PRODUCTION_K1_QUERY_PREDICTIONS=str(k1root / 'model.f32'),
                      VERIFIED_USER_KNN_BINARY=original, QUALIFYING_PREDICTIONS=str(knnroot / 'model.f32'),
                      QUALIFYING_SUPPORT=str(knnroot / 'model.support.f32'),
                      QUALIFYING_COUNTS=str(knnroot / 'model.count.f32'))
        training = [token.format(**values) for token in recipe['K1']['production_command_template']]
        training[training.index('--threads') + 1] = '1'
        inference = [token.format(**values) for token in recipe['user_knn']['production_command_template']]
        stages = [dict(name='K1-production-fixture', family='joint', command=training, output_dir=str(k1root),
                       dependency_index=None, start_epoch=1, fixed_final_epoch=41, rank=256,
                       day_rank=0, model_format=2, seed=2026, threads=1),
                  dict(name='user-knn-production-fixture', family='user_knn', command=inference,
                       output_dir=str(knnroot), threads=3)]
        plan_path = folder / 'plan.json'
        write_new(plan_path, dict(data=str(data), dataset=meta, stages=stages,
                                  frozen=dict(frozen, **{str(data): digest(data)})))
        plans.append(plan_path)
        prepare(plan_path, 0)
        execute(training, k1root / 'train.log')
        execute([k1, 'eval-joint', '--data', str(data), '--model', str(k1root / 'model.bin'),
                 '--predictions', str(k1root / 'reload.f32')], k1root / 'reload.log')
        audit(plan_path, 0)
        receipts.append(k1root / 'audit.json')
        prepare(plan_path, 1)
        execute(inference, knnroot / 'run.log')
        replay = list(inference)
        for flag, suffix in [('--predictions', '.f32'), ('--support', '.support.f32'), ('--support-count', '.count.f32')]:
            replay[replay.index(flag) + 1] = str(knnroot / ('replay' + suffix))
        execute(replay, knnroot / 'replay.log')
        audit(plan_path, 1)
        receipts.append(knnroot / 'audit.json')
        replay[0] = rebuilt
        for flag, suffix in [('--predictions', '.f32'), ('--support', '.support.f32'), ('--support-count', '.count.f32')]:
            replay[replay.index(flag) + 1] = str(knnroot / ('rebuilt' + suffix))
        execute(replay, knnroot / 'rebuilt.log')
        for suffix in ('.f32', '.support.f32', '.count.f32'):
            if (knnroot / ('model' + suffix)).read_bytes() != (knnroot / ('rebuilt' + suffix)).read_bytes():
                raise ValueError('Original/rebuilt inference mismatch')
        support = np.fromfile(knnroot / 'model.support.f32', '<f4')
        counts = np.fromfile(knnroot / 'model.count.f32', '<f4')
        if not np.any(support > 0) or not np.any(counts > 0):
            raise ValueError('Degenerate user-kNN support fixture')
        if (knnroot / 'model.f32').read_bytes() == (k1root / 'model.f32').read_bytes():
            raise ValueError('Degenerate user-kNN correction fixture')
    for subpath in ('K1/model.bin', 'K1/model.f32', 'user-knn/model.f32',
                    'user-knn/model.support.f32', 'user-knn/model.count.f32'):
        left, right = [(output / str(label) / subpath).read_bytes() for label in (0, 5)]
        if subpath.endswith('.bin'):
            left, right = left[16:], right[16:]
        if left != right:
            raise ValueError('Query labels changed K1 or user-kNN outputs')
    verify(frozen)
    for receipt in receipts:
        verify(json.loads(receipt.read_text())['outputs'])
    return dict(state='verified_user_knn_production_fixtures', frozen=frozen,
                outputs={str(path): digest(path) for path in [*plans, *receipts]}, commands=commands,
                audited_K1_models=2, audited_user_knn_stages=2, query_label_isolation=True,
                original_rebuilt_inference_exact=True, nonzero_support_and_correction=True,
                limitation='Tiny 80-user/8-item fixture; K1 exact rank256 and 41 epochs, reduced to one training '
                'thread for deterministic label isolation. User-kNN preserves three inference threads. '
                'Historical six-thread Hogwild K1 production training remains outstanding; frozen blender '
                'latent basis is not replaced. No full production outputs or qualifying score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = exercise(args.output_dir)
    write_new(args.report, result)
    print(json.dumps({key: result[key] for key in ('state', 'audited_K1_models', 'audited_user_knn_stages',
                                                  'query_label_isolation', 'nonzero_support_and_correction')}))
