#!/usr/bin/env python3
"""Score the predeclared E1 weighting arms with one unchanged P2-fit blender."""
import argparse
import copy
import gc
import json
from pathlib import Path
import shlex
import sys

import numpy as np

import rehearsal_blend as rehearsal
from local_context_chain import paired
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('w1', 'w2', 'w3')


def arm_spec(original, arm):
    if arm not in ARMS:
        raise ValueError('undeclared arm')
    spec = copy.deepcopy(original)
    replacements = {
        'artifacts/rehearsal/B-K1.f32': f'artifacts/rehearsal/e1/B-K1-{arm}.f32',
        'artifacts/rehearsal/B-knn256td.netflix.f32': f'artifacts/rehearsal/e1/B-knn256td-{arm}.f32',
        'artifacts/rehearsal/B-knn64.netflix.f32': f'artifacts/rehearsal/e1/B-knn64-{arm}.f32',
    }
    if not set(replacements).issubset(spec['b']['columns']):
        raise ValueError('core bank no longer contains the expected K1 family')
    if spec['b']['support'] != ['artifacts/rehearsal/B-knn256td.netflix.support.f32']:
        raise ValueError('unexpected core support schema')
    spec['b']['columns'] = [replacements.get(path, path) for path in spec['b']['columns']]
    spec['b']['support'] = [f'artifacts/rehearsal/e1/B-knn256td-{arm}.support.f32']
    return spec


def commands(arm):
    base = 'artifacts/rehearsal/e1/'
    return {
        f'E1-B-K1-{arm}': shlex.split(
            f'build/netflix-rowweights joint --data data/processed/netflix.bin '
            f'--model {base}B-K1-{arm}.bin --predictions {base}B-K1-{arm}.f32 '
            '--factors 256 --learning-rate .008 --regularization .02 --decay .97 '
            f'--iterations 41 --fixed-epoch --threads 5 --row-weights {base}p2-{arm}.f32'),
        f'E1-B-knn256td-{arm}': shlex.split(
            'build/netflix-rowweights knn --data data/processed/netflix.bin '
            f'--graph artifacts/r4/neighbors256.bin --joint-model {base}B-K1-{arm}.bin '
            '--neighbors 256 --power 1 --shrink .5 --time-decay .003 '
            f'--predictions {base}B-knn256td-{arm}.f32 --support {base}B-knn256td-{arm}.support.f32'),
        f'E1-B-knn64-{arm}': shlex.split(
            'build/netflix-rowweights knn --data data/processed/netflix.bin '
            f'--graph artifacts/neighbors64.bin --joint-model {base}B-K1-{arm}.bin '
            f'--power 1 --shrink .5 --predictions {base}B-knn64-{arm}.f32'),
    }


def decide(y, users, predictions):
    noise = paired(y, predictions['original'], predictions['w1'], users)
    results = {}
    for arm in ('w2', 'w3'):
        comparison = paired(y, predictions['w1'], predictions[arm], users)
        improvement = -comparison['delta_rmse']
        results[arm] = dict(comparison=comparison, passes_gate=(
            improvement >= 0.0005 and improvement > abs(noise['delta_rmse'])
            and comparison['all_folds_improve']))
    return dict(same_binary_noise=noise, arms=results,
                passes_gate=any(value['passes_gate'] for value in results.values()),
                lowest_rmse_arm=min(('w2', 'w3'), key=lambda arm: results[arm]['comparison']['treatment_rmse']),
                limitation='Reused original-probe development evidence; Stage C effect estimate, not a hidden-test score.')


def prepare(path):
    spec_path = ROOT/'artifacts/rehearsal/spec-core.json'
    baseline_path = ROOT/'artifacts/rehearsal/P0-core.json'
    spec = json.loads(spec_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    if spec != baseline['spec']:
        raise ValueError('core spec differs from the historical rehearsal')
    frozen = {str(ROOT/name): sha for name, sha in baseline['inputs'].items()}
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        source = getattr(module, '__file__', None)
        if source and Path(source).resolve().parent == ROOT/'scripts':
            sources.add(Path(source).resolve())
    for source in sources:
        frozen[str(source)] = digest(source)
    for source in [spec_path, baseline_path, ROOT/'scripts/e1_chain.sh', ROOT/'build/netflix-rowweights',
                   ROOT/'artifacts/r4/neighbors256.bin', ROOT/'artifacts/neighbors64.bin']:
        frozen[str(source)] = digest(source)
    for arm in ARMS:
        weight = ROOT/f'artifacts/rehearsal/e1/p2-{arm}.f32'
        report_path = Path(str(weight)+'.json')
        report = json.loads(report_path.read_text())
        if (digest(weight) != report['output_sha256'] or report['weight'] != int(arm[1:])
                or report['full_sha256'] != frozen[str(ROOT/spec['b']['data'])]
                or report['holdout_sha256'] != frozen[str(ROOT/spec['fit']['data'])]):
            raise ValueError('row weights changed')
        frozen[str(weight)] = report['output_sha256']
        frozen[str(report_path)] = digest(report_path)
    plan = dict(spec=spec, arm_specs={arm: arm_spec(spec, arm) for arm in ARMS}, frozen=frozen,
                commands={arm: commands(arm) for arm in ARMS},
                baseline_report=str(baseline_path), baseline_tolerance=1e-6,
                settings=dict(rounds=8000, learning_rate=.02, leaves=63, min_leaf=200,
                              threads=1, processes=1, seeds=[2026], skip_cv=True),
                protocol='Fit exactly once on unchanged P2 fit rows; score original,w1,w2,w3 without refitting or probe-selected hyperparameters.',
                gate='>=0.0005 better than w1, larger than absolute w1-vs-original noise, all five folds better.',
                lightgbm_version=rehearsal.lgb.__version__)
    if plan['lightgbm_version'] != baseline['lightgbm_version']:
        raise ValueError('LightGBM differs from historical rehearsal')
    with path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2)+'\n')


def score(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    # Fail before creating output or fitting anything on incomplete training.
    for path, expected in plan['frozen'].items():
        if digest(path) != expected:
            raise ValueError('frozen input changed: '+path)
    generated = {}
    for arm in ARMS:
        for name, command in plan['commands'][arm].items():
            path = ROOT/'experiments'/f'{name}.json'
            report = json.loads(path.read_text())
            if report['returncode'] != 0 or shlex.split(report['command']) != command:
                raise ValueError('incomplete or changed training recipe: '+name)
            if report['binary_sha256'] != plan['frozen'][str(ROOT/'build/netflix-rowweights')]:
                raise ValueError('training binary differs')
            generated[str(path)] = digest(path)
        for path in plan['arm_specs'][arm]['b']['columns'] + plan['arm_specs'][arm]['b']['support']:
            generated[str(ROOT/path)] = digest(ROOT/path)
    output.mkdir()
    fit = rehearsal.features(plan['spec']['fit'])
    settings = argparse.Namespace(**plan['settings'])
    predictor, rounds = rehearsal.train(fit['x'], fit['rx'], fit['y'], fit['users'], settings, 2026)
    del fit
    gc.collect()
    predictions = {}
    y = users = None
    for arm in ('original', *ARMS):
        spec = plan['spec'] if arm == 'original' else plan['arm_specs'][arm]
        data = rehearsal.features(spec['b'])
        raw = predictor(data['x'], data['rx'])
        if not np.isfinite(raw).all():
            raise ValueError('nonfinite prediction')
        predictions[arm] = raw
        raw.astype('<f4').tofile(output/f'{arm}.b.f32')
        if y is None:
            y, users = data['y'], data['users']
        elif not np.array_equal(y, data['y']) or not np.array_equal(users, data['users']):
            raise ValueError('arm query alignment differs')
        del data
        gc.collect()
    baseline = json.loads(Path(plan['baseline_report']).read_text())
    reproduced = rehearsal.rmse(predictions['original'], y)
    if abs(reproduced-baseline['b_rmse']) > plan['baseline_tolerance']:
        raise ValueError('historical baseline does not reproduce')
    result = decide(y, users, predictions)
    result.update(fit_rounds=rounds, baseline_rmse=reproduced,
                  historical_baseline_rmse=baseline['b_rmse'], plan_sha256=digest(plan_path),
                  generated=generated,
                  predictions={arm: digest(output/f'{arm}.b.f32') for arm in predictions})
    # Ensure no input changed while fitting/scoring.
    for path, expected in {**plan['frozen'], **generated}.items():
        if digest(path) != expected:
            raise ValueError('input changed during scoring: '+path)
    (output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'score'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.plan)
    else:
        if args.output is None:
            parser.error('score requires --output')
        score(args.plan, args.output)
