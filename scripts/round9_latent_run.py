#!/usr/bin/env python3
"""Run the predeclared latent-feature experiment and record paired fold deltas."""
import argparse
import json
import pathlib
import subprocess
import sys
from round9_latent import digest


def run(name, strategy, cmd):
    subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                    '--strategy', strategy, '--', sys.executable, *cmd], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='R9-LAT')
    parser.add_argument('--base', default='artifacts/r8/R8-B4-A2e-s3.json')
    parser.add_argument('--control', default='artifacts/r8/R8-B4-A2e-NN2.json')
    parser.add_argument('--processes', type=int, default=3)
    parser.add_argument('--nn-threads', type=int, default=2)
    options = parser.parse_args()
    if options.processes < 1 or options.nn_threads < 1:
        parser.error('processes and nn-threads must be positive')
    if pathlib.Path(options.run_id).name != options.run_id:
        parser.error('run-id must be a filename')
    tree_name = options.run_id + '-s3'
    nn_name = options.run_id + '-NN2'
    comparison_path = pathlib.Path('artifacts/r9') / (options.run_id + '-comparison.json')
    for name in (tree_name, nn_name):
        if list(pathlib.Path('experiments').glob(name + '.*')) or list(pathlib.Path('artifacts/r9').glob(name + '.*')):
            parser.error('run already exists; use a new run-id')
    if comparison_path.exists():
        parser.error('comparison already exists; use a new run-id')
    base = json.loads(pathlib.Path(options.base).read_text())
    control = json.loads(pathlib.Path(options.control).read_text())
    if pathlib.Path(control['config']['like']).resolve() != pathlib.Path(options.base).resolve():
        parser.error('control must be the neural blend of the supplied base')
    for path, expected in base['inputs'].items():
        source = pathlib.Path(path)
        if not source.exists():
            source = pathlib.Path(base['config']['bank_root']) / path
        if digest(source) != expected:
            raise ValueError(str(source))
    latent = json.loads(pathlib.Path('artifacts/r9/latent/manifest.json').read_text())
    for path, expected in {**latent['inputs'], **latent['outputs']}.items():
        if digest(path) != expected: raise ValueError(path)
    config = dict(base['config']); config['output'] = 'artifacts/r9/' + tree_name
    config['extra_support'] = [*config['extra_support'], *latent['features']]
    config['processes'] = options.processes
    args = ['scripts/round4_boost.py']
    for name, value in config.items():
        flag = '--'+name.replace('_', '-')
        if isinstance(value, bool):
            if value: args.append(flag)
        elif isinstance(value, list): args += [flag, *map(str, value)]
        else: args += [flag, str(value)]
    run(tree_name, 'Frozen ' + options.base + ' bank plus 8 user and 8 item PCs and 6 exact bias terms; three seeds', args)
    run(nn_name, 'Same NN2 settings and fixed 50/50 average, with latent features',
        ['scripts/round8_nn_blend.py', '--data', config['data'], '--like', config['output']+'.json',
         '--output', 'artifacts/r9/' + nn_name, '--lr', '3e-4', '--dropout', '.2',
         '--seeds', '2026', '7', '13', '--epochs', '60', '--patience', '5', '--threads', str(options.nn_threads)])
    result = json.loads(pathlib.Path('artifacts/r9/' + nn_name + '.json').read_text())
    delta = [x-y for x,y in zip(result['avg_fold_rmse'], control['avg_fold_rmse'])]
    tree = json.loads(pathlib.Path(config['output'] + '.json').read_text())
    summary = dict(control=control['avg_rmse'], treatment=result['avg_rmse'],
                   control_report=options.control, base_report=options.base,
                   tree_control=base['refit_rmse'], tree_treatment=tree['refit_rmse'],
                   tree_delta=tree['refit_rmse']-base['refit_rmse'],
                   delta=result['avg_rmse']-control['avg_rmse'], fold_delta=delta,
                   promote=all(x < 0 for x in delta),
                   metric='Development user-fold CV; not a final test score')
    comparison_path.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
