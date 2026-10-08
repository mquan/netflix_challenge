#!/usr/bin/env python3
"""Run the predeclared latent-feature experiment and record paired fold deltas."""
import json
import pathlib
import subprocess
import sys
from round9_latent import digest


def run(name, strategy, cmd):
    subprocess.run([sys.executable, 'scripts/experiment.py', '--name', name,
                    '--strategy', strategy, '--', sys.executable, *cmd], check=True)


def main():
    base = json.loads(pathlib.Path('artifacts/r8/R8-B4-A2e-s3.json').read_text())
    latent = json.loads(pathlib.Path('artifacts/r9/latent/manifest.json').read_text())
    for path, expected in latent['outputs'].items():
        if digest(path) != expected: raise ValueError(path)
    config = dict(base['config']); config['output'] = 'artifacts/r9/R9-LAT-s3'
    config['extra_support'] = [*config['extra_support'], *latent['features']]
    config['processes'] = 3
    args = ['scripts/round4_boost.py']
    for name, value in config.items():
        flag = '--'+name.replace('_', '-')
        if isinstance(value, bool):
            if value: args.append(flag)
        elif isinstance(value, list): args += [flag, *map(str, value)]
        else: args += [flag, str(value)]
    run('R9-LAT-s3', 'Promoted R8 bank plus 8 user and 8 item PCs and 6 exact bias terms; three seeds', args)
    run('R9-LAT-NN2', 'Same NN2 settings and fixed 50/50 average, with latent features',
        ['scripts/round8_nn_blend.py', '--data', config['data'], '--like', config['output']+'.json',
         '--output', 'artifacts/r9/R9-LAT-NN2', '--lr', '3e-4', '--dropout', '.2',
         '--seeds', '2026', '7', '13', '--epochs', '60', '--patience', '5', '--threads', '2'])
    control = json.loads(pathlib.Path('artifacts/r8/R8-B4-A2e-NN2.json').read_text())
    result = json.loads(pathlib.Path('artifacts/r9/R9-LAT-NN2.json').read_text())
    delta = [x-y for x,y in zip(result['avg_fold_rmse'], control['avg_fold_rmse'])]
    summary = dict(control=control['avg_rmse'], treatment=result['avg_rmse'],
                   delta=result['avg_rmse']-control['avg_rmse'], fold_delta=delta,
                   promote=all(x < 0 for x in delta),
                   metric='Development user-fold CV; not a final test score')
    pathlib.Path('artifacts/r9/latent-comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
