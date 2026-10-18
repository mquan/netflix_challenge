#!/usr/bin/env python3
"""Replay the frozen R4-B blend and add one completed experiment at a time."""
import argparse
import json
import pathlib
import subprocess
import sys
import time

p = argparse.ArgumentParser(__doc__)
p.add_argument('--variants', nargs='+', default=['baseline', 'neighbor-control', 'neighbor-temporal-v2', 'rbm-control', 'rbm-context', 'reference'])
p.add_argument('--processes', type=int, default=2)
a = p.parse_args()
root = pathlib.Path('artifacts/push/bank')
base = json.loads(pathlib.Path('experiments/PUSH-baseline.json').read_text())['config']
pathlib.Path('artifacts/r4').mkdir(exist_ok=True)
for variant in a.variants:
    additions = []
    if variant != 'baseline':
        manifest = pathlib.Path('experiments/PUSH-'+variant+'.json')
        while True:
            if manifest.exists():
                r = json.loads(manifest.read_text())
                if 'exit_code' in r:
                    if r['exit_code'] != 0: raise RuntimeError(f'{variant} failed')
                    break
            time.sleep(10)
        additions = ['artifacts/push/'+('reference-tx60' if variant == 'reference' else variant)+'.f32']
    output = 'artifacts/push/blend-'+variant
    if pathlib.Path(output+'.json').exists():
        print('already completed', variant, flush=True)
        continue
    cmd = [sys.executable, 'scripts/round4_boost.py', '--data', '/Users/quan/workspace/netflix_challenge/data/processed/netflix.bin',
           '--output', output, '--bank-root', str(root), '--extra']
    cmd += [str(root/x) for x in base['extra']] + additions
    cmd += ['--extra-support']+[str(root/x) for x in base['extra_support']]
    for flag in ['rounds', 'learning_rate', 'leaves', 'min_leaf']:
        cmd += ['--'+flag.replace('_','-'), str(base[flag])]
    cmd += ['--processes', str(a.processes)]
    pathlib.Path('experiments/PUSH-blend-'+variant+'-command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    print('starting', variant, flush=True)
    with open('experiments/PUSH-blend-'+variant+'.log', 'w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    r = json.loads(pathlib.Path(output+'.json').read_text())
    print(json.dumps({'variant':variant, 'rmse':r['refit_rmse'],'ridge_rmse':r['ridge_rmse']}),flush=True)
