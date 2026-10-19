#!/usr/bin/env python3
"""N1 chain after selection: honest P2 gate -> full-T refit -> blend test.

1. Waits for N1-select to finish, then runs hr_honest_gate.py on its selected
   checkpoint (P2 column written from it).
2. If the candidate adds <= --gate (default -0.0010 RMSE) beyond A2h-select on
   the 10 other honest columns, refits on full T at the selected epoch with the
   same settings (--save-epochs), writing the probe column.
3. Runs hr_blend_chain.py --columns <probe column> (R9-LAT-A2h + N1).
Status: experiments/N1-chain.status.json.
"""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = pathlib.Path('/Users/quan/workspace/netflix_challenge/artifacts/residual-chains')
DATA = '/Users/quan/workspace/netflix_challenge/data/processed/netflix.bin'
P2 = '/Users/quan/workspace/netflix_challenge/artifacts/validation/train-matched-20260927.bin'
PY = sys.executable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='N1')
    p.add_argument('--gate', type=float, default=-0.0010)
    a = p.parse_args()
    os.chdir(ROOT)
    status_path = ROOT / 'experiments' / f'{a.name}-chain.status.json'

    def status(**v):
        v['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_path.write_text(json.dumps(v, indent=2) + '\n')

    def run(stage, strategy, cmd):
        status(state='running', stage=stage)
        subprocess.run([PY, 'scripts/experiment.py', '--name', stage, '--strategy', strategy, '--', *cmd], check=True)

    select = ART / f'{a.name}-select.json'
    try:
        while 'selected_epoch' not in json.loads(select.read_text()) or subprocess.run(
                ['pgrep', '-f', f'{a.name}-select'], capture_output=True).returncode == 0:
            status(state='waiting', stage=f'{a.name}-select')
            time.sleep(300)
        report = json.loads(select.read_text())
        epoch = report['selected_epoch']
        gate_json = ROOT / 'experiments' / f'{a.name}-gate.result.json'
        run(f'{a.name}-gate', 'Honest P2 ridge gate: candidate beyond A2h-select on 10 other honest columns',
            [PY, 'scripts/hr_honest_gate.py', '--selection-data', P2,
             '--checkpoint', str(ART / f'{a.name}-select.select.pt'),
             '--column', str(ART / f'{a.name}-select.p2.f32'), '--output', str(gate_json)])
        gate = json.loads(gate_json.read_text())
        delta = gate['candidate_beyond_A2h']['delta_rmse']
        if delta > a.gate:
            status(state='stopped', reason=f'gate failed: {delta:+.6f} > {a.gate}', gate=gate)
            return
        args = dict(report['arguments'])
        args.update(data=DATA, output=str(ART / f'{a.name}-refit'), epochs=epoch, save_epochs=True)
        args.pop('selection_data'); args.pop('selection_only')
        cmd = [PY, 'scripts/choice_sequence.py']
        for k, v in args.items():
            f = '--' + k.replace('_', '-')
            if isinstance(v, bool):
                cmd += [f] if v else []
            else:
                cmd += [f, str(v)]
        run(f'{a.name}-refit', f'Full-T refit of {a.name} at selected epoch {epoch}', cmd)
        run(f'{a.name}-blend-chain', f'R9-LAT-A2h + {a.name} probe column',
            [PY, 'scripts/hr_blend_chain.py', '--name', f'{a.name}-blend',
             '--columns', str(ART / f'{a.name}-refit.f32'), '--processes', '3'])
        status(state='complete', gate=gate,
               blend=json.loads((ROOT / 'experiments' / f'{a.name}-blend.status.json').read_text()))
    except BaseException as error:
        status(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
