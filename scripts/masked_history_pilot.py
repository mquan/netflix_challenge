#!/usr/bin/env python3
"""Bound existing selection runs, refit, and evaluate their frozen-bank blends.

Only explicitly supplied trainer PIDs whose command matches the named output
are stopped. Completed selection epochs are preserved before requesting a stop.
This controller does not import either Torch or LightGBM.
"""
import argparse
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time


def read(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def stop_trainer(pid, name):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True)
    if result.returncode:
        return
    argv = shlex.split(result.stdout.strip())
    expected = 'artifacts/masked/' + name
    if len(argv) < 2 or pathlib.Path(argv[1]).name != 'masked_history.py':
        raise RuntimeError(f'PID {pid} is not a masked-history trainer')
    if '--output' not in argv or argv[argv.index('--output') + 1] != expected:
        raise RuntimeError(f'PID {pid} output does not match {expected}')
    os.kill(pid, signal.SIGTERM)


def launch(command, name):
    # The experiment/comparison children keep their own timestamped logs.
    with open('/tmp/' + name + '-controller.log', 'x') as stream:
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)


def main(a):
    jobs = []
    for entry in a.selection:
        name, pid = entry.split('=')
        if pathlib.Path(name).name != name:
            raise ValueError('invalid run name')
        jobs.append(dict(name=name, pid=int(pid), phase='selecting'))
    path = pathlib.Path('experiments/MH-pilot-progress.json')
    if path.exists():
        raise FileExistsError(path)
    record = dict(selection_epoch_budget=a.epochs, rationale='Bound first pilot after measured full-epoch throughput',
                  jobs=jobs, combined='all three completed raw predictions', started=time.time())
    processes, blend = {}, None
    while True:
        for job in jobs:
            name = job['name']
            if job['phase'] == 'selecting':
                progress = read('artifacts/masked/' + name + '.select.progress.json')
                if not progress or len(progress['records']) < a.epochs:
                    experiment = read('experiments/' + name + '.json')
                    if experiment and 'exit_code' in experiment:
                        raise RuntimeError(f'{name} exited before the selection budget')
                    continue
                records = progress['records'][:a.epochs]
                selected = min(records, key=lambda r: r['validation_rmse'])['epoch']
                job.update(selection=records, selected_epoch=selected)
                stop_trainer(job['pid'], name)
                original = read('artifacts/masked/' + name + '.json')
                command = [sys.executable, 'scripts/experiment.py', '--name', name + '-refit',
                           '--strategy', f'Fresh full-data refit at epoch {selected}, selected only from first {a.epochs} training-holdout epochs of {name}',
                           '--', sys.executable, 'scripts/masked_history.py', '--data', original['arguments']['data'],
                           '--output', 'artifacts/masked/' + name + '-refit', '--epochs', str(selected),
                           '--threads', str(original['arguments']['threads'])]
                for key, value in original['config'].items():
                    command += ['--' + key.replace('_', '-'), str(value)]
                job.update(phase='refitting', refit_command=command)
                processes[name] = launch(command, name + '-refit')
                print(json.dumps(dict(name=name, phase=job['phase'], selected_epoch=selected)), flush=True)
            if job['phase'] == 'refitting' and processes[job['name']].poll() is not None:
                if processes[job['name']].returncode:
                    raise RuntimeError(f'{name} refit failed')
                job['phase'] = 'ready_for_blend'
                job['prediction'] = 'artifacts/masked/' + name + '-refit.f32'
                job['probe_rmse'] = read('artifacts/masked/' + name + '-refit.json')['probe_rmse']
                print(json.dumps(dict(name=name, phase=job['phase'], probe_rmse=job['probe_rmse'])), flush=True)
        if blend is not None and blend.poll() is not None:
            if blend.returncode:
                raise RuntimeError('blend failed; see its controller log')
            if current is None:
                record['combined_result'] = read('artifacts/masked/MHB-all.comparison.json')['comparison']
                record['finished'] = time.time()
                path.write_text(json.dumps(record, indent=2) + '\n')
                print(json.dumps(record['combined_result']), flush=True)
                return
            current['phase'] = 'complete'
            current['blend_result'] = read(current['blend_output'] + '.comparison.json')['comparison']
            print(json.dumps(dict(name=current['name'], **current['blend_result'])), flush=True)
            blend = None
        control = read('artifacts/masked/MHB-control.comparison.json')
        if control and abs(control['comparison']['delta_rmse']) > 1e-7:
            raise RuntimeError('frozen control failed to reproduce R5')
        if blend is None and control:
            current = next((j for j in jobs if j['phase'] == 'ready_for_blend'), None)
            if current:
                current['phase'] = 'blending'
                current['blend_output'] = 'artifacts/masked/MHB-' + current['name'].split('-')[0]
                command = [sys.executable, 'scripts/masked_history_compare.py', '--output', current['blend_output'],
                           '--prediction', current['prediction'], '--processes', str(a.processes)]
                blend = launch(command, pathlib.Path(current['blend_output']).name)
            elif all(j['phase'] == 'complete' for j in jobs):
                command = [sys.executable, 'scripts/masked_history_compare.py', '--output', 'artifacts/masked/MHB-all',
                           '--processes', str(a.processes)]
                for job in jobs:
                    command += ['--prediction', job['prediction']]
                blend = launch(command, 'MHB-all')
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(tmp, path)
        time.sleep(5)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', action='append', required=True, help='run-name=trainer-pid')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--processes', type=int, default=2)
    main(p.parse_args())
