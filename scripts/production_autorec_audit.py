"""Audit complete fixed-budget AutoRec trajectories and selected snapshots."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from production_rbm_audit import digest, load_plan, verify, write_new


def checkpoint(path, epoch, plan, stage):
    import torch
    torch.set_num_threads(1)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    expected = dict(epoch=epoch, config=stage['config'], items=plan['dataset']['items'],
                    torch_threads=1, torch_version=plan['torch_version'], numpy_version=plan['numpy_version'],
                    source_sha256=plan['source_sha256'])
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError('AutoRec checkpoint epoch, configuration, libraries or source differs')
    config = stage['config']
    shapes, width = {}, 2 * plan['dataset']['items']
    for index in range(config['layers']):
        shapes[f'{2 * index}.weight'] = (config['hidden'], width)
        shapes[f'{2 * index}.bias'] = (config['hidden'],)
        width = config['hidden']
    index = 2 * config['layers'] + 1
    shapes[f'{index}.weight'] = (plan['dataset']['items'], width)
    shapes[f'{index}.bias'] = (plan['dataset']['items'],)
    state = saved.get('state', {})
    if set(state) != set(shapes) or any(
            tuple(state[key].shape) != shape or state[key].dtype != torch.float32 or
            not torch.isfinite(state[key]).all().item() for key, shape in shapes.items()):
        raise ValueError('AutoRec checkpoint tensor names, shapes, dtype or values differ')


def started_record(plan, plan_sha, index, stage):
    device_verification = {}
    if stage['config']['device'] == 'mps':
        proof_path = Path(plan['device_verification'])
        proof = json.loads(proof_path.read_text())
        expected = dict(state='verified_autorec_device_parity', device='mps', host_threads=1,
                        source_sha256=plan['source_sha256'], torch_version=plan['torch_version'],
                        numpy_version=plan['numpy_version'])
        required_checks = ('original_final_and_snapshot_parity', 'final_and_snapshot_reload_parity',
                           'query_label_independent_predictions_and_weights', 'distinct_snapshot_and_final')
        if (any(proof.get(k) != v for k, v in expected.items()) or
                any(proof.get('checks', {}).get(k) is not True for k in required_checks)):
            raise ValueError('Matching MPS-specific AutoRec verification required')
        verify(proof['inputs'])
        verify(proof['outputs'])
        device_verification = {str(proof_path): digest(proof_path), **proof['outputs']}
    return dict(state='prepared', stage=index, plan_sha256=plan_sha, command=stage['command'],
                data_sha256=plan['frozen'][plan['data']], config=stage['config'],
                source_sha256=plan['source_sha256'], torch_version=plan['torch_version'],
                numpy_version=plan['numpy_version'], device_verification=device_verification)


def prepare(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    record = started_record(plan, plan_sha, index, stage)
    if digest(path) != plan_sha:
        raise ValueError('AutoRec phase changed')
    root = Path(stage['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'started.json', record)
    return record


def audit(plan_path, index):
    path, plan_sha, plan, stage = load_plan(plan_path, index)
    root = Path(stage['output_dir'])
    if (root / 'audit.json').exists():
        raise FileExistsError(root / 'audit.json')
    started = json.loads((root / 'started.json').read_text())
    if started != started_record(plan, plan_sha, index, stage):
        raise ValueError('AutoRec started configuration changed')
    config = stage['config']
    if not config['fixed_epoch'] or config['reload_checkpoint'] is not None:
        raise ValueError('Fixed-budget AutoRec training required')
    epochs = sorted({config['epochs'], *config['save_epoch']})
    if [item['epoch'] for item in stage['outputs']] != epochs:
        raise ValueError('Missing or unexpected selected trajectory epochs')
    bound = {str(root / name): digest(root / name) for name in ('started.json', 'train.log')}
    log = (root / 'train.log').read_text()
    records = re.findall(r'^epoch=(\d+) train_rmse=(\S+) query_labels=unknown seconds=(\S+)$', log, re.M)
    if ([int(r[0]) for r in records] != list(range(1, config['epochs'] + 1)) or
            'probe_rmse' in log or 'early_stop' in log or
            any(not np.isfinite(float(v)) or float(v) < 0 for r in records for v in r[1:])):
        raise ValueError('Incomplete AutoRec budget or query scoring')
    header = re.findall(r'^autorec hidden=(\d+) layers=(\d+) batch=(\d+) lr=(\S+) '
                        r'input_drop=(\S+) dropout=(\S+) wd=(\S+) device=(\S+)$', log, re.M)
    if len(header) != 1:
        raise ValueError('Missing AutoRec training header')
    h = header[0]
    expected = [config['hidden'], config['layers'], config['batch'], config['lr'], config['input_drop'], config['dropout'], config['weight_decay']]
    if [float(v) for v in h[:7]] != expected or h[7] != config['device']:
        raise ValueError('AutoRec training header differs')
    columns = []
    for selected in stage['outputs']:
        files = {name: root / selected[name] for name in ('prediction', 'checkpoint', 'replay', 'replay_log')}
        bound.update({str(p): digest(p) for p in files.values()})
        checkpoint(files['checkpoint'], selected['epoch'], plan, stage)
        prediction = files['prediction']
        if prediction.stat().st_size != plan['dataset']['query_count'] * 4:
            raise ValueError('Wrong AutoRec qualifying prediction count')
        if not np.isfinite(np.fromfile(prediction, '<f4')).all():
            raise ValueError('Nonfinite AutoRec qualifying predictions')
        if bound[str(prediction)] != bound[str(files['replay'])]:
            raise ValueError('AutoRec checkpoint reload differs')
        replay_log = files['replay_log'].read_text()
        expected = f"reload_epoch={selected['epoch']} prediction_count={plan['dataset']['query_count']}"
        if replay_log.strip() != expected:
            raise ValueError('AutoRec reload record differs')
        columns.append(dict(column=selected['column'], development_sha256=selected['development_sha256'],
                            epoch=selected['epoch'], qualifying_path=str(prediction), qualifying_sha256=bound[str(prediction)]))
    verify(plan['frozen'])
    verify(started['device_verification'])
    verify(bound)
    if digest(path) != plan_sha:
        raise ValueError('AutoRec phase changed')
    result = dict(state='verified_complete', stage=index, plan_sha256=plan_sha,
                  group=stage['group'], fixed_epochs=config['epochs'], columns=columns,
                  outputs=bound, qualifying_count=plan['dataset']['query_count'], metric=None,
                  scope='Fixed AutoRec trajectory and its retained qualifying snapshots; no qualifying score or promotion.')
    write_new(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print((prepare if args.prepare else audit)(args.plan, args.stage)['state'])
