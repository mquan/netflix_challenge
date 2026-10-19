#!/usr/bin/env python3
"""Continue a selection-objective checkpoint without resetting its learned head.

Legacy SO1 checkpoints omit optimizer/RNG state and require an explicit restart.
Checkpoints written here also support epoch-boundary optimizer/RNG resumption.
This is an execution capability, not authorization to extend a frozen score gate.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

import selection_objective as so

SETTING_KEYS = ('weight', 'lr', 'seed', 'positives', 'negatives', 'threads')


def source_hashes():
    return {p.name: so.mh.digest(p) for p in (Path(__file__), Path(so.__file__),
        Path(so.ms.__file__), Path(so.mh.__file__), Path(__file__).with_name('gbdt_blend.py'))}


def load_parent(parent_report, checkpoint, data, optimizer_mode, device=None):
    if optimizer_mode not in ('restart', 'resume'):
        raise ValueError('optimizer mode must be explicit')
    parent_path, checkpoint_path, data_path = map(Path, (parent_report, checkpoint, data))
    parent = json.loads(parent_path.read_text())
    inputs = {str(p.resolve()): so.mh.digest(p) for p in (parent_path, checkpoint_path, data_path)}
    if parent['data_sha256'] != inputs[str(data_path.resolve())]:
        raise ValueError('continuation dataset differs from parent')
    if parent['torch_version'] != torch.__version__ or parent['numpy_version'] != np.__version__:
        raise ValueError('parent library versions differ')
    sources = source_hashes()
    for name in ('selection_objective.py', 'masked_sequence.py', 'masked_history.py', 'gbdt_blend.py'):
        if parent['sources'].get(name) != sources[name]:
            raise ValueError('parent training source differs: '+name)
    if not parent['records']:
        raise ValueError('parent has no completed checkpoint')
    last = parent['records'][-1]
    if last['checkpoint_sha256'] != inputs[str(checkpoint_path.resolve())]:
        raise ValueError('checkpoint is not the parent last recorded epoch')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved['epoch'] != last['epoch'] or saved['config'] != parent['config']:
        raise ValueError('checkpoint configuration/epoch differs from parent')
    if not {'choice_output', 'choice_bias'} <= saved['state'].keys():
        raise ValueError('continuation requires a learned choice head')
    settings = dict(parent.get('training_settings') or {k: parent['arguments'][k] for k in SETTING_KEYS})
    c = so.ms.Config(**saved['config'])
    requested_device = device or c.device
    if optimizer_mode == 'resume':
        if 'resume_state' not in saved:
            raise ValueError('parent lacks optimizer/RNG state; explicit restart is required')
        if requested_device != c.device or parent['sources'].get(Path(__file__).name) != sources[Path(__file__).name]:
            raise ValueError('exact resume requires the same device and continuation source')
    c = dataclasses.replace(c, device=requested_device)
    if c.lr != settings['lr']:
        raise ValueError('parent learning rate differs from checkpoint')
    torch.set_num_threads(settings['threads'])
    torch.manual_seed(settings['seed'])
    model = so.SelectionModel(saved['items'], c, saved['weeks'])
    model.load_state_dict(saved['state'], strict=True)
    model = model.to(c.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=c.weight_decay)
    if optimizer_mode == 'resume':
        state = saved['resume_state']
        if state['training_settings'] != settings:
            raise ValueError('resume settings differ')
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng_cpu'])
        if c.device == 'mps':
            if state['rng_mps'] is None:
                raise ValueError('missing MPS RNG state')
            torch.mps.set_rng_state(state['rng_mps'])
    return model, optimizer, settings, saved, parent, inputs, sources


def run(a):
    if a.epochs < 1:
        raise ValueError('positive fixed continuation budget required')
    output = Path(a.output)
    if list(output.parent.glob(output.name+'.*')):
        raise FileExistsError('choose a new continuation output prefix')
    model, optimizer, settings, saved, parent, inputs, sources = load_parent(
        a.parent_report, a.checkpoint, a.data, a.optimizer_mode, a.device)
    d = so.load(a.data)
    if d['items'] != saved['items'] or so.ms.weeks(d) != saved['weeks']:
        raise ValueError('checkpoint item/date dimensions differ from data')
    labels = d['probe']['rating']
    known_labels = bool(np.all((labels >= 1) & (labels <= 5)))
    if not known_labels and not np.all(labels == 0):
        raise ValueError('query ratings must be all known or all zero sentinels')
    start_epoch = saved['epoch']
    config = dataclasses.asdict(model.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(state='running', arguments=vars(a), config=config, training_settings=settings,
        data_sha256=inputs[str(Path(a.data).resolve())], sources=sources,
        torch_version=torch.__version__, numpy_version=np.__version__,
        parent_inputs=inputs, parent_epoch=start_epoch, fixed_final_epoch=start_epoch+a.epochs,
        optimizer_policy=a.optimizer_mode,
        sampler_policy='preserve seed and continue global fine-tuning epoch numbers',
        metric='Fixed-budget continuation; development RMSE only when query labels exist. No new selection or promotion gate.',
        records=[])
    so.mh.atomic_json(str(output)+'.json', report)
    for epoch in range(start_epoch+1, start_epoch+a.epochs+1):
        rec = so.train_epoch(model, d, optimizer, epoch, settings['weight'], settings['seed'],
                             settings['positives'], settings['negatives'])
        pred = so.ms.predict(model, d)
        pred_path = str(output)+f'.e{epoch}.f32'
        so.mh.write_f32(pred_path, pred)
        rec['prediction_sha256'] = so.mh.digest(pred_path)
        rec['validation_rmse'] = float(np.sqrt(np.mean((np.clip(pred.astype(float), 1, 5)-labels)**2))) if known_labels else None
        checkpoint_path = str(output)+f'.e{epoch}.pt'
        # Capture after prediction, at the same boundary used by uninterrupted
        # training. Reconstructed NumPy batch RNGs depend only on seed + epoch.
        resume_state = dict(optimizer=optimizer.state_dict(), rng_cpu=torch.get_rng_state(),
            rng_mps=torch.mps.get_rng_state() if model.config.device == 'mps' else None,
            training_settings=settings)
        torch.save(dict(config=config, items=d['items'], weeks=saved['weeks'], epoch=epoch,
                        warm_epoch=saved['warm_epoch'], state=model.state_dict(), resume_state=resume_state), checkpoint_path)
        rec['checkpoint_sha256'] = so.mh.digest(checkpoint_path)
        report['records'].append(rec)
        so.mh.atomic_json(str(output)+'.json', report)
        print(json.dumps(rec), flush=True)
    for path, expected in inputs.items():
        if so.mh.digest(path) != expected:
            raise ValueError('parent input changed during continuation: '+path)
    if source_hashes() != sources:
        raise ValueError('continuation source changed during run')
    report['state'] = 'complete'
    so.mh.atomic_json(str(output)+'.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('parent-report', 'checkpoint', 'data', 'output'):
        p.add_argument('--'+flag, required=True)
    p.add_argument('--epochs', type=int, required=True, help='additional complete epochs; freeze before running')
    p.add_argument('--optimizer-mode', choices=('restart', 'resume'), required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), help='defaults to parent; changing device requires restart')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
