#!/usr/bin/env python3
"""Read-only SO1 gradient audit at the frozen warm checkpoint on real P2 batches.

CPU training-mode dropout, with unused padding rows removed for bounded memory.
The audited model is never updated. Optional optimizer comparisons use temporary
parameter copies; no validation scoring or recipe change occurs.
"""
import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import selection_objective as so


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()+b'\0')
        h.update(str(array.dtype).encode()+str(array.shape).encode())
        h.update(memoryview(array).cast('B'))
    return h.hexdigest()


def gradients(model):
    values = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('nonfinite gradient: '+name)
            values[name] = parameter.grad.detach().clone()
    return values


def geometry(rating, auxiliary, names):
    aa, bb, ab = 0., 0., 0.
    for name in names:
        a, b = rating.get(name), auxiliary.get(name)
        if a is not None:
            a = a.double()
            aa += float(a.square().sum())
        if b is not None:
            b = b.double()
            bb += float(b.square().sum())
        if a is not None and b is not None:
            ab += float((a*b).sum())
    na, nb = math.sqrt(aa), math.sqrt(bb)
    return dict(parameters=list(names), rating_norm=na, weighted_auxiliary_norm=nb,
        auxiliary_to_rating_ratio=nb/na if na else None,
        cosine=ab/(na*nb) if na and nb else None,
        combined_norm=math.sqrt(max(aa+bb+2*ab, 0)))


def first_adam_step(model, rating, auxiliary, common):
    """Exact default AdamW first steps on copies; original model stays unchanged."""
    parameters = dict(model.named_parameters())
    union = sorted(set(rating) | set(auxiliary))
    control_norm = geometry(rating, {}, union)['rating_norm']
    treatment_norm = geometry(rating, auxiliary, union)['combined_norm']
    updates = []
    for include_choice, norm in ((False, control_norm), (True, treatment_norm)):
        copies = [torch.nn.Parameter(parameters[name].detach().clone()) for name in common]
        optimizer = torch.optim.AdamW(copies, lr=model.config.lr, weight_decay=model.config.weight_decay)
        scale = min(1., 5/(norm+1e-6))
        for name, parameter in zip(common, copies):
            parameter.grad = rating[name].clone()
            if include_choice:
                parameter.grad.add_(auxiliary[name])
            parameter.grad.mul_(scale)
        optimizer.step()
        updates.append({name: parameter.detach()-parameters[name].detach() for name, parameter in zip(common, copies)})
        del optimizer, copies
    control, treatment = updates
    difference = {name: treatment[name]-control[name] for name in common}
    changes = geometry(control, difference, common)
    alignment = geometry(control, treatment, common)
    return dict(parameters=common, control_update_norm=changes['rating_norm'],
        treatment_update_norm=alignment['weighted_auxiliary_norm'],
        treatment_minus_control_norm=changes['weighted_auxiliary_norm'],
        relative_update_change=changes['auxiliary_to_rating_ratio'],
        control_treatment_cosine=alignment['cosine'],
        control_clipping_scale=min(1., 5/(control_norm+1e-6)),
        treatment_clipping_scale=min(1., 5/(treatment_norm+1e-6)),
        scope='Independent first step with zero optimizer moments for each sampled batch; copies of common active parameters only. Not a multi-step training trajectory.')


def audit(plan, output, batches=4, adam_first_step=False):
    if batches < 1:
        raise ValueError('positive batch count required')
    plan_path, output = Path(plan), Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = plan_path.read_bytes()
    p = json.loads(raw)
    for path, sha in p['frozen'].items():
        if so.mh.digest(path) != sha:
            raise ValueError('frozen SO1 input changed: '+path)
    if torch.__version__ != p['libraries']['torch'] or np.__version__ != p['libraries']['numpy']:
        raise ValueError('SO1 library versions changed')
    settings = p['settings']
    torch.set_num_threads(1)
    model, epoch = so.warm_model(p['warm_checkpoint'], device='cpu', seed=settings['seed'], lr=settings['lr'])
    model.config = dataclasses.replace(model.config, pad_rows=1)
    before = state_digest(model)
    c = model.config
    d = so.load(p['data'])
    counts = np.diff(d['train_offsets']).astype(np.int64)
    active = np.flatnonzero(counts > 0)
    repeats = np.minimum(np.maximum(-(-counts[active]//c.seq_len), 1), max(c.windows, 1))
    rng = np.random.default_rng(settings['seed']+1)
    auxiliary_rng = np.random.default_rng(settings['seed']+1+1000000)
    groups = so.ms.length_batches(so.ms.capped_lengths(d, c), rng.permutation(np.repeat(active, repeats)), c, rng)
    selected = groups[:batches]
    if len(selected) != batches:
        raise ValueError('not enough batches')
    weight = p['arms']['treatment']['weight']
    report = dict(state='running', source_sha256=so.mh.digest(__file__),
        plan=str(plan_path), plan_sha256=hashlib.sha256(raw).hexdigest(), warm_epoch=epoch,
        model_state_before=before, fixed_weight=weight, selected_batches=batches,
        adam_first_step_on_parameter_copies=adam_first_step,
        sampler='first consecutive groups from frozen epoch-1 sampler; no optimizer updates between groups',
        config=dataclasses.asdict(c), differences_from_queued_training=['CPU device', 'pad_rows=1', 'no parameter updates'],
        limitation='Initial-checkpoint gradient diagnostics only; CPU dropout/padding differ from queued MPS training. Not an ensemble result, score gate, or MPS throughput estimate.', records=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    model.train()
    for index, users in enumerate(selected, 1):
        start = time.monotonic()
        rating = so.ms.prepare_batch(d, users, c, rng=rng)
        auxiliary = so.auxiliary_batch(d, users, c, auxiliary_rng, settings['positives'], settings['negatives'])
        model.zero_grad(set_to_none=True)
        raw_rating = model.forward_batch(rating)
        rating_loss = so.mh.loss_value(model, raw_rating, rating['target_ratings'])
        if not torch.isfinite(rating_loss):
            raise FloatingPointError('nonfinite rating loss')
        rating_loss.backward()
        rating_grads = gradients(model)
        rating_value = float(rating_loss.detach())
        del raw_rating, rating_loss
        model.zero_grad(set_to_none=True)
        choice_loss = model.choice_loss(auxiliary)
        if not torch.isfinite(choice_loss):
            raise FloatingPointError('nonfinite choice loss')
        (weight*choice_loss).backward()
        auxiliary_grads = gradients(model)
        common = sorted(set(rating_grads) & set(auxiliary_grads))
        all_names = sorted(set(rating_grads) | set(auxiliary_grads))
        all_geometry = geometry(rating_grads, auxiliary_grads, all_names)
        record = dict(batch=index, users=users.tolist(), user_count=len(users),
            history_min=int(counts[users].min()), history_max=int(counts[users].max()),
            history_mean=float(counts[users].mean()), rating_targets=rating['target_ratings'].numel(),
            choice_targets=auxiliary['choice_labels'].numel(),
            rating_width=rating['seq_width'], auxiliary_width=auxiliary['seq_width'],
            rating_loss=rating_value, choice_loss=float(choice_loss.detach()),
            logged_shared_tensors=geometry(rating_grads, auxiliary_grads, ['code_out.weight', 'blocks.0.qkv.weight']),
            common_active_parameters=geometry(rating_grads, auxiliary_grads, common),
            all_parameters=all_geometry,
            clipping_scale_if_applied=min(1., 5/(all_geometry['combined_norm']+1e-6)),
            seconds=time.monotonic()-start)
        if adam_first_step:
            record['first_adam_step'] = first_adam_step(model, rating_grads, auxiliary_grads, common)
            record['seconds'] = time.monotonic()-start
        report['records'].append(record)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: record[k] for k in ('batch', 'user_count', 'rating_targets', 'rating_width', 'rating_loss', 'choice_loss', 'logged_shared_tensors', 'clipping_scale_if_applied', 'seconds')}), flush=True)
        if adam_first_step:
            print(json.dumps(dict(batch=index, first_adam_step=record['first_adam_step'])), flush=True)
        del rating, auxiliary, choice_loss, rating_grads, auxiliary_grads
    after = state_digest(model)
    if before != after:
        raise ValueError('audit unexpectedly changed model parameters')
    for path, sha in p['frozen'].items():
        if so.mh.digest(path) != sha:
            raise ValueError('frozen SO1 input changed during audit: '+path)
    if plan_path.read_bytes() != raw:
        raise ValueError('SO1 plan changed during audit')
    report.update(state='complete', model_state_after=after, unchanged_model_parameters=True,
                  frozen_inputs_rechecked=len(p['frozen']))
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--adam-first-step', action='store_true')
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps(dict(state=result['state'], unchanged_model_parameters=result['unchanged_model_parameters'])), flush=True)
