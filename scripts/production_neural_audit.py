"""Verify one completed fixed-budget production neural column before queue handoff."""
import argparse
import json
from pathlib import Path

import numpy as np

from production_neural import digest
from selection_objective_queue import verify

FIELDS = ('column', 'column_sha256', 'registry_sha256', 'archive_sha256',
          'adapter_sha256', 'dataset_manifest_sha256', 'data_sha256',
          'fixed_epochs', 'sources', 'config', 'threads', 'torch_version', 'numpy_version')


def audit(plan_path, index):
    plan_path = Path(plan_path)
    plan_sha = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    verify(plan['frozen'])
    if not 0 <= index < len(plan['stages']):
        raise ValueError('invalid production stage index')
    stage = plan['stages'][index]
    root = Path(stage['output_dir'])
    destination = root/'audit.json'
    if destination.exists():
        raise FileExistsError(destination)
    report_path = root/'report.json'
    report = json.loads(report_path.read_text())
    expected = stage['expected']
    if set(expected) != set(FIELDS) or any(report.get(k) != expected[k] for k in FIELDS):
        raise ValueError('production recipe, data, sources, config or libraries differ')
    if report['state'] != 'complete' or report['training_requested'] is not True or report['metric'] is not None:
        raise ValueError('complete unlabeled fixed-budget training required')
    epochs = expected['fixed_epochs']
    records = report['records']
    if (report['fixed_epochs'] != epochs or len(records) != epochs or
            [r['epoch'] for r in records] != list(range(1, epochs+1)) or
            any(r['targets'] <= 0 or 'validation_rmse' in r for r in records)):
        raise ValueError('incomplete epoch budget or query scoring')
    bound = {str(report_path): digest(report_path)}
    for name, sha in expected['sources'].items():
        path = root/'source'/name
        if digest(path) != sha:
            raise ValueError('extracted source differs')
        bound[str(path)] = sha
    prediction, checkpoint = root/'model.f32', root/'model.pt'
    if (digest(prediction) != report['predictions_sha256'] or
            digest(checkpoint) != report['checkpoint_sha256']):
        raise ValueError('production output hash differs')
    values = np.fromfile(prediction, '<f4')
    if (values.shape != (plan['qualifying_count'],) or not np.isfinite(values).all() or
            report['prediction_count'] != len(values)):
        raise ValueError('invalid qualifying prediction coverage or values')
    if any(abs(float(fn(values))-report['prediction_'+name]) > 1e-6
           for name, fn in [('min', np.min), ('max', np.max)]):
        raise ValueError('prediction range differs')
    import torch
    torch.set_num_threads(1)
    if str(torch.__version__) != expected['torch_version'] or np.__version__ != expected['numpy_version']:
        raise ValueError('audit library differs')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['epoch'] != epochs or saved['config'] != expected['config'] or saved['items'] != plan['items']:
        raise ValueError('checkpoint metadata differs')
    if not saved['state'] or any(not torch.isfinite(t).all().item() for t in saved['state'].values()):
        raise ValueError('invalid checkpoint tensors')
    bound.update({str(prediction): report['predictions_sha256'],
                  str(checkpoint): report['checkpoint_sha256']})
    verify(plan['frozen']); verify(bound)
    if digest(plan_path) != plan_sha:
        raise ValueError('production plan changed')
    result = dict(state='verified_complete', stage=index, column=expected['column'],
        plan_sha256=plan_sha, fixed_epochs=epochs, qualifying_count=len(values), outputs=bound,
        metric=None, scope='Raw ordered qualifying column, not a qualifying score or bank promotion. '
                          'All retained columns, support features and the fitted final blender remain required.')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', required=True, type=int)
    args = parser.parse_args()
    print(audit(args.plan, args.stage)['state'])
