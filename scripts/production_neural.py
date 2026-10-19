#!/usr/bin/env python3
"""Run a retained neural recipe at its fixed epoch using its exact archived source.

Default: verify inputs and write a preparation report. --train explicitly runs
the fixed budget. Unknown query labels are never scored or used for selection.
Each invocation uses one isolated source set and a new output directory.
"""
import argparse
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def find_recipe(registry, column):
    matches = [r for r in registry['recipes'] if Path(r['column']).name == column]
    if len(matches) != 1:
        raise ValueError('column must identify exactly one retained recipe')
    recipe = matches[0]
    if not isinstance(recipe['fixed_epochs'], int) or recipe['fixed_epochs'] <= 0:
        raise ValueError('recipe has no positive fixed epoch')
    return recipe


def extract_sources(recipe, registry, archive, output):
    if digest(archive) != registry['archive_sha256']:
        raise ValueError('source archive hash mismatch')
    output.mkdir()
    with zipfile.ZipFile(archive) as z:
        for name, sha in recipe['sources'].items():
            if name not in ('masked_history.py', 'masked_sequence.py', 'gbdt_blend.py'):
                raise ValueError('unexpected source filename')
            content = z.read(registry['sources'][sha]['archive_member'])
            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError('source member hash mismatch')
            (output / name).write_bytes(content)


def load_trainer(recipe, source_dir):
    if any(name in sys.modules for name in ('gbdt_blend', 'masked_history', 'masked_sequence')):
        raise RuntimeError('run each archived source set in a fresh Python process')
    primary = Path(recipe['command_template'][1]).name
    if primary not in recipe['sources'] or primary not in ('masked_history.py', 'masked_sequence.py'):
        raise ValueError('invalid primary trainer')
    sys.path.insert(0, str(source_dir.resolve()))
    trainer = importlib.import_module(primary[:-3])
    for name, sha in recipe['sources'].items():
        module = sys.modules[name[:-3]]
        if Path(module.__file__).resolve() != (source_dir / name).resolve() or digest(module.__file__) != sha:
            raise ValueError('import did not use the exact archived source')
    return trainer


def configuration(trainer, recipe, data, prefix):
    replacements = {'{DATASET}': str(data), '{OUTPUT_PREFIX}': str(prefix)}
    command = [replacements.get(arg, arg) for arg in recipe['command_template'][2:]]
    args = trainer.parser().parse_args(command)
    if args.epochs != recipe['fixed_epochs'] or args.threads < 1:
        raise ValueError('recipe budget mismatch')
    if args.selection_data or args.selection_only or args.max_batches or getattr(args, 'save_epochs', False):
        raise ValueError('production recipe must use complete fixed epochs without selection')
    c = trainer.Config(**{field.name: getattr(args, field.name) for field in dataclasses.fields(trainer.Config)})
    return c, args


def train_fixed(trainer, d, c, epochs, prefix):
    """Call the archived refit path directly, bypassing its development scorer."""
    import numpy as np
    if epochs < 1:
        raise ValueError('positive epoch budget required')
    model, records, completed = trainer.fit(d, c, epochs, str(prefix), select=False, max_batches=0)
    if completed != epochs or len(records) != epochs or any('validation_rmse' in r for r in records):
        raise ValueError('trainer did not execute the complete fixed budget')
    pred = trainer.predict(model, d)
    if pred.shape != (len(d['probe']),) or not np.isfinite(pred).all():
        raise ValueError('invalid qualifying predictions')
    # Retained columns enter the blender as raw predictions. Clip only the
    # final ensemble, matching development inference.
    path = Path(str(prefix) + '.f32')
    temporary = Path(str(path) + '.tmp')
    with temporary.open('xb') as stream:
        pred.astype('<f4').tofile(stream)
    temporary.replace(path)
    return model, pred, dict(fixed_epochs=epochs, records=records,
        prediction_count=len(pred), predictions_sha256=digest(path),
        checkpoint_sha256=digest(str(prefix) + '.pt'),
        prediction_min=float(pred.min()), prediction_max=float(pred.max()))


def validate_dataset(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('state') != 'complete' or manifest.get('purpose') != 'production_only_unlabeled_queries':
        raise ValueError('verified production dataset manifest required')
    root = manifest_path.parent
    for name, sha in manifest['outputs'].items():
        if Path(name).name != name or digest(root / name) != sha:
            raise ValueError('production dataset output hash mismatch')
    name = 'train-probe-qualifying.bin'
    if name not in manifest['outputs']:
        raise ValueError('production dataset missing')
    return root / name, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--column', required=True, help='retained column filename from the recipe registry')
    parser.add_argument('--dataset-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--registry', default=str(ROOT / 'experiments/production-neural-recipes.json'))
    parser.add_argument('--archive', default=str(ROOT / 'experiments/production-neural-sources.zip'))
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    registry = json.loads(Path(args.registry).read_text())
    recipe = find_recipe(registry, args.column)
    if digest(recipe['column']) != recipe['column_sha256']:
        raise ValueError('retained reference column hash mismatch')
    data, manifest = validate_dataset(args.dataset_manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / 'source'
    extract_sources(recipe, registry, args.archive, source_dir)
    trainer = load_trainer(recipe, source_dir)
    c, parsed = configuration(trainer, recipe, data, output / 'model')
    import numpy as np
    d = trainer.load(data)
    if len(d['probe']) != manifest['qualifying_count'] or not np.all(d['probe']['rating'] == 0):
        raise ValueError('qualifying queries must have unknown-label zero sentinels')
    if len(d['train_ratings']) != manifest['train_count'] or not np.all((d['train_ratings'] >= 1) & (d['train_ratings'] <= 5)):
        raise ValueError('invalid production training ratings')
    trainer.torch.set_num_threads(parsed.threads)
    report = dict(state='prepared', column=recipe['column'], column_sha256=recipe['column_sha256'],
        registry_sha256=digest(args.registry), archive_sha256=digest(args.archive),
        adapter_sha256=digest(__file__), dataset_manifest_sha256=digest(args.dataset_manifest),
        data_sha256=manifest['outputs'][data.name], fixed_epochs=recipe['fixed_epochs'],
        sources=recipe['sources'], config=dataclasses.asdict(c), threads=parsed.threads,
        torch_version=str(trainer.torch.__version__), numpy_version=np.__version__,
        training_requested=args.train, metric=None,
        limitation='No qualifying labels exist. This adapter does not compute query RMSE or select epochs. Prepared status does not establish trained predictions.')
    report_path = output / 'report.json'
    trainer.atomic_json(report_path, report)
    if args.train:
        try:
            report['state'] = 'training'
            trainer.atomic_json(report_path, report)
            _, _, result = train_fixed(trainer, d, c, recipe['fixed_epochs'], output / 'model')
            if digest(data) != report['data_sha256']:
                raise ValueError('production data changed during training')
            report.update(result, state='complete', elapsed_seconds=time.monotonic() - start)
        except BaseException as error:
            report.update(state='failed', error=repr(error), elapsed_seconds=time.monotonic() - start)
            trainer.atomic_json(report_path, report)
            raise
        trainer.atomic_json(report_path, report)
    print(json.dumps(dict(state=report['state'], column=args.column, fixed_epochs=recipe['fixed_epochs']), indent=2), flush=True)


if __name__ == '__main__':
    main()
