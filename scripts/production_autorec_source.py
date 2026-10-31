"""Create an isolated checkpoint-capable AutoRec source, preserving its learning code."""
import argparse
import difflib
import json
from pathlib import Path
import runpy
import sys

from production_rbm_audit import digest, write_new

ROOT = Path(__file__).resolve().parents[1]

CAPTURE = '''
def save_production_checkpoint(args, path, epoch, net, items):
    import hashlib
    path = pathlib.Path(path)
    payload = dict(epoch=epoch, config=vars(args).copy(), items=items,
                   state={key: value.detach().cpu() for key, value in net.state_dict().items()},
                   torch_threads=torch.get_num_threads(), torch_version=str(torch.__version__),
                   numpy_version=np.__version__,
                   source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)

'''

RELOAD = '''    if args.reload_checkpoint:
        if not args.fixed_epoch:
            raise ValueError('production reload requires fixed-epoch mode')
        saved = torch.load(args.reload_checkpoint, map_location='cpu', weights_only=True)
        expected = vars(args).copy()
        original = saved['config'].copy()
        for field in ('reload_checkpoint', 'predictions'):
            expected.pop(field)
            original.pop(field)
        if original != expected or saved['items'] != items:
            raise ValueError('AutoRec reload configuration differs')
        net.load_state_dict(saved['state'], strict=True)
        pred, _ = evaluate()
        pred.astype('<f4').tofile(args.predictions)
        print(f"reload_epoch={saved['epoch']} prediction_count={len(pred)}", flush=True)
        return

'''


def instrument(source):
    replacements = [
        ('def main():\n', CAPTURE + '\ndef main():\n'),
        ('    args = ap.parse_args()\n',
         "    ap.add_argument('--reload-checkpoint')\n    args = ap.parse_args()\n"),
        ('            pred.astype(\'<f4\').tofile(snapshot)\n',
         '            pred.astype(\'<f4\').tofile(snapshot)\n'
         "            save_production_checkpoint(args, snapshot.with_suffix('.pt'), epoch, net, items)\n"),
        ('            pred.astype(\'<f4\').tofile(args.predictions)\n',
         '            pred.astype(\'<f4\').tofile(args.predictions)\n'
         "            save_production_checkpoint(args, pathlib.Path(args.predictions).with_suffix('.pt'), epoch, net, items)\n"),
        ("    print(f'autorec hidden={args.hidden}", RELOAD + "    print(f'autorec hidden={args.hidden}"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Unexpected frozen AutoRec source structure')
        source = source.replace(before, after)
    return source


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = ROOT / 'scripts/autorec.py'
    loader = ROOT / 'scripts/gbdt_blend.py'
    registry_path = ROOT / 'experiments/production-autorec-recipes.json'
    registry = json.loads(registry_path.read_text())
    if digest(source) != registry['source_sha256']:
        raise ValueError('Frozen AutoRec source changed')
    original = source.read_text()
    adapted = instrument(original)
    compile(adapted, str(output / 'autorec.py'), 'exec')
    output.mkdir(parents=True)
    (output / 'original-autorec.py').write_bytes(source.read_bytes())
    (output / 'gbdt_blend.py').write_bytes(loader.read_bytes())
    (output / 'autorec.py').write_text(adapted)
    patch = output / 'checkpoint-and-reload.patch'
    patch.write_text(''.join(difflib.unified_diff(original.splitlines(keepends=True), adapted.splitlines(keepends=True),
                                               fromfile='original-autorec.py', tofile='autorec.py')))
    report = dict(state='isolated_autorec_checkpoint_source_prepared', registry_sha256=digest(registry_path),
                  generator_sha256=digest(Path(__file__)), original_source_sha256=digest(source),
                  source_sha256=digest(output / 'autorec.py'), loader_sha256=digest(loader), patch_sha256=digest(patch),
                  scope='Checkpoint capture and reload are added around the unchanged learning loop. Device and CPU host thread count must be explicit in the production plan. Historical source provenance remains as limited in the original AutoRec registry.')
    write_new(output / 'source-manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='mode', required=True)
    prep = subparsers.add_parser('prepare')
    prep.add_argument('--output-dir', required=True)
    run = subparsers.add_parser('run')
    run.add_argument('--source', required=True)
    run.add_argument('--threads', type=int, default=1)
    run.add_argument('source_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(json.dumps(prepare(args.output_dir)))
        return
    if args.threads < 1:
        parser.error('threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    arguments = args.source_args[1:] if args.source_args[:1] == ['--'] else args.source_args
    sys.argv = [args.source, *arguments]
    runpy.run_path(args.source, run_name='__main__')


if __name__ == '__main__':
    main()
