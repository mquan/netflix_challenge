#!/usr/bin/env python3
"""Recover fixed-epoch neural recipes and exact recorded source bytes from git."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]


def selected_epoch(column, record):
    match = re.search(r'\.refit\.e(\d+)\.f32$', Path(column).name)
    if match:
        return int(match.group(1))
    epoch = record.get('selected_epoch')
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError('no recorded fixed epoch')
    return epoch


def main():
    inventory = json.loads((ROOT / 'experiments/production-bank-inventory.json').read_text())
    recipes, wanted = [], {}
    for column in inventory['columns']:
        path = Path(column['path'])
        if '/artifacts/masked/' not in str(path) or '/reference/' in str(path):
            continue
        name = path.name.split('.refit.e')[0] if '.refit.e' in path.name else path.stem
        record_path = ROOT / 'artifacts/masked' / (name + '.json')
        record = json.loads(record_path.read_text())
        assert digest(path) == column['sha256']
        epoch = selected_epoch(path, record)
        history = record.get('refit', [])
        progress_path = ROOT / 'artifacts/masked' / (name + '.refit.progress.json')
        if not history and progress_path.exists():
            history = json.loads(progress_path.read_text())['records']
        assert epoch in {entry['epoch'] for entry in history}
        args = dict(record['arguments'])
        assert not args.get('selection_data') and not args.get('selection_only')
        assert args.get('max_batches', 0) == 0
        assert digest(ROOT / args['data']) == record['data_sha256']
        args.update(epochs=epoch, data='{DATASET}', output='{OUTPUT_PREFIX}')
        if 'save_epochs' in args:
            args['save_epochs'] = False
        primary = 'masked_sequence.py' if 'seq_len' in record['config'] else 'masked_history.py'
        sources = {primary: record['source_sha256'], 'gbdt_blend.py': record['loader_sha256']}
        if record.get('base_source_sha256'):
            sources['masked_history.py'] = record['base_source_sha256']
        for filename, expected in sources.items():
            wanted.setdefault(filename, set()).add(expected)
        command = ['{PYTHON}', '{SOURCE_DIRECTORY}/' + primary]
        for key, value in args.items():
            if value is None or value is False:
                continue
            command.append('--' + key.replace('_', '-'))
            if value is not True:
                command.append(str(value))
        recipes.append(dict(column=str(path), column_sha256=column['sha256'], fixed_epochs=epoch,
            training_report=str(record_path), training_report_sha256=digest(record_path),
            epoch_history_sha256=digest(progress_path) if not record.get('refit') else digest(record_path),
            historical_selected_column_digest_present=record.get('predictions_sha256') == column['sha256'],
            original_data_sha256=record['data_sha256'], torch_version=record['torch_version'],
            numpy_version=record['numpy_version'], sources=sources, command_template=command))
    recovered, contents = {}, {}
    for filename, hashes in wanted.items():
        path = 'scripts/' + filename
        commits = subprocess.check_output(['git', 'log', '--all', '--format=%H', '--', path], cwd=ROOT, text=True).splitlines()
        for commit in dict.fromkeys(commits):
            result = subprocess.run(['git', 'show', commit + ':' + path], cwd=ROOT, capture_output=True)
            if result.returncode:
                continue
            sha = hashlib.sha256(result.stdout).hexdigest()
            if sha in hashes and sha not in recovered:
                recovered[sha] = dict(commit=commit, path=path, archive_member=sha + '/' + filename)
                contents[sha] = result.stdout
    missing = set().union(*wanted.values()) - set(recovered)
    if missing:
        raise ValueError(f'unrecovered recorded sources: {sorted(missing)}')
    output = ROOT / 'experiments/production-neural-recipes.json'
    archive = ROOT / 'experiments/production-neural-sources.zip'
    if output.exists() or archive.exists():
        raise FileExistsError('recovery evidence already exists')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for sha, info in recovered.items():
            assert hashlib.sha256(contents[sha]).hexdigest() == sha
            stream.writestr(info['archive_member'], contents[sha])
    report = dict(recipes=recipes, sources=recovered, archive_sha256=digest(archive),
        limitation='Source bytes and fixed-epoch recipes recovered. No retraining, Stage C dataset, qualifying-order export, or final blender deployment has been validated. '
                   'Templates require an isolated source directory containing each recipe source set. Unlabeled-query RMSE must not be treated as a score.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(neural_recipes=len(recipes), exact_source_versions=len(recovered),
                         recipes_with_historical_selected_digest=sum(r['historical_selected_column_digest_present'] for r in recipes))), flush=True)


if __name__ == '__main__':
    main()
