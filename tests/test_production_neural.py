"""Exercise exact archived trainers in separate processes, including label poisoning."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from production_neural import configuration, extract_sources, find_recipe, load_trainer, train_fixed


def worker(column, destination):
    import dataclasses
    import numpy as np
    from test_joint import dataset

    destination = Path(destination)
    registry = json.loads((ROOT / 'experiments/production-neural-recipes.json').read_text())
    recipe = find_recipe(registry, column)
    sources = destination / 'source'
    extract_sources(recipe, registry, ROOT / 'experiments/production-neural-sources.zip', sources)
    trainer = load_trainer(recipe, sources)
    trainer.torch.set_num_threads(1)
    path = destination / 'fixture.bin'
    dataset(path, correlated_train=True)
    d = trainer.load(path)
    d['probe'] = d['probe'].copy()
    d['probe']['rating'] = 0
    c, _ = configuration(trainer, recipe, path, destination / 'fixture')
    fields = {f.name for f in dataclasses.fields(c)}
    small = dict(rank=4, hidden=8, batch_users=32, nnz_budget=512, dropout=0., device='cpu',
                 d_model=8, heads=2, layers=1, ff=16, seq_len=16, max_targets=4,
                 batch_tokens=256, pad_width=8, pad_rows=8)
    c = dataclasses.replace(c, **{k: v for k, v in small.items() if k in fields})
    first, pred, report = train_fixed(trainer, d, c, 2, destination / 'zero')
    changed = dict(d)
    changed['probe'] = d['probe'].copy()
    changed['probe']['rating'] = 5
    second, other, altered_report = train_fixed(trainer, changed, c, 2, destination / 'poison')
    for name, value in first.state_dict().items():
        assert trainer.torch.equal(value, second.state_dict()[name]), name
    np.testing.assert_array_equal(pred, other)
    np.testing.assert_allclose(pred[::-1], trainer.predict(first, d, d['probe'][::-1].copy()), rtol=0, atol=1e-6)
    saved = trainer.torch.load(destination / 'zero.pt', map_location='cpu', weights_only=True)
    assert saved['epoch'] == 2
    for name, value in first.state_dict().items():
        assert trainer.torch.equal(value, saved['state'][name]), name
    assert report['fixed_epochs'] == 2 and len(report['records']) == 2
    assert all(r['targets'] > 0 for r in report['records'])
    assert report['predictions_sha256'] == altered_report['predictions_sha256']
    assert 'probe_rmse' not in report and all('validation_rmse' not in r for r in report['records'])
    invalid = dict(recipe, command_template=recipe['command_template'] + ['--selection-data', 'forbidden'])
    try:
        configuration(trainer, invalid, path, destination / 'bad')
    except ValueError:
        pass
    else:
        raise AssertionError('selection allowed in production configuration')
    result = dict(column=column, sources=recipe['sources'], fixture_epochs=2,
                  checks=['identical trained parameters and predictions with zero versus poisoned query labels',
                          'prediction order preserved', 'final epoch checkpoint matches trained state',
                          'selection configuration rejected', 'no query metric reported'], passed=True)
    (destination / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')


class ProductionNeuralTests(unittest.TestCase):
    def test_all_four_archived_trainer_versions(self):
        for column in ('MH3-categorical-mse-refit.f32', 'MT0-refit.refit.e12.f32',
                       'MT5-refit.f32', 'A2h-refit.refit.e12.f32'):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temp:
                result = subprocess.run([sys.executable, __file__, '--worker', column, temp],
                                        capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(json.loads((Path(temp) / 'verification.json').read_text())['passed'])

    def test_archive_tampering_is_rejected(self):
        registry = json.loads((ROOT / 'experiments/production-neural-recipes.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'bad.zip').write_bytes(b'not the archived source')
            with self.assertRaisesRegex(ValueError, 'archive hash'):
                extract_sources(registry['recipes'][0], registry, path / 'bad.zip', path / 'source')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
