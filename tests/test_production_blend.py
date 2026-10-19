"""Retained blend arithmetic, persisted inference, and missing-column safeguards."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import production_blend as pb
from gbdt_blend import HEADER, PROBE
from test_joint import dataset


def settings():
    return dict(tree_config=dict(learning_rate=.05, leaves=7, min_leaf=5),
        neural_config=dict(hidden=8, layers=1, dropout=.1, lr=.001, weight_decay=.01,
                           batch=32, threads=1, patience=2),
        budgets=dict(tree=[dict(seed=7, steps=4)], neural=[dict(seed=13, steps=2)]),
        lightgbm_version=pb.importlib.metadata.version('lightgbm'), torch_version=pb.torch.__version__)


def fixture(root):
    dataset(root / 'dev.bin')
    d = pb.load(root / 'dev.bin')
    rows = len(d['probe'])
    rng = np.random.default_rng(108)
    p = np.column_stack([d['probe']['rating']+rng.normal(0, .6, rows), rng.uniform(1, 5, rows)]).astype('<f4')
    auxiliary = rng.normal(size=(rows, 3)).astype('<f4')
    auxiliary[:, -1] = 1  # Constant-feature standardization floor.
    context = np.column_stack([np.ones(rows), rng.normal(0, .1, (rows, 4))]).astype('<f4')
    groups, schema = {}, {}
    for key, value in [('predictions', p), ('auxiliary', auxiliary), ('context', context)]:
        path = root / (key+'.npy')
        np.save(path, value, allow_pickle=False)
        groups[key] = [dict(name=key+str(i), path=str(path), sha256=pb.digest(path), column=i) for i in range(value.shape[1])]
        schema[key] = [v['name'] for v in groups[key]]
    # Preserve a duplicate prediction as its own regularized input.
    groups['predictions'].append(dict(groups['predictions'][0], name='prediction_alias'))
    schema['predictions'].append('prediction_alias')
    raw = bytearray((root / 'dev.bin').read_bytes())
    offset = HEADER.itemsize + 8*(d['users']+1) + 5*len(d['train_ratings'])
    queries = np.frombuffer(raw, dtype=PROBE, count=rows, offset=offset)
    queries['rating'] = 0
    (root / 'query.bin').write_bytes(raw)
    dev = dict(rows=rows, dataset=dict(path=str(root / 'dev.bin'), sha256=pb.digest(root / 'dev.bin')), **groups)
    query = dict(rows=rows, dataset=dict(path=str(root / 'query.bin'), sha256=pb.digest(root / 'query.bin')), **groups)
    bank = dict(development=dev, qualifying=query, schema=schema, settings=settings())
    return bank, offset


class ProductionBlendTests(unittest.TestCase):
    def test_budget_rule_and_normalization_match_retained_code(self):
        tree = dict(inputs={'a': 'hash'}, config=dict(seeds=[7, 13]),
                    folds=[dict(fold=i, chosen_rounds=[x, 2*x]) for i, x in enumerate([4, 8, 3, 9, 5])])
        neural = dict(inputs=tree['inputs'], config=tree['config'],
                      folds=[dict(fold=i, chosen_epochs=[x, x+1]) for i, x in enumerate([1, 4, 2, 5, 3])])
        self.assertEqual(pb.budgets(tree, neural), dict(tree=[dict(seed=7, steps=5), dict(seed=13, steps=10)],
            neural=[dict(seed=7, steps=3), dict(seed=13, steps=4)]))
        neural['folds'][-1]['fold'] = 0
        with self.assertRaises(ValueError):
            pb.budgets(tree, neural)
        rng = np.random.default_rng(42)
        x = rng.normal(size=(200, 4)); x[:, -1] = 5
        query = np.full((3, 4), 1e9)
        fit = pb.fit_normalizer(x)
        combined = np.concatenate([x, query])
        expected = pb.nn.standardize(combined, np.arange(len(combined)) < len(x))
        np.testing.assert_array_equal(pb.transform(combined, fit), expected)
        np.testing.assert_array_equal(fit['std'][-1], 1e-6)

    def test_roundtrip_matches_original_components_and_rejects_schema_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, _ = fixture(root)
            matrices, schema, d, _ = pb.read_bank(bank['development'])
            np.testing.assert_array_equal(matrices['predictions'][:, 0], matrices['predictions'][:, -1])
            model_dir = root / 'model'
            report = pb.train(matrices, d['probe']['rating'], schema, bank['settings'], model_dir)
            result = pb.predict(matrices, schema, model_dir)
            p, aux, context = (matrices[k] for k in ('predictions', 'auxiliary', 'context'))
            rx = pb.design(p, context)[0]
            base = 3.6 + rx @ pb.ridge_fit(rx, d['probe']['rating'])
            mean = p.mean(1, keepdims=True)
            features = np.column_stack([mean-3.6, p-mean, aux])
            x = pb.nn.standardize(features, np.ones(len(p), dtype=bool))
            model, _ = pb.nn.train(x, d['probe']['rating']-base,
                pb.SimpleNamespace(**bank['settings']['neural_config']), 13, 2)
            expected_neural = base+pb.nn.predict(model, x)
            # Independent LightGBM invocation in a fresh process; parent uses Torch.
            np.save(root / 'reference-x.npy', np.column_stack([p, aux]).astype(np.float32))
            source = ('import sys,numpy as np,lightgbm as lgb; '
                      'm=lgb.Booster(model_file=sys.argv[1]); '
                      'np.save(sys.argv[3],m.predict(np.load(sys.argv[2]),num_threads=1))')
            subprocess.run([sys.executable, '-c', source, str(model_dir / report['trees'][0]['file']),
                            str(root / 'reference-x.npy'), str(root / 'reference-tree.npy')], check=True)
            expected_tree = (base+np.load(root / 'reference-tree.npy')).astype(np.float32)
            np.testing.assert_array_equal(result['tree'], expected_tree)
            np.testing.assert_array_equal(result['neural'], expected_neural.astype('<f4'))
            np.testing.assert_array_equal(result['average'], ((expected_tree.astype(float)+expected_neural)/2).astype('<f4'))
            np.testing.assert_array_equal(result['predictions'], np.clip(result['average'], 1, 5))
            self.assertTrue(np.all((result['predictions'] >= 1) & (result['predictions'] <= 5)))
            wrong = dict(schema, predictions=schema['predictions'][::-1])
            with self.assertRaisesRegex(ValueError, 'column order'):
                pb.predict(matrices, wrong, model_dir)
            with (model_dir / report['networks'][0]['file']).open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                pb.predict(matrices, schema, model_dir)
            with self.assertRaisesRegex(ValueError, 'unknown sentinels'):
                pb.train(matrices, np.zeros(len(p)), schema, bank['settings'], root / 'invalid')

    def test_cli_fit_infer_query_label_isolation_and_pending_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank, offset = fixture(root)
            manifest = root / 'bank.json'
            manifest.write_text(json.dumps(bank))
            cmd = [sys.executable, str(Path(pb.__file__).resolve())]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1')
            def run(*args):
                result = subprocess.run(cmd+list(map(str, args)), capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            run('fit', '--bank', manifest, '--output', root / 'model')
            run('infer', '--bank', manifest, '--model', root / 'model', '--output', root / 'out0')
            raw = bytearray((root / 'query.bin').read_bytes())
            np.frombuffer(raw, dtype=PROBE, count=200, offset=offset)['rating'] = 5
            (root / 'query.bin').write_bytes(raw)
            bank['qualifying']['dataset']['sha256'] = pb.digest(root / 'query.bin')
            manifest.write_text(json.dumps(bank))
            run('infer', '--bank', manifest, '--model', root / 'model', '--output', root / 'out5')
            for name in ('tree', 'neural', 'average', 'predictions'):
                self.assertEqual((root / 'out0' / (name+'.f32')).read_bytes(), (root / 'out5' / (name+'.f32')).read_bytes())
            bank['qualifying']['predictions'][0] = dict(name='not_ready', pending=True)
            with self.assertRaisesRegex(ValueError, 'production columns pending'):
                pb.read_bank(bank['qualifying'])


if __name__ == '__main__':
    unittest.main()
