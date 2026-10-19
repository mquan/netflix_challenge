"""Production metadata must preserve schema, training-only statistics, and query order."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import production_metadata as meta
import test_production_dataset as fixtures
from production_dataset import create


class ProductionMetadataTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProductionDatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        create(self.fixture.source, self.fixture.training, self.fixture.qualifying, self.root / 'production')
        self.data = meta.load(self.fixture.source)
        self.query = meta.load(self.root / 'production/train-probe-qualifying.bin')
        self.tags = self.root / 'tags.tsv'
        self.tags.write_text('movie_id\tkind\tgenres\taudience\tfranchise\tera_popularity\ttone\tconfidence\n'
            '1\tfilm\taction|comedy\tteen\tPair\t2\tlight\thigh\n'
            '2\tseries\tdrama\tadult\t\t1\tdark\tlow\n'
            '3\tfilm\tdrama|romance\tfamily\tPair\t3\tneutral\tmedium\n')

    def arguments(self):
        cache = self.root / 'artifacts/r4/meta-context.npz'
        cache.parent.mkdir(parents=True)
        values = {name: values for name, values, _ in meta.blocks(self.data)}
        np.savez(cache, meta=values['meta'], context=values['context'])
        table = meta.movie_table(self.tags, 3)
        columns = {}
        for name in sorted(table):
            path = self.root / ('artifacts/r8/meta-'+name+'.f32')
            path.parent.mkdir(parents=True, exist_ok=True)
            table[name][self.data['probe']['item']-1].tofile(path)
            columns[str(path.relative_to(self.root))] = meta.digest(path)
        blender = self.root / 'blend.json'
        blender.write_text(json.dumps(dict(config=dict(extra_support=list(columns)), inputs=columns)))
        script_root = Path(meta.__file__).parent
        frozen = {str(self.root / 'data/processed/netflix.bin'): meta.digest(self.fixture.source),
                  str(cache): meta.digest(cache), str(blender): meta.digest(blender)}
        for name in ('gbdt_blend.py', 'moonshot_nn_blend.py', 'round4_boost.py'):
            frozen[str(script_root / name)] = meta.digest(script_root / name)
        plan = self.root / 'plan.json'
        plan.write_text(json.dumps(dict(frozen=frozen, base_report=str(blender))))
        return dict(data=self.fixture.source, reference_root=self.root, reference_plan=plan,
                    production_manifest=self.root / 'production/manifest.json', tags=self.tags,
                    output=self.root / 'export')

    def test_export_uses_merged_training_statistics_in_qualifying_order(self):
        args = self.arguments()
        report = meta.export(**args)
        self.assertEqual(report['state'], 'complete')
        self.assertEqual(len(report['stages']['qualifying']), 19)
        columns = report['stages']['qualifying']['meta']['columns']
        values = np.load(report['stages']['qualifying']['meta']['path'])
        # Qualifying user order is 2,0,1; all now have two training ratings.
        np.testing.assert_array_equal(values[:, columns.index('log_user_n')], np.full(3, np.log(3), dtype='<f4'))
        np.testing.assert_array_equal(values[:, columns.index('user_mean')], np.array([25/7, 27/7, 22/7], dtype='<f4'))
        kinds = np.fromfile(report['stages']['qualifying']['movie_kind']['path'], '<f4')
        np.testing.assert_array_equal(kinds, [0, 0, 1])  # Movies 1,3,2, not user order.
        franchise = np.fromfile(report['stages']['qualifying']['movie_franchise_size']['path'], '<f4')
        np.testing.assert_array_equal(franchise, [2, 2, 0])
        for outputs in report['stages'].values():
            for value in outputs.values():
                self.assertEqual(meta.digest(value['path']), value['sha256'])
        with self.assertRaises(FileExistsError):
            meta.export(**args)

    def test_query_labels_do_not_change_features_but_training_labels_do(self):
        expected = {name: values for name, values, _ in meta.blocks(self.query)}
        queries = self.query['probe'].copy()[::-1]
        queries['rating'] = 5
        changed = dict(self.query, probe=queries)
        for name, values, _ in meta.blocks(changed):
            np.testing.assert_array_equal(values, expected[name][::-1])
        changed = dict(self.query, train_ratings=np.ones_like(self.query['train_ratings']))
        actual = {name: values for name, values, _ in meta.blocks(changed)}
        self.assertFalse(np.array_equal(actual['meta'], expected['meta']))
        self.assertFalse(np.array_equal(actual['extra_meta'], expected['extra_meta']))
        np.testing.assert_array_equal(actual['context'], expected['context'])

    def test_reject_changed_cache_and_misordered_movie_ids(self):
        args = self.arguments()
        cache = self.root / 'artifacts/r4/meta-context.npz'
        with cache.open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'input digest mismatch'):
            meta.export(**args)
        rows = self.tags.read_text().splitlines()
        self.tags.write_text('\n'.join([rows[0], rows[2], rows[1], rows[3]])+'\n')
        with self.assertRaisesRegex(ValueError, 'ascending dense movie IDs'):
            meta.movie_table(self.tags, 3)


if __name__ == '__main__':
    unittest.main()
