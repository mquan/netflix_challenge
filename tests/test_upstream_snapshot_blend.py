import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import upstream_snapshot_blend as b
import upstream_exchange as x


class SnapshotBlendTests(unittest.TestCase):
    def test_component_clip_is_explicit(self):
        z = b.design(np.array([[-2., -2.], [8., 8.]]), [True, False])
        np.testing.assert_array_equal(z, [[0., -2., 1.], [6., 8., 1.]])

    def test_control_matches_original_and_scoring_labels_are_isolated(self):
        rng = np.random.default_rng(17)
        matrix = rng.normal(3.5, 2., (100, 3))
        labels = rng.uniform(1, 5, 100)
        folds = np.tile(np.arange(5), 20)
        actual, _ = b.evaluate(list(matrix.T), [True]*3, labels, folds)
        np.testing.assert_allclose(actual, x.crossfit(matrix, labels, folds), atol=1e-12)
        context = rng.normal(size=(100, 2))
        before, _ = b.evaluate(list(matrix.T), [True]*3, labels, folds, context, (0, 1))
        changed = labels.copy(); changed[folds == 2] += 10
        after, _ = b.evaluate(list(matrix.T), [True]*3, changed, folds, context, (0, 1))
        np.testing.assert_allclose(before[folds == 2], after[folds == 2], atol=1e-12)

    def test_saved_outputs_require_receipts_and_exact_duplicates_are_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fresh = root/'fresh'
            (fresh/'preds_new').mkdir(parents=True)
            labels = np.ones(10, dtype=np.float32)
            users = np.arange(10)
            with patch.multiple(b.u, ROOT=root, BASE=root, FRESH=fresh):
                for parent, epochs, components in [('tsvdx4-60', 20, []),
                        ('tsvdx4-400lm', 20, ['bias', 'mf', 'nsvd1']), ('dnn-64', 5, ['base', 'net'])]:
                    outputs = {}
                    names = [f'{parent}_ep{i:02}' for i in range(1, epochs+1)]
                    names += [parent+'-'+c for c in components]
                    for name in names:
                        path = fresh/'preds_new'/(name+'.probex.npy')
                        np.save(path, labels)
                        outputs[str(path.relative_to(fresh))] = b.u.sha(path)
                    b.u.write(root/'fresh-receipts'/(parent+'.json'),
                              dict(source_commit=b.u.PIN, job=parent, outputs=outputs))
                bank = {name: labels for name in b.e.ARMS['all-fourteen']}
                manifest = b.freeze(labels, users, bank)
                entries = {r['name']: r for r in manifest['entries']}
                self.assertEqual(entries['dnn-64_ep05']['duplicate_of'], 'tsvdx4-60')
                # Components retain their separate unclipped semantics.
                self.assertIsNone(entries['tsvdx4-400lm-bias']['duplicate_of'])
                np.save(fresh/'preds_new/dnn-64_ep01.probex.npy', labels+1)
                with self.assertRaisesRegex(ValueError, 'Saved output changed'):
                    b.freeze(labels, users, bank)

    def test_metadata_aggregation_and_unseen_identity(self):
        context = b.context_features(np.array([0, 0, 1]), np.array([0, 1, 1]),
                    np.array([100., 200., 300.]), np.array([0, 2]), np.array([1, 2]), np.array([400., 500.]))
        np.testing.assert_allclose(context[0], [np.log(3)/7-1, np.log(3)/10-1,
                                               400/3000-.5, .25, .15])
        self.assertTrue(np.isfinite(context).all())
        np.testing.assert_allclose(context[1, [0, 1, 3, 4]], [-1, -1, .5, .5])


if __name__ == '__main__':
    unittest.main()
