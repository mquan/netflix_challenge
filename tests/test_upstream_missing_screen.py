import unittest
import numpy as np
import upstream_missing_screen as s
import upstream_exchange as x
import upstream_recipe_costs as costs


def job(outputs=()):
    return dict(outputs=list(outputs), inputs=[], inputs_from=[])


class MissingScreenTests(unittest.TestCase):
    def test_exact_producer_precedes_auxiliary_name_inference(self):
        jobs = {'root': job(), 'other': job(['preds_new/root_ep03.probex.npy'])}
        self.assertEqual(s.producer_for('root_ep03', jobs)[0], 'other')
        self.assertEqual(s.producer_for('root-mf', jobs)[0], 'root')
        self.assertEqual(s.producer_for('root_ep09', jobs)[0], 'root')
        self.assertIsNone(s.producer_for('rootr__knn3', jobs)[0])
        self.assertIsNone(s.producer_for('root__unregistered', jobs)[0])

    def test_ambiguous_native_output_is_rejected(self):
        jobs = {k:job(['preds_new/model.probex.npy']) for k in ['a','b']}
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            s.producer_for('model', jobs)

    def test_added_columns_preserve_unpenalized_baseline_intercept(self):
        rng = np.random.default_rng(11)
        values = rng.uniform(0, 6, (100, 4))
        labels = rng.uniform(1, 5, 100)
        folds = np.tile(np.arange(5), 20)
        z = np.column_stack([values[:, :2], np.ones(100), values[:, 2:]])
        arms = {'control': [], 'one': [3], 'two': [3, 4]}
        actual = s.score_arms(lambda lo,hi:z[lo:hi], labels, folds, 3, arms)
        for record, columns in zip(actual, [values[:,:2], values[:,:3], values]):
            pred = x.crossfit(columns, labels, folds)
            self.assertAlmostEqual(record['rmse'], x.metric(pred, labels), places=12)
            np.testing.assert_allclose(record['fold_rmse'],
                 [x.metric(pred[folds==f], labels[folds==f]) for f in range(5)], atol=1e-12)

    def test_memory_bound_accounts_for_both_dense_user_day_arrays(self):
        row = dict(users=2, items=3, user_day_pairs=5, rows=20)
        low = costs.tx_lower_bound(row, 7, True)
        dense = costs.tx_lower_bound(row, 7, False, 2)
        self.assertEqual(low['factor_bytes'], 4*7*(3*2+18*3))
        self.assertEqual(dense['factor_bytes']-low['factor_bytes'], 4*7*(3+2*5))
        self.assertEqual(dense['neighborhood_bytes'], 4*3*3*2)
        self.assertEqual(dense['minimum_total_bytes'], dense['factor_bytes']+4*3*3*2+15*20)

    def test_scoring_fold_labels_do_not_affect_any_subset_coefficients(self):
        rng = np.random.default_rng(21)
        z = np.column_stack([rng.normal(size=(50, 2)), np.ones(50), rng.normal(size=50)])
        folds = np.tile(np.arange(5), 10)
        y = rng.normal(size=50)
        grams = np.array([z[folds==f].T@z[folds==f] for f in range(5)])
        def fit(target):
            rhs = np.array([z[folds==f].T@target[folds==f] for f in range(5)])
            return s.solve_subsets(grams, rhs, 3, {'control': [], 'single': [3]})
        before = fit(y)
        y[folds==2] += 100
        after = fit(y)
        np.testing.assert_allclose(before[2], after[2], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
