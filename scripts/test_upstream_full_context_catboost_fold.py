"""Scheduling/barrier fixtures; fake models never admit production results."""
import unittest
from types import SimpleNamespace
import numpy as np
import upstream_full_context_catboost_fold as module


class FoldTests(unittest.TestCase):
    def fixture(self):
        groups = np.tile(np.arange(10, dtype=np.int8), 4)
        users = np.arange(40, dtype=np.int64)
        features = np.zeros((40, 713), dtype='<f4')
        features[:, 0] = users
        labels = (users % 5 + 1).astype(np.float64)
        baselines = np.stack([2.1 + users * .01, 2.3 + users * .01])
        return features, baselines, labels, users, groups

    def test_all_inner_before_fresh_full_and_zero_slot_denominator(self):
        x, b, y, users, groups = self.fixture()
        events = []; saved = []; populations = []; counts = iter([2, 0, 3])
        p = module.masks(users, groups, 2)
        def train(features, residuals, seed, iterations, guard, validation=None):
            events.append(('inner' if validation is not None else 'full', seed, iterations))
            eligible = p['fit'] if validation is not None else p['train']
            np.testing.assert_array_equal(features[:, 0], users[eligible])
            pop = 0 if validation is not None else 1
            np.testing.assert_array_equal(residuals, y[eligible] - b[pop, eligible])
            np.testing.assert_array_equal(features[:, -1], b[pop, eligible].astype('<f4'))
            if validation is not None:
                np.testing.assert_array_equal(validation[0][:, 0], users[p['valid']])
                np.testing.assert_array_equal(validation[1], y[p['valid']] - b[0, p['valid']])
            return SimpleNamespace(tree_count_=iterations, seed=seed)
        def select(model, residuals):
            np.testing.assert_array_equal(residuals, y[p['valid']] - b[0, p['valid']])
            count = next(counts)
            return count, {'selected_iterations': count}
        def predict(model, features, guard):
            np.testing.assert_array_equal(features[:, 0], users[p['held']])
            return np.full(len(features), .12 if model.seed == 2028 else -.03, dtype=np.float64)
        result = module.run(x, b, y, users, groups, 2, save_model=lambda m, s, c: saved.append((s, c)),
            save_population=lambda stage, proof: populations.append(stage), release=lambda: None, guard=lambda: None,
            train=train, predict=predict, select=select)
        self.assertEqual(events, [('inner', 2028, 2000), ('inner', 9, 2000), ('inner', 15, 2000),
                                  ('full', 2028, 2), ('full', 15, 3)])
        self.assertEqual(saved, [(2026, 2), (13, 3)])
        self.assertEqual(populations, ['inner', 'full'])
        expected = np.clip(b[1, p['held']] + .12 / 3 - .03 / 3, 1.05, 4.95).astype('<f4')
        self.assertEqual(result['prediction'].tobytes(), expected.tobytes())

    def test_held_outcome_changes_do_not_change_any_fit_or_prediction(self):
        results = []
        for replace_held in (False, True):
            x, b, y, users, groups = self.fixture(); p = module.masks(users, groups, 0)
            if replace_held: y[p['held']] = 5
            fitted = []
            def train(features, residuals, seed, iterations, guard, validation=None):
                fitted.append((features.tobytes(), residuals.tobytes(), seed))
                return SimpleNamespace(tree_count_=iterations)
            result = module.run(x, b, y, users, groups, 0, save_model=lambda *a: None,
                save_population=lambda *a: None, release=lambda: None, guard=lambda: None,
                train=train, select=lambda *a: (0, {}), predict=lambda *a: self.fail('zero slot predicted'))
            results.append((fitted, result['prediction'].tobytes()))
        self.assertEqual(results[0], results[1])

    def test_actual_CPU_inner_selection_full_refits_and_all_held_bits(self):
        x, b, y, users, groups = self.fixture()
        x[:, 1] = y
        # Synthetic signal only; no production arrays or score are admitted.
        b[:] = 2.5
        saved = {}
        result = module.run(x, b, y, users, groups, 0,
            save_model=lambda model, seed, count: saved.update({seed: model}),
            save_population=lambda *a: None, release=lambda: None, guard=lambda: None)
        p = module.masks(users, groups, 0)
        correction = np.zeros(int(p['held'].sum()), dtype=np.float64)
        self.assertTrue(any(result['chosen_iterations']))
        for seed, count in zip(module.SEEDS, result['chosen_iterations']):
            if count:
                self.assertEqual(saved[seed].tree_count_, count)
                correction += module.backend.predict(saved[seed], x[p['held']], lambda: None) / 3
            else:
                self.assertNotIn(seed, saved)
        expected = np.clip(b[1, p['held']] + correction, 1.05, 4.95).astype('<f4')
        self.assertEqual(result['prediction'].tobytes(), expected.tobytes())

    def test_cross_group_user_or_mutating_callback_rejected(self):
        x, b, y, users, groups = self.fixture(); users[1] = users[0]
        with self.assertRaises(ValueError): module.masks(users, groups, 0)
        x, b, y, users, groups = self.fixture()
        def mutate(*args, **kwargs):
            x[0, 2] = 123
            return SimpleNamespace(tree_count_=2000)
        with self.assertRaisesRegex(ValueError, 'changed'):
            module.run(x, b, y, users, groups, 0, save_model=lambda *a: None,
                save_population=lambda *a: None, release=lambda: None, guard=lambda: None,
                train=mutate, select=lambda *a: (0, {}))


if __name__ == '__main__': unittest.main()
