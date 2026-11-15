import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from persona_fit import fit_user, probe_correction


class PersonaRecoveryTest(unittest.TestCase):
    def test_recurring_day_personas_predict_held_out_ratings(self):
        """Two persistent raters recur on different days within one account."""
        rng = np.random.default_rng(3)
        train_x, train_y, train_day = [], [], []
        held_x, held_y, held_day, implicit_x = [], [], [], []
        days = 80
        for day in range(days):
            rater = day % 2
            movies = []
            for event in range(6):
                movie = rng.normal(size=8)
                movie[0] += (1 if rater == 0 else -1) * 1.2
                rating_residual = (1 if rater == 0 else -1) * (.7*movie[1]+.2)
                rating_residual += rng.normal(scale=.05)
                movies.append(movie)
                if event == 5:
                    held_x.append(movie)
                    held_y.append(rating_residual)
                    held_day.append(day)
                else:
                    train_x.append(movie)
                    train_y.append(rating_residual)
                    train_day.append(day)
            implicit_x.extend(movies)
        x = np.column_stack((train_x, np.ones(len(train_x))))
        query = np.column_stack((held_x, np.ones(len(held_x))))
        train_day = np.asarray(train_day)
        one, two, centers, weights = fit_user(x, np.asarray(train_y), train_day, 1., 8)
        one_pred, two_pred, _ = probe_correction(
            query, np.asarray(held_day), train_day, weights,
            np.repeat(np.arange(days), 6), np.asarray(implicit_x), one, two, centers)
        one_rmse = np.sqrt(np.mean((one_pred - held_y)**2))
        two_rmse = np.sqrt(np.mean((two_pred - held_y)**2))
        self.assertLess(two_rmse, one_rmse * .5)


if __name__ == '__main__':
    unittest.main()
