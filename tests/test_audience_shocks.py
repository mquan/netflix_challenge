import dataclasses
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from audience_shocks import Config, Upstream, candidates, design, detect, export, pilot, rows


class AudienceShocksTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(19)
        u = np.repeat(np.arange(100), 20)
        i = np.tile(np.arange(20), 100)
        return rows(u, i, rng.integers(0, 140, len(u)), rng.integers(1, 6, len(u)))

    def test_target_movie_exclusion(self):
        r = self.fixture(); cfg = Config()
        a = Upstream(r, 100, 20, cfg)
        changed = {k: v.copy() for k, v in r.items()}
        changed['rating'][r['item'] == 0] = 1
        b = Upstream(changed, 100, 20, cfg)
        query = {k: v[r['item'] == 0] for k, v in r.items()}
        np.testing.assert_allclose(a.profile(query), b.profile(query), atol=1e-14)

    def test_outer_labels_cannot_change_fit_or_predictions(self):
        r = self.fixture(); cfg = Config(permutations=2, bootstrap=10, min_users=2)
        event = dict(movie=0, start=0, end=20, confidence=.8, before_users=30, after_users=30)
        with mock.patch('audience_shocks.detect', return_value=([event], {})):
            _, a = pilot(r, 100, 20, cfg)
        self.assertTrue(np.any(a['models']['full'] != 0))
        changed = {k: v.copy() for k, v in r.items()}
        changed['rating'][a['split'] == 3] = 1
        with mock.patch('audience_shocks.detect', return_value=([event], {})):
            _, b = pilot(changed, 100, 20, cfg)
        self.assertEqual(a['events'], b['events'])
        for key in a['models']:
            np.testing.assert_array_equal(a['models'][key], b['models'][key])
        for key in a['predictions']:
            np.testing.assert_array_equal(a['predictions'][key], b['predictions'][key])

    def test_boundaries_overlaps_and_export_order(self):
        r = self.fixture(); up = Upstream(r, 100, 20, Config())
        query = rows([1]*4, [0]*4, np.array([3, 4, 7, 8])*7, [0]*4)
        event = dict(movie=0, start=4, end=8, confidence=.75, before_users=20, after_users=25)
        x = design(query, up, [event, event], 'bias')
        np.testing.assert_array_equal(x.sum(1), [0, .75, .75, 0])
        state = dict(upstream=up, events=[event], models={'full': np.ones(5)})
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp)/'probe'
            export(query, state, path)
            p = np.fromfile(str(path)+'.f32', dtype='<f4')
            expected = up.base(query)+design(query, up, [event], 'full')@np.ones(5)
            np.testing.assert_array_equal(p, expected.astype('f4'))
            correction = np.fromfile(str(path)+'.correction.f32', dtype='<f4')
            np.testing.assert_array_equal(correction[[0, 3]], 0)

    def test_composition_shift_without_volume_change(self):
        week = np.repeat(np.arange(16), 80)
        movie = np.tile(np.repeat([0, 1], 40), 16)
        user = np.tile(np.arange(80), 16)
        r = rows(user, movie, week*7, np.ones(len(week))*3)
        profile = np.zeros((len(week), 4))
        profile[:, 0] = np.where(week < 8, -1., 1.) * np.where(movie == 0, 1., -1.)
        found = candidates(r, profile, Config(min_users=5))
        event = next(e for e in found if e['movie'] == 0 and e['start'] == 8)
        self.assertEqual(event['activity'], 0.)
        self.assertGreater(event['composition'], 3.)

    def test_known_activity_shift_and_null(self):
        # 40 users/movie/week, constant overall platform volume. Movie 0
        # replaces movie 1 in the second half, while movie 2 stays constant.
        u, i, w = [], [], []
        for week in range(16):
            for movie, count in ((0, 10 if week < 8 else 70), (1, 70 if week < 8 else 10), (2, 40)):
                u.extend(range(count)); i.extend([movie]*count); w.extend([week]*count)
        r = rows(u, i, np.array(w)*7, np.ones(len(u))*3)
        cfg = Config(permutations=5, min_users=5)
        class Profiles:
            def profile(self, r):
                return np.zeros((len(r['user']), 4))
        ev, _ = detect(r, Profiles(), cfg)
        self.assertTrue(any(e['movie'] == 0 and e['start'] <= 8 < e['end'] for e in ev))
        self.assertFalse(any(e['movie'] == 2 for e in ev))
        null = rows(np.tile(np.arange(40), 16*3), np.tile(np.repeat(np.arange(3), 40), 16),
                    np.repeat(np.arange(16)*7, 120), np.ones(1920)*3)
        ev, _ = detect(null, Profiles(), cfg)
        self.assertEqual(ev, [])


if __name__ == '__main__':
    unittest.main()
