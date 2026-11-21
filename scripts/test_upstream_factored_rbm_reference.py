"""Independent scalar CD reference and analytical checks for the GPU-port oracle."""
import unittest
import numpy as np
from upstream_factored_rbm_reference import batch_gradients, expected, update


def fixture():
    rng = np.random.default_rng(31)
    shapes = dict(p=(4, 5, 2), q=(2, 3), d=(4, 3), bv=(4, 5), bh=(3,), bu=(3, 5), but=(5, 5))
    m = {k: rng.normal(0, .15, size=s) for k, s in shapes.items()}
    b = dict(items=np.array([0, 2, 1, 2, 3, 0]), users=np.array([0, 0, 1, 1, 2, 2]),
             ratings=np.array([0, 4, 2, 1, 3, 0]), days=np.array([0, 1, 2, 2, 3, 4]),
             conditional_items=np.array([0, 2, 3, 1, 2, 0, 3]),
             conditional_users=np.array([0, 0, 0, 1, 1, 2, 2]), day_owner=np.array([0, 0, 1, 2, 2]))
    return m, b, rng.random((3, 3, 3)), rng.random((3, 6))


def scalar(m, b, hu, vu):
    """Literal per-user/per-rating/per-factor loops, separate from batch algebra."""
    grads = {k: np.zeros_like(v) for k, v in m.items()}
    states = b['ratings'].copy(); pos = []; neg = []
    ni, _, nf = m['p'].shape; nh = len(m['bh'])
    sigmoid = lambda a: 1 / (1 + np.exp(-a))
    for user in range(len(m['bu'])):
        rows = np.flatnonzero(b['users'] == user)
        cr = b['conditional_items'][b['conditional_users'] == user]
        def acts(v):
            result = m['bh'].copy()
            for j in range(nh):
                for f in range(nf):
                    z = sum(m['p'][b['items'][row], v[t], f] for t, row in enumerate(rows))
                    result[j] += z * m['q'][f, j]
                for item in cr: result[j] += m['d'][item, j]
            return result
        data = b['ratings'][rows]; v = data.copy(); a = acts(v); ep = 2 * sigmoid(a) - 1
        for step in range(len(hu)):
            h = np.where(hu[step, user] < sigmoid(a), 1., -1.)
            for t, row in enumerate(rows):
                item, day = b['items'][row], b['days'][row]
                logits = m['bv'][item] + m['bu'][user] + m['but'][day]
                for k in range(5):
                    for f in range(nf):
                        logits[k] += m['p'][item, k, f] * sum(m['q'][f, j] * h[j] for j in range(nh))
                ex = np.exp(logits - max(logits)); draw = vu[step, row] * sum(ex)
                v[t] = 4
                for k in range(5):
                    if draw <= ex[k]: v[t] = k; break
                    draw -= ex[k]
            a = acts(v)
        en = 2 * sigmoid(a) - 1; states[rows] = v; pos.append(ep); neg.append(en)
        for t, row in enumerate(rows):
            item, day, kd, km = b['items'][row], b['days'][row], data[t], v[t]
            for f in range(nf):
                grads['p'][item, kd, f] += sum(m['q'][f, j] * ep[j] for j in range(nh))
                grads['p'][item, km, f] -= sum(m['q'][f, j] * en[j] for j in range(nh))
                for j in range(nh):
                    grads['q'][f, j] += m['p'][item, kd, f] * ep[j] - m['p'][item, km, f] * en[j]
            for key, index in [('bv', item), ('bu', user), ('but', day)]:
                grads[key][index, kd] += 1; grads[key][index, km] -= 1
        grads['bh'] += ep - en
        for item in cr: grads['d'][item] += ep - en
    return dict(grads, visible_state=states, positive_expected=np.array(pos), negative_expected=np.array(neg))


class ReferenceTests(unittest.TestCase):
    def test_all_cd3_gradients_against_literal_user_loops(self):
        m, b, hu, vu = fixture(); original = {k: v.copy() for k, v in m.items()}
        wanted = scalar(m, b, hu, vu)
        for tile in (1, 2, 16384):
            got = batch_gradients(m, b, hu, vu, tile)
            for k, value in wanted.items(): np.testing.assert_allclose(got[k], value, atol=2e-14, rtol=2e-13)
            for k in m: np.testing.assert_array_equal(m[k], original[k])

    def test_bipolar_expectation_and_native_categorical_boundary(self):
        self.assertEqual(expected(np.array([0.]))[0], 0.)
        m, b, hu, vu = fixture(); m = {k: np.zeros_like(v) for k, v in m.items()}
        vu[:] = np.array([0., .2, .4, .6, .8, .999])
        got = batch_gradients(m, b, hu[:1], vu[:1])
        np.testing.assert_array_equal(got['visible_state'], [0, 0, 1, 2, 3, 4])
        for k in ('p', 'q', 'd', 'bh'): np.testing.assert_array_equal(got[k], 0)

    def test_bias_gradient_conservation(self):
        m, b, hu, vu = fixture(); g = batch_gradients(m, b, hu, vu)
        for key in ('bv', 'bu', 'but'): np.testing.assert_array_equal(g[key].sum(axis=1), 0)
        np.testing.assert_array_equal(g['bu'].sum(axis=0), g['bv'].sum(axis=0))
        np.testing.assert_array_equal(g['but'].sum(axis=0), g['bu'].sum(axis=0))

    def test_updates_unaveraged_local_bias_and_untouched_shared_momentum(self):
        m, b, hu, vu = fixture(); g = batch_gradients(m, b, hu, vu)
        velocity = {k: np.full_like(v, .03) for k, v in m.items()}
        for key in ('p', 'q', 'd', 'bv', 'bh', 'bu', 'but'): g[key] = np.zeros_like(m[key])
        g['bu'][0, 0] = 2; g['but'][0, 0] = -3
        new, mv = update(m, velocity, g)
        np.testing.assert_allclose(mv['p'], .9 * velocity['p'] - .02 / 3 * .001 * m['p'])
        np.testing.assert_allclose(mv['bv'], .9 * velocity['bv'])
        self.assertAlmostEqual(mv['bu'][0, 0], .027 + .001 * (2 - .01 * m['bu'][0, 0]))
        self.assertAlmostEqual(mv['but'][0, 0], .027 + .0005 * (-3 - .01 * m['but'][0, 0]))
        g['batch_users'] = 1; _, one = update(m, velocity, g)
        np.testing.assert_allclose(one['d'], .9 * velocity['d'] - .02 * .001 * m['d'])
        for k in m: np.testing.assert_allclose(new[k], m[k] + mv[k])

    def test_reject_wrong_day_owner_and_duplicate_conditional_pairs(self):
        m, b, hu, vu = fixture(); wrong = dict(b); wrong['day_owner'] = np.zeros(5, dtype=int)
        with self.assertRaises(ValueError): batch_gradients(m, wrong, hu, vu)
        wrong = dict(b); wrong['conditional_items'] = np.r_[b['conditional_items'], 0]
        wrong['conditional_users'] = np.r_[b['conditional_users'], 0]
        with self.assertRaises(ValueError): batch_gradients(m, wrong, hu, vu)


if __name__ == '__main__': unittest.main()
