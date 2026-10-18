"""Contracts for the user-user residual kNN (netflix user-knn)."""
import array
import math
import pathlib
import struct
import tempfile
import unittest

from test_joint import run

USERS, ITEMS = 80, 8


def rows():
    """Train/probe rows. Day equals item-1, so item order is date order."""
    train, probe = {}, {}
    for u in range(USERS):
        group = u % 2
        items = [i for i in range(1, ITEMS+1) if (u*5+i*3) % 4 != 0]
        held = items.pop((u*3) % len(items))
        def rating(i):
            r = 5 if i % 2 == group else 1
            return max(1, min(5, r+((u*7+i) % 3)-1))
        train[u] = [(i, i-1, rating(i)) for i in items]
        probe[u] = (held, held-1, rating(held))
    return train, probe


def write_dataset(path, flip_probe=False):
    train, probe = rows()
    flat = [r for u in range(USERS) for r in train[u]]
    with path.open('wb') as f:
        f.write(struct.pack('<8sIIQQQHH4xd', b'NFPDAT2\0', USERS, ITEMS, len(flat), USERS,
                            len(flat)+USERS, 0, ITEMS-1, sum(r for _, _, r in flat)/len(flat)))
        def put(code, values):
            values = list(values)
            f.write(struct.pack('<'+str(len(values))+code, *values))
        offsets = [0]
        for u in range(USERS):
            offsets.append(offsets[-1]+len(train[u]))
        put('Q', offsets)
        put('H', [i for i, _, _ in flat])
        put('H', [day for _, day, _ in flat])
        put('B', [r for _, _, r in flat])
        for u in range(USERS):
            i, day, r = probe[u]
            f.write(struct.pack('<IHHB3x', u, i, day, 6-r if flip_probe else r))
        known = {u: sorted([(i, day) for i, day, _ in train[u]]+[probe[u][:2]]) for u in range(USERS)}
        offsets = [0]
        for u in range(USERS):
            offsets.append(offsets[-1]+len(known[u]))
        put('Q', offsets)
        put('H', [i for u in range(USERS) for i, _ in known[u]])
        put('H', [day for u in range(USERS) for _, day in known[u]])
    return train, probe, len(flat)


def floats(path):
    a = array.array('f')
    a.frombytes(pathlib.Path(path).read_bytes())
    return list(a)


def write_floats(path, values):
    pathlib.Path(path).write_bytes(array.array('f', values).tobytes())


def reference(train, probe, base, alpha, power, shrink, min_common):
    """Exact shrunk residual correlation over every rater (no screen)."""
    res = {u: {i: r-base for i, _, r in train[u]} for u in train}
    out, support, count = [], [], []
    for u in range(USERS):
        i = probe[u][0]
        num = den = 0.0
        used = 0
        for v in range(USERS):
            if i not in res[v]:
                continue
            common = [j for j in res[u] if j in res[v]]
            sxy = sum(res[u][j]*res[v][j] for j in common)
            sxx = sum(res[u][j]**2 for j in common)
            syy = sum(res[v][j]**2 for j in common)
            if len(common) < min_common or sxx <= 0 or syy <= 0:
                continue
            rho = sxy/math.sqrt(sxx*syy)*len(common)/(len(common)+alpha)
            if rho <= 0:
                continue
            w = rho**power
            num += w*res[v][i]
            den += w
            used += 1
        out.append(base+num/(den+shrink))
        support.append(den)
        count.append(used)
    return out, support, count


class UserKnnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, name, flip=False):
        p = self.p
        train, probe, n = write_dataset(p/(name+'.bin'), flip)
        run('joint', '--data', p/(name+'.bin'), '--model', p/(name+'.joint'),
            '--iterations', 1, '--factors', 4)
        write_floats(p/(name+'.train'), [3.0]*n)
        write_floats(p/(name+'.probe'), [3.0]*USERS)
        return train, probe

    def user_knn(self, name, out, *extra):
        p = self.p
        return run('user-knn', '--data', p/(name+'.bin'), '--joint-model', p/(name+'.joint'),
                   '--base-train', p/(name+'.train'), '--base-probe', p/(name+'.probe'),
                   '--predictions', p/(out+'.f32'), '--support', p/(out+'.support'),
                   '--support-count', p/(out+'.count'), '--factor-predictions', p/(out+'.factor'),
                   '--candidates', 1000, '--neighbors', 1000, '--alpha', 3, '--power', 2,
                   '--shrink', .5, '--min-common', 2, *extra)

    def test_matches_exact_reference_when_screen_keeps_every_rater(self):
        train, probe = self.prepare('a')
        self.user_knn('a', 'out', '--screen-rank', 2)
        want, support, count = reference(train, probe, 3.0, 3, 2, .5, 2)
        got = floats(self.p/'out.f32')
        self.assertTrue(any(abs(x-3.0) > .05 for x in want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=4)
        for g, w in zip(floats(self.p/'out.support'), support):
            self.assertAlmostEqual(g, w, places=4)
        self.assertEqual(floats(self.p/'out.count'), [float(c) for c in count])

    def test_probe_labels_and_threads_do_not_change_outputs(self):
        self.prepare('a')
        self.prepare('b', flip=True)
        self.user_knn('a', 'a1', '--candidates', 5, '--neighbors', 3, '--threads', 1)
        self.user_knn('b', 'b3', '--candidates', 5, '--neighbors', 3, '--threads', 3)
        for suffix in ('.f32', '.support', '.count', '.factor'):
            self.assertEqual((self.p/('a1'+suffix)).read_bytes(), (self.p/('b3'+suffix)).read_bytes())

    def test_query_subset_leaves_other_rows_at_base_and_size_check(self):
        self.prepare('a')
        self.user_knn('a', 'full', '--screen-rank', 0)
        self.user_knn('a', 'half', '--screen-rank', 0, '--query-modulo', 2)
        full, half = floats(self.p/'full.f32'), floats(self.p/'half.f32')
        for u in range(USERS):
            self.assertEqual(half[u], full[u] if u % 2 == 0 else 3.0)
        run('user-knn', '--data', self.p/'a.bin', '--joint-model', self.p/'a.joint',
            '--base-train', self.p/'a.probe', '--base-probe', self.p/'a.probe',
            '--predictions', self.p/'bad.f32', ok=False)


if __name__ == '__main__':
    unittest.main()
