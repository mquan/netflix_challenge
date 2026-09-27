"""Small numeric contracts for the new optimizer, persistence and blend folds."""
import array
import pathlib
import struct
import subprocess
import tempfile
import unittest


BIN = str(pathlib.Path('build/netflix').resolve())


def run(*args, ok=True):
    result = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if ok and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not ok and not result.returncode:
        raise AssertionError('expected failure')
    return result


def fold(user):
    mask = (1 << 64)-1
    x = user+0x9e3779b97f4a7c15
    x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27))*0x94d049bb133111eb) & mask
    return (x ^ (x >> 31)) % 5


def dataset(path, change_fold=None, correlated_train=False):
    users, items = 100, 5
    train = [1+(u+(0 if correlated_train else i)) % 5 for u in range(users) for i in range(3)]
    probe = [(u, i+1, i, 1+(u+i) % 5) for u in range(users) for i in (3, 4)]
    with path.open('wb') as f:
        f.write(struct.pack('<8sIIQQQHH4xd', b'NFPDAT2\0', users, items,
                            len(train), len(probe), users*5, 0, 4, sum(train)/len(train)))
        def put(code, values):
            values = list(values)
            f.write(struct.pack('<'+str(len(values))+code, *values))
        put('Q', (u*3 for u in range(users+1)))
        put('H', [1, 2, 3]*users)
        put('H', [0, 1, 2]*users)
        put('B', train)
        for u, i, day, r in probe:
            if change_fold is not None and fold(u) == change_fold:
                r = 6-r
            f.write(struct.pack('<IHHB3x', u, i, day, r))
        put('Q', (u*5 for u in range(users+1)))
        put('H', [1, 2, 3, 4, 5]*users)
        put('H', [0, 1, 2, 3, 4]*users)
    return probe


class JointTests(unittest.TestCase):
    def test_implicit_matches_naive(self):
        run('check-implicit')

    def test_probe_labels_do_not_change_one_epoch_training(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            dataset(p/'a.bin')
            dataset(p/'b.bin', change_fold=0)
            for name in ('a', 'b'):
                run('joint', '--data', p/(name+'.bin'), '--model', p/(name+'.model'),
                    '--iterations', 1, '--factors', 4, '--day-factors', 2, '--predictions', p/(name+'.f32'))
            # First 16 bytes contain magic+dataset fingerprint; all parameters
            # must be bit-identical even though validation labels changed.
            self.assertEqual((p/'a.model').read_bytes()[16:], (p/'b.model').read_bytes()[16:])
            self.assertEqual((p/'a.f32').read_bytes(), (p/'b.f32').read_bytes())
            run('eval-joint', '--data', p/'a.bin', '--model', p/'a.model',
                '--predictions', p/'reload.f32')
            self.assertEqual((p/'a.f32').read_bytes(), (p/'reload.f32').read_bytes())
            run('eval-joint', '--data', p/'b.bin', '--model', p/'a.model', ok=False)

    def test_neighborhood_reload_and_probe_label_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            for name in ('a', 'b'):
                dataset(p/(name+'.bin'), change_fold=0 if name == 'b' else None, correlated_train=True)
                run('time-baseline', '--data', p/(name+'.bin'), '--model', p/(name+'.base'), '--iterations', 1)
                run('graph', '--data', p/(name+'.bin'), '--graph', p/(name+'.graph'),
                    '--neighbors', 2, '--history-cap', 3)
                run('neighborhood', '--data', p/(name+'.bin'), '--time-model', p/(name+'.base'),
                    '--graph', p/(name+'.graph'), '--model', p/(name+'.model'),
                    '--predictions', p/(name+'.f32'), '--iterations', 1)
            self.assertEqual((p/'a.graph').read_bytes()[16:], (p/'b.graph').read_bytes()[16:])
            self.assertEqual((p/'a.model').read_bytes()[16:], (p/'b.model').read_bytes()[16:])
            self.assertEqual((p/'a.f32').read_bytes(), (p/'b.f32').read_bytes())
            weights = array.array('f'); weights.frombytes((p/'a.model').read_bytes()[36:])
            self.assertTrue(any(abs(w) > 1e-6 for w in weights))
            run('eval-neighborhood', '--data', p/'a.bin', '--time-model', p/'a.base',
                '--graph', p/'a.graph', '--model', p/'a.model', '--predictions', p/'reload.f32')
            self.assertEqual((p/'a.f32').read_bytes(), (p/'reload.f32').read_bytes())
            run('eval-neighborhood', '--data', p/'b.bin', '--time-model', p/'b.base',
                '--graph', p/'b.graph', '--model', p/'a.model', '--predictions', p/'bad.f32', ok=False)

    def test_blend_heldout_users_do_not_fit_their_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            rows = dataset(p/'a.bin')
            dataset(p/'b.bin', change_fold=0)
            for j in (0, 1):
                pred = array.array('f', [r + (.1 if j == 0 else -.2)*(1+u % 3)
                                        for u, _, _, r in rows])
                (p/('p'+str(j))).write_bytes(pred.tobytes())
            for name in ('a', 'b'):
                run('ensemble', '--data', p/(name+'.bin'), '--prediction', p/'p0',
                    '--prediction', p/'p1', '--predictions', p/(name+'.f32'))
            a = array.array('f'); a.frombytes((p/'a.f32').read_bytes())
            b = array.array('f'); b.frombytes((p/'b.f32').read_bytes())
            for n, row in enumerate(rows):
                if fold(row[0]) == 0:
                    self.assertEqual(a[n], b[n])
            (p/'p0').write_bytes(b'bad')
            run('ensemble', '--data', p/'a.bin', '--prediction', p/'p0', ok=False)


if __name__ == '__main__':
    unittest.main()
