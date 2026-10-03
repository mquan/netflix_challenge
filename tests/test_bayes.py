"""Contracts for the Bayesian timeSVD++ flipped sampler (`netflix bayes`)."""
import array
import math
import pathlib
import random
import re
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


def dataset(path, flip_probe=False, users=120, items=30, per_user=12, seed=7):
    """Rank-2 synthetic ratings; the last two dated ratings per user are probe."""
    rng = random.Random(seed)
    pu = [[rng.gauss(0, 1) for _ in range(2)] for _ in range(users)]
    qi = [[rng.gauss(0, 1) for _ in range(2)] for _ in range(items)]
    train, probe, implicit = [], [], []
    for u in range(users):
        chosen = rng.sample(range(items), per_user)  # random item order vs. dates
        days = sorted(rng.randrange(0, 60) for _ in chosen)
        rows = []
        for i, day in zip(chosen, days):
            score = 3.4 + 0.8*sum(a*b for a, b in zip(pu[u], qi[i])) + rng.gauss(0, .3)
            rows.append((day, i+1, max(1, min(5, round(score)))))
        rows.sort()
        train.append(rows[:-2])
        probe.extend((u, i, day, (6-r) if flip_probe else r) for day, i, r in rows[-2:])
        implicit.append(sorted((day, i) for day, i, _ in rows))
    flat = [r for rows in train for r in rows]
    mean = sum(r for _, _, r in flat)/len(flat)
    with path.open('wb') as f:
        f.write(struct.pack('<8sIIQQQHH4xd', b'NFPDAT2\0', users, items, len(flat),
                            len(probe), sum(map(len, implicit)), 0, 59, mean))
        def put(code, values):
            values = list(values)
            f.write(struct.pack('<'+str(len(values))+code, *values))
        offsets = [0]
        for rows in train: offsets.append(offsets[-1]+len(rows))
        put('Q', offsets)
        put('H', (i for _, i, _ in flat)); put('H', (d for d, _, _ in flat)); put('B', (r for _, _, r in flat))
        for u, i, day, r in probe: f.write(struct.pack('<IHHB3x', u, i, day, r))
        offsets = [0]
        for rows in implicit: offsets.append(offsets[-1]+len(rows))
        put('Q', offsets)
        put('H', (i for rows in implicit for _, i in rows)); put('H', (d for rows in implicit for d, _ in rows))
    return probe


def floats(path):
    a = array.array('f'); a.frombytes(path.read_bytes()); return a


def rmse(pred, probe):
    return math.sqrt(sum((r-min(5, max(1, p)))**2 for p, (_, _, _, r) in zip(pred, probe))/len(probe))


class BayesTests(unittest.TestCase):
    def test_incremental_residuals_stay_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp); dataset(p/'d.bin')
            for threads in (1, 3):
                out = run('bayes', '--data', p/'d.bin', '--model', p/'m.bin', '--factors', 3,
                          '--samples', 3, '--burn-in', 1, '--threads', threads, '--session', 2,
                          '--check-residuals').stdout
                drift = [float(x) for x in re.findall(r'residual_drift=(\S+)', out)]
                self.assertEqual(len(drift), 3)
                self.assertLess(max(drift), 1e-6, out)

    def test_learns_low_rank_structure(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp); probe = dataset(p/'d.bin')
            run('bayes', '--data', p/'d.bin', '--model', p/'m.bin', '--factors', 4,
                '--samples', 60, '--burn-in', 10, '--predictions', p/'b.f32')
            pred = floats(p/'b.f32')
            self.assertEqual(len(pred), len(probe))
            # Global-mean predictor as the bar to clear.
            raw = (p/'d.bin').read_bytes()
            mean = struct.unpack_from('<d', raw, 48)[0]
            self.assertLess(rmse(pred, probe), rmse([mean]*len(probe), probe) - .1)
            # Plain Bayesian MF should approach the generator's oracle (~0.44
            # after rounding noise) on this rank-2 data.
            run('bayes', '--data', p/'d.bin', '--model', p/'mf.bin', '--factors', 4,
                '--samples', 60, '--burn-in', 10, '--predictions', p/'mf.f32', '--user-bag', 0,
                '--item-bag', 0, '--day', 0, '--item-bin', 0, '--session', 0, '--drift', 0)
            self.assertLess(rmse(floats(p/'mf.f32'), probe), .55)

    def test_probe_labels_are_never_read(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            dataset(p/'a.bin'); dataset(p/'b.bin', flip_probe=True)
            for name in ('a', 'b'):
                run('bayes', '--data', p/(name+'.bin'), '--model', p/(name+'.model'), '--factors', 3,
                    '--samples', 3, '--burn-in', 1, '--predictions', p/(name+'.f32'))
            self.assertEqual((p/'a.model').read_bytes()[16:], (p/'b.model').read_bytes()[16:])
            self.assertEqual((p/'a.f32').read_bytes(), (p/'b.f32').read_bytes())
            run('bayes', '--data', p/'b.bin', '--model', p/'x.model', '--init-model', p/'a.model',
                '--samples', 4, ok=False)

    def test_resume_and_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp); dataset(p/'d.bin')
            common = ('--data', p/'d.bin', '--factors', 3, '--burn-in', 1)
            run('bayes', *common, '--model', p/'full.model', '--samples', 4,
                '--snapshot-every', 2, '--predictions', p/'full.f32')
            self.assertTrue((p/'full.s0002.f32').exists() and (p/'full.s0004.f32').exists())
            self.assertEqual((p/'full.s0004.f32').read_bytes(), (p/'full.f32').read_bytes())
            run('bayes', *common, '--model', p/'half.model', '--samples', 2)
            run('bayes', '--data', p/'d.bin', '--model', p/'resumed.model', '--init-model', p/'half.model',
                '--samples', 4, '--burn-in', 1, '--predictions', p/'resumed.f32')
            # Resumed residuals are recomputed from parameters, so draws may
            # differ only by floating-point rounding.
            a, b = floats(p/'full.f32'), floats(p/'resumed.f32')
            self.assertLess(max(abs(x-y) for x, y in zip(a, b)), 1e-3)
            # A finished model re-emits its posterior mean without sampling.
            run('bayes', '--data', p/'d.bin', '--model', p/'again.model', '--init-model', p/'full.model',
                '--samples', 4, '--predictions', p/'again.f32')
            self.assertEqual((p/'again.f32').read_bytes(), (p/'full.f32').read_bytes())


if __name__ == '__main__':
    unittest.main()
