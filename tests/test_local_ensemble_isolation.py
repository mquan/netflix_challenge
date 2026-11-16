"""End-to-end outer-label poisoning for the three reused blend stages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from gbdt_blend import HEADER, PROBE, blend_fold


class OuterIsolation(unittest.TestCase):
    def test_empty_support_bins_and_zero_network_outputs_use_pooled_fit(self):
        from round9_enn_blend import stack_apply, stack_fit
        p = np.zeros((12, 4))
        base = np.full(12, 3.6)
        target = np.arange(12, dtype=float)/10
        bins = np.ones(12, dtype=int)
        weights, alpha, losses = stack_fit(p, base, target, bins, 2,
                                           np.arange(12) % 2, [.01, 1.0])
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_allclose(weights[0], weights[1])
        np.testing.assert_allclose(stack_apply(weights, p, base, bins), target.mean())

    def test_poisoned_outer_labels_do_not_change_training_or_predictions(self):
        rng = np.random.default_rng(41)
        users, items, per_user = 240, 12, 5
        n = users*per_user
        probe = np.zeros(n, dtype=PROBE)
        probe['user'] = np.repeat(np.arange(users), per_user)
        probe['item'] = rng.integers(1, items+1, n)
        probe['day'] = rng.integers(1, 40, n)
        probe['rating'] = rng.integers(1, 6, n)
        held = blend_fold(probe['user']) == 0
        train_items = rng.integers(1, items+1, n, dtype=np.uint16)
        train_days = np.tile(np.arange(per_user, dtype=np.uint16), users)
        train_ratings = rng.integers(1, 6, n, dtype=np.uint8)
        offsets = np.arange(0, n+1, per_user, dtype='<u8')
        h = np.zeros(1, dtype=HEADER)
        for key, value in dict(magic=b'NFLXBIN1', users=users, items=items,
                               train_count=n, probe_count=n, implicit_count=n,
                               min_day=0, max_day=40, mean=3.0).items():
            h[key] = value
        preds = 3.0+rng.normal(0, .3, (n, 6))
        context = np.column_stack([np.ones(n), rng.normal(size=(n, 4))]).astype('f4')
        meta = np.column_stack([np.full(n, np.log1p(per_user)),
                                np.log1p(np.bincount(train_items, minlength=items+1)[probe['item']]),
                                rng.normal(size=(n, 9))]).astype('f4')
        outputs = []
        for poisoned in [False, True]:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                bank = root/'bank'
                (bank/'experiments').mkdir(parents=True)
                (root/'artifacts/r4').mkdir(parents=True)
                np.savez(root/'artifacts/r4/meta-context.npz', meta=meta, context=context)
                p = probe.copy()
                if poisoned:
                    p['rating'][held] = 6-p['rating'][held]
                data = root/'data.bin'
                with data.open('wb') as f:
                    for array in [h, offsets, train_items, train_days, train_ratings,
                                  p, offsets, train_items, train_days]:
                        f.write(array.tobytes())
                import hashlib
                inputs = {}
                for j in range(preds.shape[1]):
                    path = bank/f'p{j}.f32'
                    preds[:, j].astype('<f4').tofile(path)
                    inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                (bank/'experiments/UNION-round-manifest.json').write_text(json.dumps(dict(reports=dict(union=dict(inputs=inputs)))))
                (bank/'experiments/BOOST-support-inputs.json').write_text('{}')
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1')
                def run(script, *args):
                    result = subprocess.run([sys.executable, str(ROOT/'scripts'/script), *map(str, args)],
                                            cwd=root, env=env, text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT)
                    if result.returncode:
                        self.fail(result.stdout[-6000:])
                run('round4_boost.py', '--data', data, '--output', root/'tree', '--bank-root', bank,
                    '--rounds', 3, '--leaves', 3, '--min-leaf', 5, '--processes', 1, '--seeds', 29)
                run('round8_nn_blend.py', '--data', data, '--like', root/'tree.json', '--output', root/'nn',
                    '--epochs', 2, '--patience', 1, '--hidden', 4, '--layers', 1,
                    '--batch', 128, '--threads', 1, '--seeds', 29)
                run('round9_enn_blend.py', '--data', data, '--like', root/'tree.json', '--nn2', root/'nn',
                    '--output', root/'enn', '--nets', 4, '--hidden', 3, '--subset', 2,
                    '--epochs', 2, '--patience', 1, '--bins', 2, '--batch', 128,
                    '--threads', 1, '--max-folds', 1, '--seed', 29, '--stack-alphas', '.01', '1')
                output = {key: np.fromfile(root/name, dtype='<f4')[held] for key, name in
                          [('tree', 'tree.refit.f32'), ('nn', 'nn.nn.f32'), ('enn', 'enn.enn.f32')]}
                reports = {key: json.loads((root/name).read_text())['folds'][0] for key, name in
                           [('tree', 'tree.json'), ('nn', 'nn.json'), ('enn', 'enn.json')]}
                outputs.append((output, reports))
        for key in ['tree', 'nn', 'enn']:
            np.testing.assert_array_equal(outputs[0][0][key], outputs[1][0][key], err_msg=key)
        for family, fields in [('tree', ['chosen_rounds']), ('nn', ['chosen_epochs']),
                               ('enn', ['chosen_epochs', 'stack_alpha', 'stack_alpha_rmse'])]:
            for field in fields:
                self.assertEqual(outputs[0][1][family][field], outputs[1][1][family][field])


if __name__ == '__main__':
    unittest.main()
