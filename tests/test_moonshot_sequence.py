"""M0 contracts for the moonshot sequence model (MOONSHOT_REBUILD.md)."""
import dataclasses
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import moonshot_sequence as ms
from test_selection_objective import fixture

N1_SOURCE = ROOT / '.claude/worktrees/honest-residual-chains/scripts/choice_sequence.py'


def load_n1():
    spec = importlib.util.spec_from_file_location('n1_choice_sequence', N1_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChunkedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.x = torch.randn(5, 37, 16)
        lengths = torch.tensor([37, 20, 1, 33, 9])
        self.mask = (torch.arange(37)[None, :] < lengths[:, None])[:, None, None, :]

    def pair(self, chunk):
        torch.manual_seed(11)
        full = ms.Block(16, 4, 32, .1, 0)
        chunked = ms.Block(16, 4, 32, .1, chunk)
        chunked.load_state_dict(full.state_dict())
        return full.eval(), chunked.eval()

    def test_chunked_equals_full_attention(self):
        for chunk in (1, 4, 8, 36, 37, 64):
            full, chunked = self.pair(chunk)
            with torch.no_grad():
                a, b = full(self.x, self.mask), chunked(self.x, self.mask)
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=0), chunk)

    def test_chunked_gradients_match(self):
        full, chunked = self.pair(8)
        xa, xb = self.x.clone().requires_grad_(), self.x.clone().requires_grad_()
        full(xa, self.mask).square().sum().backward()
        chunked(xb, self.mask).square().sum().backward()
        # float32 summation order differs between chunked and full softmax paths:
        # require agreement to ~1e-6 of each gradient's scale (observed ~1e-7).
        def close(a, b):
            return torch.allclose(a, b, atol=1e-6 * float(a.abs().max()) + 1e-7, rtol=0)
        self.assertTrue(close(xa.grad, xb.grad))
        for (n, pa), (_, pb) in zip(full.named_parameters(), chunked.named_parameters()):
            self.assertTrue(close(pa.grad, pb.grad), n)


class ForkParityTests(unittest.TestCase):
    def test_default_fork_reproduces_n1(self):
        n1 = load_n1()
        d = fixture()
        settings = dict(rank=4, hidden=8, d_model=8, heads=2, ff=16, layers=1, context='session', time_bins=4,
                        dropout=.1, batch_users=3, batch_tokens=96, seq_len=24, max_targets=8, pad_width=4,
                        pad_rows=2, choice_tokens=1)
        outputs = []
        for module in (n1, ms):
            c = module.Config(**settings)
            torch.manual_seed(5)
            model = module.SequenceModel(d['items'], c, module.weeks(d)).eval()
            torch.manual_seed(7)
            pred = module.predict(model, d)
            outputs.append(pred)
        self.assertEqual(outputs[0].tobytes(), outputs[1].tobytes())

    def test_new_field_defaults_are_off(self):
        self.assertEqual(ms.Config().attn_chunk, 0)
        self.assertNotIn('attn_chunk', {f.name for f in dataclasses.fields(load_n1().Config)})


SMALL = dict(rank=4, hidden=8, d_model=8, heads=2, ff=16, layers=1, context='session', time_bins=4,
             dropout=.1, batch_users=3, batch_tokens=96, seq_len=24, max_targets=8, pad_width=4,
             pad_rows=2, choice_tokens=1)


class MlpHeadTests(unittest.TestCase):
    def test_zero_initialized_head_adds_exactly_nothing(self):
        d = fixture()
        torch.manual_seed(5)
        base = ms.SequenceModel(d['items'], ms.Config(**SMALL), ms.weeks(d)).eval()
        torch.manual_seed(5)
        head = ms.SequenceModel(d['items'], ms.Config(**SMALL, head_mlp=16), ms.weeks(d)).eval()
        # The head's parameters are created last, so the shared ones initialize identically.
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, head.state_dict()[name]), name)
        self.assertEqual(ms.predict(base, d).tobytes(), ms.predict(head, d).tobytes())

    def test_head_trains(self):
        d = fixture()
        c = ms.Config(**SMALL, head_mlp=16, lr=.01)
        model, records, _ = ms.fit(d, c, 2, Path(self.tmp) / 'head')
        self.assertGreater(float(model.head_net[-1].weight.abs().sum()), 0)
        self.assertTrue(all(r['train_rmse'] == r['train_rmse'] for r in records))

    def test_head_requires_session_context(self):
        args = ms.parser().parse_args(['--data', 'x', '--output', str(Path(self.tmp) / 'o'),
                                       '--head-mlp', '8', '--context', 'none', '--time-bins', '0'])
        with self.assertRaises(ValueError):
            ms.run(args)

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


class ScheduleTests(unittest.TestCase):
    def test_lr_factor_shape(self):
        c = ms.Config(warmup_steps=10, cosine_epochs=4, lr_floor=.1)
        self.assertAlmostEqual(ms.lr_factor(c, 0, 0.), .1)
        self.assertAlmostEqual(ms.lr_factor(c, 9, 0.), 1.)
        self.assertAlmostEqual(ms.lr_factor(c, 100, 2.), .1 + .9 * .5)
        self.assertAlmostEqual(ms.lr_factor(c, 100, 4.), .1)
        self.assertAlmostEqual(ms.lr_factor(c, 100, 9.), .1)
        values = [ms.lr_factor(c, 100, e / 10) for e in range(41)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertEqual(ms.lr_factor(ms.Config(), 0, 0.), 1.)

    def test_scheduled_fit_is_prefix_consistent(self):
        """With a fixed horizon, a 1-epoch refit equals epoch 1 of a 2-epoch run."""
        import tempfile
        d = fixture()
        c = ms.Config(**SMALL, warmup_steps=3, cosine_epochs=2)
        with tempfile.TemporaryDirectory() as tmp:
            one, _, _ = ms.fit(d, c, 1, Path(tmp) / 'one')
            two, records, _ = ms.fit(d, c, 2, Path(tmp) / 'two', save_epochs=True)
            self.assertEqual(len(records), 2)
            epoch1 = ms.np.fromfile(str(Path(tmp) / 'two') + '.e1.f32', dtype=ms.np.float32)
            plain, _, _ = ms.fit(d, ms.Config(**SMALL), 1, Path(tmp) / 'plain')
        self.assertEqual(ms.predict(one, d).tobytes(), epoch1.tobytes())
        # The schedule is actually applied: it changes the trained model.
        self.assertNotEqual(ms.predict(one, d).tobytes(), ms.predict(plain, d).tobytes())

    def test_new_schedule_and_head_defaults_are_off(self):
        c = ms.Config()
        self.assertEqual((c.head_mlp, c.warmup_steps, c.cosine_epochs), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
