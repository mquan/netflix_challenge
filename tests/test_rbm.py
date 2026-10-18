"""Visible-context RBM contracts on a small dated dataset."""
import pathlib
import struct
import tempfile
import unittest

from test_joint import dataset, run


class RbmTests(unittest.TestCase):
    def test_context_reload_and_probe_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            for name, changed in (('a', None), ('b', 0)):
                dataset(p / f'{name}.bin', change_fold=changed)
                run('rbm', '--data', p / f'{name}.bin', '--model', p / f'{name}.model',
                    '--predictions', p / f'{name}.f32', '--hidden', 4,
                    '--iterations', 2, '--fixed-epoch', '--batch', 20,
                    '--threads', 2, '--visible-context', 7)
            a = (p / 'a.model').read_bytes()
            b = (p / 'b.model').read_bytes()
            self.assertEqual(a[16:], b[16:])
            self.assertEqual((p / 'a.f32').read_bytes(), (p / 'b.f32').read_bytes())
            self.assertEqual(struct.unpack_from('<Q', a)[0], 0x4E4650524253324D)
            self.assertEqual(struct.unpack_from('<I', a, 20)[0], 2)
            run('eval-rbm', '--data', p / 'a.bin', '--model', p / 'a.model',
                '--predictions', p / 'reload.f32')
            self.assertEqual((p / 'a.f32').read_bytes(), (p / 'reload.f32').read_bytes())
            run('eval-rbm', '--data', p / 'b.bin', '--model', p / 'a.model', ok=False)

    def test_legacy_format_remains_loadable(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            dataset(p / 'data.bin')
            run('rbm', '--data', p / 'data.bin', '--model', p / 'old.model',
                '--predictions', p / 'old.f32', '--hidden', 4,
                '--iterations', 1, '--batch', 20, '--threads', 2)
            self.assertEqual(struct.unpack_from('<Q', (p / 'old.model').read_bytes())[0],
                             0x4E4650524253314D)
            run('eval-rbm', '--data', p / 'data.bin', '--model', p / 'old.model',
                '--predictions', p / 'reload.f32')
            self.assertEqual((p / 'old.f32').read_bytes(), (p / 'reload.f32').read_bytes())

    def test_main_branch_conditional_format_remains_loadable(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            dataset(p / 'data.bin')
            run('rbm', '--data', p / 'data.bin', '--model', p / 'conditional.model',
                '--predictions', p / 'conditional.f32', '--hidden', 4,
                '--iterations', 1, '--batch', 20, '--threads', 2,
                '--conditional-biases')
            run('eval-rbm', '--data', p / 'data.bin', '--model', p / 'conditional.model',
                '--predictions', p / 'reloaded.f32')
            self.assertEqual((p / 'conditional.f32').read_bytes(),
                             (p / 'reloaded.f32').read_bytes())


if __name__ == '__main__':
    unittest.main()
