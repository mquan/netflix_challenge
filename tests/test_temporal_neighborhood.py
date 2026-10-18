"""Contracts for the jointly fitted temporal neighborhood model."""
import pathlib
import struct
import tempfile
import unittest

from test_joint import dataset, run


class TemporalNeighborhoodTests(unittest.TestCase):
    def test_fixed_epoch_excludes_probe_labels_and_reloads(self):
        with tempfile.TemporaryDirectory() as temp:
            p = pathlib.Path(temp)
            for name in ('a', 'b'):
                dataset(p / (name + '.bin'), change_fold=0 if name == 'b' else None,
                        correlated_train=True)
                run('graph', '--data', p / (name + '.bin'),
                    '--graph', p / (name + '.graph'), '--neighbors', 2,
                    '--history-cap', 3)
                run('temporal-neighborhood', '--data', p / (name + '.bin'),
                    '--graph', p / (name + '.graph'), '--model', p / (name + '.model'),
                    '--predictions', p / (name + '.f32'), '--iterations', 3,
                    '--fixed-epoch', '--decay-rate', .001)
            # The first 28 bytes are magic, data/graph fingerprints and epoch.
            self.assertEqual((p / 'a.model').read_bytes()[28:],
                             (p / 'b.model').read_bytes()[28:])
            self.assertEqual((p / 'a.f32').read_bytes(), (p / 'b.f32').read_bytes())
            decay = struct.unpack('<100f', (p / 'a.model').read_bytes()[-400:])
            self.assertTrue(any(value > 0 for value in decay))
            run('eval-temporal-neighborhood', '--data', p / 'a.bin',
                '--graph', p / 'a.graph', '--model', p / 'a.model',
                '--predictions', p / 'reload.f32')
            self.assertEqual((p / 'a.f32').read_bytes(), (p / 'reload.f32').read_bytes())
            run('eval-temporal-neighborhood', '--data', p / 'b.bin',
                '--graph', p / 'b.graph', '--model', p / 'a.model',
                '--predictions', p / 'bad.f32', ok=False)


if __name__ == '__main__':
    unittest.main()
