"""Native production checkpoint selection must not depend on query labels."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from test_joint import dataset

BIN = str(Path(os.environ.get('NETFLIX_TEST_BIN', 'build/netflix')).resolve())


def run(*args):
    result = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


def query_labels(path, rating):
    raw = bytearray(path.read_bytes())
    header = struct.unpack_from('<8sIIQQQHH4xd', raw)
    users, training, queries = header[1], header[3], header[4]
    offset = 56 + 8 * (users + 1) + 5 * training
    for i in range(queries):
        raw[offset + 12 * i + 8] = rating
    path.write_bytes(raw)


class NativeFixedEpochTests(unittest.TestCase):
    def check_family(self, family, epoch_offset):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            for label in (0, 5):
                prefix = p / str(label)
                data, graph = Path(str(prefix) + '.data'), Path(str(prefix) + '.graph')
                model, predictions = Path(str(prefix) + '.model'), Path(str(prefix) + '.f32')
                dataset(data, correlated_train=True)
                query_labels(data, label)
                run('graph', '--data', data, '--graph', graph, '--neighbors', 3, '--history-cap', 3)
                common = ['--data', data, '--graph', graph, '--model', model, '--predictions', predictions]
                if family == 'neighborhood':
                    baseline = Path(str(prefix) + '.base')
                    run('time-baseline', '--data', data, '--model', baseline, '--iterations', 2)
                    common += ['--time-model', baseline]
                    train = []
                else:
                    train = ['--factors', 4, '--patience', 1]
                log = run(family, *common, *train, '--iterations', 6, '--fixed-epoch')
                self.assertNotIn('early_stop', log)
                self.assertEqual(struct.unpack_from('<I', model.read_bytes(), epoch_offset)[0], 6)
                reload_path = Path(str(prefix) + '.reload')
                evaluate = common.copy()
                evaluate[evaluate.index('--predictions') + 1] = reload_path
                run('eval-' + family, *evaluate)
                self.assertEqual(predictions.read_bytes(), reload_path.read_bytes())
            # The dataset fingerprint includes query labels; parameters must not.
            self.assertEqual((p / '0.model').read_bytes()[16:], (p / '5.model').read_bytes()[16:])
            self.assertEqual((p / '0.f32').read_bytes(), (p / '5.f32').read_bytes())
            if family == 'integrated':
                self.assertEqual((p / '0.model.nbr').read_bytes(), (p / '5.model.nbr').read_bytes())

    def test_neighborhood_keeps_fixed_epoch_with_unknown_queries(self):
        self.check_family('neighborhood', 24)

    def test_integrated_keeps_fixed_epoch_with_unknown_queries(self):
        self.check_family('integrated', 20)


if __name__ == '__main__':
    unittest.main()
