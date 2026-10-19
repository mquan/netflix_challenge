"""Production K1/user-kNN compatibility and zero-label query isolation."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from test_user_knn import write_dataset

K1 = Path(os.environ.get('K1_TEST_BIN', 'artifacts/production-replay/PR11-source/netflix-K1-rebuilt')).resolve()
UKNN = Path(os.environ.get('UKNN_TEST_BIN', 'artifacts/production-replay/PR11-source/netflix-uknn-rebuilt')).resolve()
ORIGINAL = Path('artifacts/r8/bin/netflix-uknn').resolve()


def run(binary, *args):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


class ProductionUserKnnTests(unittest.TestCase):
    def test_fixed_K1_and_user_knn_ignore_unknown_query_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {}
            for label in (0, 5):
                data, model = root / f'{label}.data', root / f'{label}.model'
                _, _, training = write_dataset(data)
                body = bytearray(data.read_bytes())
                users = struct.unpack_from('<I', body, 8)[0]
                queries = struct.unpack_from('<Q', body, 24)[0]
                offset = 56 + 8 * (users + 1) + 5 * training
                for row in range(queries):
                    body[offset + row * 12 + 8] = label
                data.write_bytes(body)
                run(K1, 'joint', '--data', data, '--model', model,
                    '--factors', 4, '--iterations', 3, '--threads', 1, '--fixed-epoch',
                    '--learning-rate', '.008', '--regularization', '.02', '--decay', '.97',
                    '--predictions', root / f'{label}.base')
                self.assertEqual(struct.unpack_from('<I', model.read_bytes(), 20)[0], 3)
                for variant, binary in [('rebuilt', UKNN), ('original', ORIGINAL)]:
                    prefix = root / f'{label}.{variant}'
                    run(binary, 'user-knn', '--data', data, '--joint-model', model,
                        '--predictions', str(prefix) + '.f32', '--support', str(prefix) + '.support',
                        '--support-count', str(prefix) + '.count', '--threads', 3)
                    outputs[label, variant] = [(Path(str(prefix) + suffix)).read_bytes()
                                               for suffix in ('.f32', '.support', '.count')]
                self.assertEqual(outputs[label, 'rebuilt'], outputs[label, 'original'])
            # Only the header fingerprint may depend on query labels.
            self.assertEqual((root / '0.model').read_bytes()[16:], (root / '5.model').read_bytes()[16:])
            self.assertEqual((root / '0.base').read_bytes(), (root / '5.base').read_bytes())
            self.assertEqual(outputs[0, 'rebuilt'], outputs[5, 'rebuilt'])


if __name__ == '__main__':
    unittest.main()
