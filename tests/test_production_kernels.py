"""Unknown-query contracts for recovered GP and fixed-budget local models."""
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from test_joint import dataset, run

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'artifacts/production-replay/PR12-source'


class ProductionKernelTests(unittest.TestCase):
    def test_gp_analytic_solves(self):
        for binary in ('moonshot_gp_initial', 'moonshot_gp_persistent'):
            subprocess.run([str(BIN / binary), '--self-test'], check=True, capture_output=True)

    def test_unknown_query_labels_leave_all_retained_outputs_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in (0, 5):
                data, model = root / f'{label}.data', root / f'{label}.model'
                dataset(data, correlated_train=True)
                body = bytearray(data.read_bytes())
                header = struct.unpack_from('<8sIIQQQHH4xd', body)
                offset = 56 + 8 * (header[1] + 1) + 5 * header[3]
                for row in range(header[4]):
                    body[offset + row * 12 + 8] = label
                data.write_bytes(body)
                run('joint', '--data', data, '--model', model, '--factors', 4,
                    '--iterations', 1, '--fixed-epoch')
                recipes = [('gp', 'moonshot_gp_initial', ['--neighbors', '2']),
                           ('persistent', 'moonshot_gp_persistent', ['--neighbors', '2', '--time-scale', '30', '--persistent', '.5']),
                           ('global', 'moonshot_local', ['--groups', '1', '--rank', '4', '--epochs', '6']),
                           ('local', 'moonshot_local', ['--groups', '3', '--rank', '2', '--epochs', '6'])]
                for name, binary, options in recipes:
                    subprocess.run([str(BIN / binary), '--data', str(data), '--model', str(model),
                                    '--output', str(root / f'{label}-{name}'), *options],
                                   check=True, capture_output=True)
            for name, suffixes in [('gp', ['.0.f32', '.1.f32', '.2.f32', '.mask.f32']),
                                   ('persistent', ['.0.f32', '.mask.f32']),
                                   ('global', ['.epoch6.f32']), ('local', ['.epoch6.f32'])]:
                for suffix in suffixes:
                    self.assertEqual((root / ('0-' + name + suffix)).read_bytes(),
                                     (root / ('5-' + name + suffix)).read_bytes())
            for name in ('global', 'local'):
                self.assertNotEqual((root / f'0-{name}.epoch1.f32').read_bytes(),
                                    (root / f'0-{name}.epoch6.f32').read_bytes())


if __name__ == '__main__':
    unittest.main()
