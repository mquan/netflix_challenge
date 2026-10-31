"""Exercise real preserved-binary Bayes handoffs and rejection of corrupt state."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from production_bayes_audit import audit, checkpoint, prepare
from production_rbm_audit import digest
from recover_bayes_recipes import posterior
from test_bayes import dataset
from test_native_fixed_epochs import query_labels

BIN = ROOT / 'artifacts/r8/bin/netflix-bayes-0c1f29d'


@unittest.skipUnless(BIN.is_file(), 'Historical binary unavailable')
class BayesAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = self.root / 'data'
        dataset(data, users=12, items=10, per_user=8)
        query_labels(data, 0)
        seed = self.root / 'metadata.bin'
        subprocess.run([str(BIN), 'bayes', '--data', str(data), '--model', str(seed),
                        '--factors', '3', '--samples', '2', '--burn-in', '1', '--threads', '1',
                        '--item-bin', '2', '--drift', '2'], check=True, capture_output=True)
        info, _ = posterior(seed)
        stages = []
        for index, (start, final) in enumerate(((0, 2), (2, 6))):
            out = self.root / f'stage{index}'
            cmd = [str(BIN), 'bayes', '--data', str(data), '--model', str(out / 'model.bin'),
                   '--predictions', str(out / 'model.f32'), '--samples', str(final),
                   '--burn-in', '1', '--threads', '1']
            if index:
                cmd += ['--init-model', str(self.root / 'stage0/model.bin')]
            else:
                cmd += ['--factors', '3', '--seed', '13', '--item-bin', '2', '--drift', '2']
            stages.append(dict(name=f'fixture{index}', command=cmd, output_dir=str(out),
                               dependency_index=0 if index else None, rank=3, threads=1,
                               start_sweep=start, final_sweep=final, burn_in=1, groups=info['groups']))
        self.plan = dict(data=str(data), frozen={str(data): digest(data), str(BIN): digest(BIN)},
                         dataset=dict(fingerprint=info['fingerprint'], query_count=info['prediction_count']),
                         stages=stages)
        self.path = self.root / 'plan.json'
        self.path.write_text(json.dumps(self.plan))

    def train(self, index):
        prepare(self.path, index)
        stage = self.plan['stages'][index]
        out = Path(stage['output_dir'])
        result = subprocess.run(stage['command'], capture_output=True, text=True, check=True)
        (out / 'train.log').write_text(result.stdout + result.stderr)
        reload_cmd = [str(BIN), 'bayes', '--data', self.plan['data'], '--init-model', str(out / 'model.bin'),
                      '--model', str(out / 'reload.bin'), '--predictions', str(out / 'reload.f32'),
                      '--samples', str(stage['final_sweep']), '--burn-in', '1', '--threads', '1']
        result = subprocess.run(reload_cmd, capture_output=True, text=True, check=True)
        (out / 'reload.log').write_text(result.stdout + result.stderr)
        return out

    def test_fresh_and_resumed_posterior_handoff(self):
        self.train(0)
        first = audit(self.path, 0)
        self.assertEqual((first['final_sweep'], first['kept']), (2, 1))
        self.train(1)
        result = audit(self.path, 1)
        self.assertEqual((result['final_sweep'], result['kept']), (6, 5))
        self.assertEqual(len(result['dependencies']), 2)
        self.assertIsNone(result['metric'])
        with self.assertRaises(FileExistsError):
            audit(self.path, 1)

    def test_resume_rejects_missing_or_mutated_parent(self):
        with self.assertRaises(FileNotFoundError):
            prepare(self.path, 1)
        self.assertFalse((self.root / 'stage1').exists())
        out = self.train(0)
        audit(self.path, 0)
        with (out / 'model.bin').open('ab') as stream:
            stream.write(b'x')
        with self.assertRaises(ValueError):
            prepare(self.path, 1)
        self.assertFalse((self.root / 'stage1').exists())

    def test_audit_rejects_budget_header_output_and_reload_changes(self):
        out = self.train(0)
        cases = [('train.log', lambda b: b.replace(b'sweep=2 ', b'sweep=3 ')),
                 ('train.log', lambda b: b.replace(b'threads=1 ', b'threads=4 ')),
                 ('train.log', lambda b: b.replace(b'kept=1 sweep=2', b'kept=2 sweep=2')),
                 ('model.f32', lambda b: b[:-4]),
                 ('reload.f32', lambda b: struct.pack('<f', 0) + b[4:]),
                 ('reload.log', lambda b: b + b'\nsweep=3 unexpected\n')]
        for name, change in cases:
            path = out / name
            original = path.read_bytes()
            path.write_bytes(change(original))
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit(self.path, 0)
            self.assertFalse((out / 'audit.json').exists())
            path.write_bytes(original)
        audit(self.path, 0)

    def test_checkpoint_rejects_wrong_dataset_window_and_nonfinite_weights(self):
        out = self.train(0)
        stage = self.plan['stages'][0]
        for key, value in [('final_sweep', 3), ('rank', 4), ('burn_in', 0)]:
            altered = dict(stage, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint(out / 'model.bin', self.plan['dataset'], altered)
        changed = dict(self.plan['dataset'], fingerprint=0)
        with self.assertRaises(ValueError):
            checkpoint(out / 'model.bin', changed, stage)
        path = out / 'model.bin'
        body = bytearray(path.read_bytes())
        # Header + feature descriptors + scalar block + first prior + vector size.
        struct.pack_into('<f', body, 28 + 8 * 12 + 24 + 16 + 8, float('nan'))
        path.write_bytes(body)
        with self.assertRaisesRegex(ValueError, 'nonfinite Bayes parameter'):
            checkpoint(path, self.plan['dataset'], stage)


if __name__ == '__main__':
    unittest.main()
