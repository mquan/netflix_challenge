import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_autorec_audit import audit, prepare
from production_rbm_audit import digest

torch.set_num_threads(1)


class AutoRecAuditTests(unittest.TestCase):
    def fixture(self, root, device='cpu'):
        data, source = root / 'data', root / 'source'
        data.write_bytes(b'data')
        source.write_bytes(b'source')
        output = root / 'trajectory'
        config = dict(data=str(data), predictions=str(output / 'model.f32'), hidden=2, layers=1,
                      epochs=3, fixed_epoch=True, save_epoch=[1], batch=4, lr=.0002, weight_decay=1e-5,
                      input_drop=.5, dropout=.5, patience=5, seed=2026, device=device, reload_checkpoint=None)
        outputs = [dict(epoch=1, prediction='model.e1.f32', checkpoint='model.e1.pt', replay='replay.e1.f32',
                        replay_log='replay.e1.log', column='snapshot', development_sha256='old-snapshot'),
                   dict(epoch=3, prediction='model.f32', checkpoint='model.pt', replay='replay.f32',
                        replay_log='replay.log', column='final', development_sha256='old-final')]
        stage = dict(group='fixture', output_dir=str(output), command=['autorec', '--fixed-epoch'], config=config, outputs=outputs)
        plan = dict(data=str(data), dataset=dict(items=3, query_count=2), stages=[stage],
                    source_sha256=digest(source), torch_version=str(torch.__version__), numpy_version=np.__version__,
                    device_verification=str(root / 'device-proof.json'),
                    frozen={str(data): digest(data), str(source): digest(source)})
        path = root / 'plan.json'
        path.write_text(json.dumps(plan))
        if device == 'mps':
            return path, plan, output
        prepare(path, 0)
        (output / 'train.log').write_text('autorec hidden=2 layers=1 batch=4 lr=0.0002 input_drop=0.5 dropout=0.5 wd=1e-05 device=cpu\n' +
            ''.join(f'epoch={i} train_rmse=1.0 query_labels=unknown seconds=1.0\n' for i in range(1, 4)))
        for item in outputs:
            payload = dict(epoch=item['epoch'], config=copy.deepcopy(config), items=3,
                           torch_threads=1, torch_version=str(torch.__version__), numpy_version=np.__version__,
                           source_sha256=digest(source), state={
                               '0.weight': torch.zeros(2, 6), '0.bias': torch.zeros(2),
                               '3.weight': torch.zeros(3, 2), '3.bias': torch.zeros(3)})
            torch.save(payload, output / item['checkpoint'])
            for name in (item['prediction'], item['replay']):
                np.array([2.0, 4.0], '<f4').tofile(output / name)
            (output / item['replay_log']).write_text(f"reload_epoch={item['epoch']} prediction_count=2\n")
        return path, plan, output

    def test_MPS_requires_matching_device_proof_and_binds_its_payloads(self):
        for damage in ('missing', 'cpu', 'source', 'checks', 'changed_payload'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path, plan, output = self.fixture(root, 'mps')
                payload = root / 'device-payload'
                payload.write_bytes(b'fixture device artifact')
                proof = dict(state='verified_autorec_device_parity', device='mps', host_threads=1,
                    source_sha256=plan['source_sha256'], torch_version=plan['torch_version'], numpy_version=plan['numpy_version'],
                    inputs={}, outputs={str(payload): digest(payload)},
                    checks=dict(original_final_and_snapshot_parity=True, final_and_snapshot_reload_parity=True,
                                query_label_independent_predictions_and_weights=True, distinct_snapshot_and_final=True))
                if damage == 'cpu':
                    proof['device'] = 'cpu'
                elif damage == 'source':
                    proof['source_sha256'] = 'other'
                elif damage == 'checks':
                    proof['checks']['distinct_snapshot_and_final'] = False
                if damage != 'missing':
                    Path(plan['device_verification']).write_text(json.dumps(proof))
                if damage == 'changed_payload':
                    prepare(path, 0)
                    payload.write_bytes(b'changed')
                    with self.assertRaises(ValueError):
                        audit(path, 0)
                else:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        prepare(path, 0)
                    self.assertFalse(output.exists())

    def test_complete_shared_snapshot_and_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path, plan, output = self.fixture(Path(temp))
            result = audit(path, 0)
            self.assertEqual(result['fixed_epochs'], 3)
            self.assertEqual([c['epoch'] for c in result['columns']], [1, 3])
            self.assertIsNone(result['metric'])
            with self.assertRaises(FileExistsError):
                audit(path, 0)

    def test_wrong_checkpoint_metadata_and_tensors_rejected(self):
        for damage in ('epoch', 'source', 'config', 'threads', 'version', 'shape', 'dtype', 'nonfinite', 'keys'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan, output = self.fixture(Path(temp))
                checkpoint = output / 'model.e1.pt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if damage == 'epoch':
                    saved['epoch'] = 3
                elif damage == 'source':
                    saved['source_sha256'] = 'different'
                elif damage == 'config':
                    saved['config']['dropout'] = .1
                elif damage == 'threads':
                    saved['torch_threads'] = 8
                elif damage == 'version':
                    saved['torch_version'] = 'other'
                elif damage == 'shape':
                    saved['state']['0.weight'] = torch.zeros(2, 7)
                elif damage == 'dtype':
                    saved['state']['0.weight'] = saved['state']['0.weight'].double()
                elif damage == 'nonfinite':
                    saved['state']['0.weight'][0, 0] = float('nan')
                else:
                    del saved['state']['0.bias']
                torch.save(saved, checkpoint)
                with self.assertRaises(ValueError):
                    audit(path, 0)
                self.assertFalse((output / 'audit.json').exists())

    def test_incomplete_budget_or_query_scoring_rejected(self):
        for damage in ('missing_epoch', 'query_score', 'header', 'reload_log', 'missing_snapshot'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan, output = self.fixture(Path(temp))
                log = output / 'train.log'
                if damage == 'missing_epoch':
                    log.write_text(log.read_text().replace('epoch=2', 'missing=2'))
                elif damage == 'query_score':
                    log.write_text(log.read_text().replace('query_labels=unknown', 'probe_rmse=3'))
                elif damage == 'header':
                    log.write_text(log.read_text().replace('hidden=2', 'hidden=3'))
                elif damage == 'reload_log':
                    (output / 'replay.e1.log').write_text('reload_epoch=3 prediction_count=2\n')
                else:
                    (output / 'model.e1.pt').unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    audit(path, 0)
                self.assertFalse((output / 'audit.json').exists())

    def test_invalid_predictions_reload_and_frozen_input_rejected(self):
        for damage in ('count', 'nonfinite', 'reload', 'frozen'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temp:
                path, plan, output = self.fixture(Path(temp))
                if damage == 'count':
                    np.array([2], '<f4').tofile(output / 'model.e1.f32')
                elif damage == 'nonfinite':
                    np.array([2, float('inf')], '<f4').tofile(output / 'model.e1.f32')
                elif damage == 'reload':
                    np.array([2, 3], '<f4').tofile(output / 'replay.e1.f32')
                else:
                    Path(plan['data']).write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    audit(path, 0)
                self.assertFalse((output / 'audit.json').exists())


if __name__ == '__main__':
    unittest.main()
