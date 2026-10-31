import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from production_neural import digest
from production_neural_audit import FIELDS, audit

torch.set_num_threads(1)


class AuditTests(unittest.TestCase):
    def fixture(self, root):
        (root/'source').mkdir()
        source = root/'source'/'trainer.py'
        source.write_text('fixed source')
        expected = {k:'bound' for k in FIELDS}
        expected.update(fixed_epochs=2, sources={'trainer.py':digest(source)},
            config={'device':'cpu'}, threads=1, torch_version=str(torch.__version__), numpy_version=np.__version__)
        values = np.array([1.2, 2., 3.1, 4.2, 5., 2.1], '<f4')
        values.tofile(root/'model.f32')
        saved = dict(epoch=2, config=expected['config'], items=4, state={'weight':torch.ones(2)})
        torch.save(saved, root/'model.pt')
        report = dict(expected, state='complete', training_requested=True, metric=None,
            records=[dict(epoch=e, targets=10) for e in [1,2]], prediction_count=len(values),
            prediction_min=float(values.min()), prediction_max=float(values.max()),
            predictions_sha256=digest(root/'model.f32'), checkpoint_sha256=digest(root/'model.pt'))
        (root/'report.json').write_text(json.dumps(report))
        plan = dict(frozen={}, qualifying_count=len(values), items=4,
                    stages=[dict(output_dir=str(root), expected=expected)])
        (root/'plan.json').write_text(json.dumps(plan))
        return report, saved

    def test_complete_payload_and_checkpoint_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            result = audit(root/'plan.json', 0)
            self.assertEqual(result['state'], 'verified_complete')
            self.assertEqual(result['qualifying_count'], 6)
            self.assertIsNone(result['metric'])
            self.assertEqual(result['outputs'][str(root/'model.f32')], digest(root/'model.f32'))

    def test_incomplete_budget_and_query_scoring_are_rejected(self):
        for change in ['budget', 'query_score', 'config']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                report, _ = self.fixture(root)
                if change == 'budget': report['records'].pop()
                elif change == 'query_score': report['records'][0]['validation_rmse'] = .1
                else: report['config'] = {'device':'other'}
                (root/'report.json').write_text(json.dumps(report))
                with self.assertRaises(ValueError): audit(root/'plan.json', 0)
                self.assertFalse((root/'audit.json').exists())

    def test_corrupt_output_source_and_wrong_checkpoint_epoch_are_rejected(self):
        for change in ['prediction', 'source', 'checkpoint']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                report, saved = self.fixture(root)
                if change == 'prediction': (root/'model.f32').write_bytes(b'corrupt')
                elif change == 'source': (root/'source'/'trainer.py').write_text('changed')
                else:
                    saved['epoch'] = 1
                    torch.save(saved, root/'model.pt')
                    report['checkpoint_sha256'] = digest(root/'model.pt')
                    (root/'report.json').write_text(json.dumps(report))
                with self.assertRaises(ValueError): audit(root/'plan.json', 0)
                self.assertFalse((root/'audit.json').exists())


if __name__ == '__main__':
    unittest.main()
