from pathlib import Path
import tempfile
import unittest
import numpy as np
import upstream_extension_bank as bank
import upstream_extension_run as run
import upstream_extension_production as production


class ExtensionRunChecks(unittest.TestCase):
    def test_schema_preserves_parent_columns_and_intercept(self):
        design=np.array([[2.,5.,1.]])
        features=np.array([[.25,-.5]],dtype=np.float32)
        added=np.array([[-1.,7.]],dtype=np.float32)
        z,f=bank.blocks(design,features,added)
        np.testing.assert_array_equal(z,[[2.,5.,0.,6.,1.]])
        np.testing.assert_array_equal(f[:,:2],features)
        np.testing.assert_allclose(f[:,2:],[[-3.6,2.4]],rtol=0,atol=3e-7)
        self.assertEqual(f.dtype,np.float32)

    def test_audit_gate_waits_then_rejects_partial_or_changed_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(run.first_gate(root),'waiting')
            vector=root/'qualifying-final.f32'
            np.full(2817131,3.5,dtype='<f4').tofile(vector)
            run.u.write(root/'recovered-audit/report.json',dict(state='complete'))
            run.u.write(root/'report.json',dict(state='complete',qualifying_labels_read=False,
                qualifying_rows=2817131,frozen_prediction_sha256=run.u.sha(vector)))
            record=dict(prediction_sha256=run.u.sha(vector),audit_report_sha256=run.u.sha(root/'recovered-audit/report.json'),
                groups=dict(quiz=dict(rows=1408789,rmse=.858),test=dict(rows=1408342,rmse=.856)),
                both_groups_below_target=False)
            run.u.write(root/'independent-qualifying-verification.json',record)
            self.assertEqual(run.first_gate(root),'continue_fixed_extension')
            record['groups']['quiz']['rmse']=.85725
            run.u.write(root/'independent-qualifying-verification.json',record)
            self.assertEqual(run.first_gate(root),'continue_fixed_extension')
            record['groups']['quiz']['rmse']=.856
            record['both_groups_below_target']=True
            run.u.write(root/'independent-qualifying-verification.json',record)
            self.assertEqual(run.first_gate(root),'target_already_verified')
            with vector.open('r+b') as stream:stream.write(b'bad!')
            with self.assertRaisesRegex(ValueError,'vector changed'):run.first_gate(root)

    def test_false_completion_boolean_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);vector=root/'qualifying-final.f32'
            np.full(2817131,3.5,dtype='<f4').tofile(vector)
            run.u.write(root/'recovered-audit/report.json',dict(state='complete'))
            run.u.write(root/'report.json',dict(state='complete',qualifying_labels_read=False,
                qualifying_rows=2817131,frozen_prediction_sha256=run.u.sha(vector)))
            run.u.write(root/'independent-qualifying-verification.json',dict(prediction_sha256=run.u.sha(vector),
                audit_report_sha256=run.u.sha(root/'recovered-audit/report.json'),both_groups_below_target=True,
                groups=dict(quiz=dict(rows=1408789,rmse=.858),test=dict(rows=1408342,rmse=.856))))
            with self.assertRaisesRegex(ValueError,'boolean disagrees'):run.first_gate(root)

    def test_core_and_extension_are_bound_in_production_hashes(self):
        hashes=production.code_hashes()
        for name in ['scripts/upstream_production.py','scripts/upstream_extension_bank.py',
                     'scripts/upstream_extension_production.py','scripts/upstream_source_extension.py']:
            self.assertIn(name,hashes)


if __name__=='__main__':unittest.main()
