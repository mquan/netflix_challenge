from pathlib import Path
import tempfile
import unittest
import numpy as np
import upstream_extension_bank as bank
import upstream_extension_run_current as run
import upstream_extension_production_current as production


class ExtensionRunChecks(unittest.TestCase):
    def test_current_best_reference_rejects_stale_score_or_different_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);prepared=base/'prepared';prepared.mkdir()
            for name in ['labels.npy','users.npy','groups.npy']:
                np.save(prepared/name,np.array([1,2,3]))
            roots=[]
            for name,score in [('context',.86006),('tree',.86002)]:
                root=base/'experiments'/name;root.mkdir(parents=True);roots.append(root)
                vector=base/'artifacts'/name/'fixed-half-tree-nn.f32';vector.parent.mkdir(parents=True)
                np.array([3.,3.5,4.],dtype='<f4').tofile(vector)
                row=dict(arm='fixed-half-tree-nn',rmse=score,prediction_sha256=run.u.sha(vector))
                run.u.write(root/'evaluation.json',dict(records=[row]))
                run.u.write(root/'manifest.json',dict(qualifying_labels_read=False,
                    prepared_sha256={p.name:run.u.sha(p) for p in prepared.iterdir()}))
                run.u.write(root/'independent-verification.json',dict(state='complete',fixed_average_bitwise_equal=True,
                    evaluation_sha256=run.u.sha(root/'evaluation.json'),records=[dict(arm=row['arm'],
                        independent_rmse=score,prediction_sha256=row['prediction_sha256'])]))
            # A candidate better than the old .86040 baseline but worse than
            # today's .86002 reference must not pass production admission.
            reference=run.development_reference(roots,prepared)
            self.assertEqual(reference['rmse'],.86002)
            self.assertFalse(.8601<reference['rmse'])
            self.assertTrue(.8600<reference['rmse'])
            np.save(prepared/'users.npy',np.array([3,2,1]))
            with self.assertRaisesRegex(ValueError,'row identity'):run.development_reference(roots,prepared)
            np.save(prepared/'users.npy',np.array([1,2,3]))
            row['rmse']=.85;run.u.write(roots[1]/'evaluation.json',dict(records=[row]))
            with self.assertRaisesRegex(ValueError,'verification changed'):run.development_reference(roots,prepared)

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
                     'scripts/upstream_extension_production.py','scripts/upstream_source_extension.py',
                     'scripts/upstream_extension_run_current.py','scripts/upstream_extension_production_current.py']:
            self.assertIn(name,hashes)


if __name__=='__main__':unittest.main()
