"""Readonly recovery resource and source/provenance regression checks."""
import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import upstream_rbm400_context_source_alignment_execution as module
import upstream_rbm400_context_execution_reference_v2 as original
import verify_upstream_rbm400_context_source_alignment as verifier
import verify_upstream_rbm400_context_data_reference_v2 as old_verifier


class ExecutionTests(unittest.TestCase):
    def test_original_resource_guards_and_numerical_helpers_remain_identical(self):
        for old, new, names in [(original, module, ('cap','safe','ready','continuation_ready','same_child','stop_child','WorkerGuard')),
                (old_verifier,verifier,('known_score','replay_query','coefficients'))]:
            for name in names:
                self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(old,name)))),
                                 ast.dump(ast.parse(inspect.getsource(getattr(new,name)))))
        self.assertEqual(module.PHASES,('verify',))
        self.assertEqual(module.cap('verify'),3*2**30)
        self.assertEqual(module.FLOOR,int(18.75*2**30))
        self.assertEqual(module.TOTAL_LIMIT,24*2**30)
        with self.assertRaises(ValueError):module.cap('assemble')

    def test_original_source_contract_checked_before_new_snapshot(self):
        with patch.object(original,'check_sources',side_effect=ValueError('original source changed')) as checked:
            with self.assertRaisesRegex(ValueError,'original source changed'):module.source_files()
        checked.assert_called_once()

    def test_proof_must_bind_retained_original_manifest_instead_of_new_output_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            (old/'manifest.json').write_text('original unchanged assembly')
            (new/'manifest.json').write_text('wrong new manifest')
            proof=dict(state=verifier.STATE,qualifying_labels_read=False,neural_models_fitted=0,
                       manifest_sha256=module.inputs.sha(old/'manifest.json'))
            path=new/'independent-verification.json';path.write_text(json.dumps(proof))
            value=dict(state=module.RESULTS['verify'][1],execution_plan_sha256='p',qualifying_labels_read=False,
                       neural_models_fitted=0,verification_sha256=module.inputs.sha(path),
                       all_original_and_both_arm_operator_statistic_and_ridge_bits_verified=True)
            with patch.object(module,'STUDY',new),patch.object(module,'ORIGINAL',old):
                module.validate_result('verify',value,'p')
                (old/'manifest.json').write_text('changed retained assembly')
                with self.assertRaisesRegex(ValueError,'assembled manifest'):module.validate_result('verify',value,'p')


if __name__=='__main__':unittest.main()
