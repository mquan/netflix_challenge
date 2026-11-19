"""Reject altered family averages and wrong recovered group populations."""
import unittest
from unittest.mock import patch
import numpy as np
import verify_upstream_first85_completion as v


class CompletionAuditTest(unittest.TestCase):
    def test_fixed_average_rejects_valid_but_substituted_final(self):
        tree=np.array([1.05,3.25,4.95],dtype='<f4');nn=np.array([4.95,2.75,1.05],dtype='<f4')
        final=((tree.astype(np.float64)+nn.astype(np.float64))/2).astype('<f4')
        self.assertTrue(v.average_contract(tree,nn,final))
        with self.assertRaisesRegex(ValueError,'exact fixed family average'):
            v.average_contract(tree,nn,final+.01)
        with self.assertRaisesRegex(ValueError,'Invalid clipped'):
            v.average_contract(tree.astype(np.float64),nn,final)

    def test_both_original_source_groups_must_pass_strict_target(self):
        labels=np.array([[1,0],[2,0],[3,1],[4,1]],dtype=np.int16)
        # Use an exactly representable boundary to distinguish '<' from '<='.
        with patch.object(v,'TARGET',1.0):
            predictions=labels[:,0].astype(np.float64)+1.0
            groups=v.score_groups(predictions,labels,expected_counts=(2,2))
            self.assertTrue(all(g['rmse']==1.0 for g in groups.values()))
            self.assertFalse(any(g['below_target'] for g in groups.values()))
        predictions=np.array([1.1,2.1,3.1,4.1])
        groups=v.score_groups(predictions,labels,expected_counts=(2,2))
        self.assertTrue(all(g['below_target'] for g in groups.values()))
        for group in groups.values():self.assertAlmostEqual(group['rmse'],.1)
        with self.assertRaisesRegex(ValueError,'required populations'):
            v.score_groups(predictions,labels,expected_counts=(1,3))
        labels[0,1]=2
        with self.assertRaisesRegex(ValueError,'label shape/domain'):
            v.score_groups(predictions,labels,expected_counts=(2,2))


if __name__=='__main__':unittest.main()
