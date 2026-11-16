import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_qualifying import frozen_predictions,scores,sha


class AuditTests(unittest.TestCase):
    def test_refuses_unfinished_or_modified_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'qualifying-final.f32'
            np.array([2.,3.],dtype='<f4').tofile(p)
            r=dict(state='training',qualifying_labels_read=False,qualifying_rows=2,frozen_prediction_sha256=sha(p))
            (root/'report.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'complete'): frozen_predictions(root)
            r['state']='complete';(root/'report.json').write_text(json.dumps(r))
            np.testing.assert_array_equal(frozen_predictions(root)[0],[2,3])
            np.array([2.,4.],dtype='<f4').tofile(p)
            with self.assertRaisesRegex(ValueError,'hash'): frozen_predictions(root)

    def test_scores_use_the_declared_split_and_exact_target(self):
        p=np.array([1.,2.,3.,4.]);r=np.array([1,3,3,5]);t=np.array([0,0,1,1])
        out=scores(p,r,t)
        self.assertAlmostEqual(out['quiz']['rmse'],np.sqrt(.5))
        self.assertAlmostEqual(out['test']['rmse'],np.sqrt(.5))
        self.assertTrue(out['target_reached'])
        with self.assertRaises(ValueError): scores(p,r,np.array([0,0,2,1]))


if __name__=='__main__': unittest.main()
