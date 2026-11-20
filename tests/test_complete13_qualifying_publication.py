import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_research_journey as journey


class Complete13Publication(unittest.TestCase):
    path=journey.ROOT/'experiments/upstream-complete13-calibration/independent-canonical-verification.json'

    def latest(self):return json.loads(self.path.read_text())

    def test_complete_original_proof(self):
        self.assertEqual(journey.validate_complete13_qualifying(self.path,self.latest()),(True,True))

    def test_root_and_journey_share_current_generated_content(self):
        root=(journey.ROOT/'index.html').read_bytes();detail=(journey.ROOT/'research_journey.html').read_bytes()
        self.assertEqual(root,detail)
        self.assertIn(b'FRESH complete13 qualifying',root)
        self.assertNotIn(b'"strategy":"Public bank"',root)

    def test_incomplete_simplex_supports_rejected(self):
        value=self.latest();value['simplex_supports_examined']=127
        with self.assertRaises(ValueError):journey.validate_complete13_qualifying(self.path,value)

    def test_missing_actual_qualifying_model_rejected(self):
        value=self.latest();value['actual_own352_qualifying_models_reopened']=14
        with self.assertRaises(ValueError):journey.validate_complete13_qualifying(self.path,value)

    def test_non_oof_score_claim_rejected(self):
        value=self.latest();value['known_score_is_nested_cross_validation']=True
        with self.assertRaises(ValueError):journey.validate_complete13_qualifying(self.path,value)

    def test_control_cannot_replace_declared_primary(self):
        value=self.latest();value['primary']='complete13-ridge10'
        with self.assertRaises(ValueError):journey.validate_complete13_qualifying(self.path,value)

    def assert_changed_bytes_rejected(self,relative):
        target=journey.ROOT/relative;original=Path.read_bytes;value=self.latest()
        def changed(path,*args,**kwargs):
            content=original(path,*args,**kwargs)
            return content+b'changed' if path==target else content
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaises(ValueError):journey.validate_complete13_qualifying(self.path,value)

    def test_changed_frozen_primary_rejected(self):
        self.assert_changed_bytes_rejected('artifacts/upstream-complete13-calibration/complete13-simplex.f32')

    def test_changed_frozen_control_rejected(self):
        self.assert_changed_bytes_rejected('artifacts/upstream-complete13-calibration/complete13-ridge10.f32')

    def test_changed_actual_saved_model_rejected(self):
        self.assert_changed_bytes_rejected('artifacts/upstream-own352-temporal-mps/nn/4/nn-13.pt')


if __name__=='__main__':unittest.main()
