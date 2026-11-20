"""Fail closed when publishing the new unequal fixed-weight best."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_research_journey as journey

FIXTURE=json.loads((Path(__file__).parent/'fixtures/weighted_native_pool_status.json').read_text())

class WeightedPoolStatusTests(unittest.TestCase):
    def test_actual_complete_weighted_result_and_all_seven_family_replays_validate(self):
        with patch.object(Path,'read_text',self.changed_read(None,lambda d:None)):
            r=journey.validate_fresh_status()
        self.assertEqual(r['best_development_rmse'],.8591267885072646)
        self.assertEqual(len(r['development_required_replay_paths']),7)
        self.assertFalse(r['goal_achieved'])
    def changed_read(self,target,change):
        original=Path.read_text
        def read(path,*args,**kwargs):
            if path==journey.ROOT/'experiments/fresh-reproduction-status.json':return json.dumps(FIXTURE)
            text=original(path,*args,**kwargs)
            if path==target:
                d=json.loads(text);change(d);return json.dumps(d)
            return text
        return read
    def test_wrong_declared_weights_rejected(self):
        target=journey.ROOT/'experiments/upstream-native-nn-pool6-tree31/verification.json'
        with patch.object(Path,'read_text',self.changed_read(target,lambda d:d.update(weights=dict(tree=.5,nn_pool=.5)))):
            with self.assertRaises(ValueError):journey.validate_fresh_status()
    def test_missing_actual_origin_replay_proof_rejected(self):
        target=journey.ROOT/'experiments/upstream-native-nn-pool6/independent-canonical-verification.json'
        with patch.object(Path,'read_text',self.changed_read(target,lambda d:d.update(all_original_source_models_and_complete_replay_receipts_rehashed=False))):
            with self.assertRaises(ValueError):journey.validate_fresh_status()

if __name__=='__main__':unittest.main()
