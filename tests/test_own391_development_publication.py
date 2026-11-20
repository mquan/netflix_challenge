import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_research_journey as journey


class Matched391Publication(unittest.TestCase):
    canonical=journey.EXPERIMENTS/'upstream-own391-paired-h128-mps/independent-canonical-verification.json'
    model=canonical.parent/'independent-verification.json'

    def values(self):
        current=json.loads(self.canonical.read_text());model=json.loads(self.model.read_text())
        sha=hashlib.sha256(self.canonical.read_bytes()).hexdigest()
        record=dict(independent_verification_sha256=sha,verified_development_report_sha256=sha,
            best_development_rmse=current['fixed_half_tree154_rmse'])
        return record,current,model

    def validate(self,record,current,model):
        journey.validate_own391_development(record,self.canonical,current,self.model,model)

    def test_all_original_pipelines_and_canonical_score_pass(self):
        self.validate(*self.values())

    def negative_control(self):
        path=journey.EXPERIMENTS/'upstream-own391-paired-h512-mps/independent-canonical-verification.json'
        model_path=path.parent/'independent-verification.json';current=json.loads(path.read_text())
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        record=dict(independent_verification_sha256=sha,verified_development_report_sha256=sha,
            best_development_rmse=current['fixed_half_tree154_rmse'])
        return record,path,current,model_path,json.loads(model_path.read_text())

    def test_complete_worse_control_can_be_recorded_without_promotion(self):
        journey.validate_own391_development(*self.negative_control(),require_improvement=False)

    def test_worse_control_cannot_be_promoted_as_development_best(self):
        with self.assertRaises(ValueError):journey.validate_own391_development(*self.negative_control())

    def test_negative_control_still_requires_full_model_replay(self):
        record,path,current,model_path,model=self.negative_control();model['records'][0]['all_normalized_input_bits_equal']=False
        with self.assertRaises(ValueError):journey.validate_own391_development(record,path,current,model_path,model,require_improvement=False)

    def test_unmatched_qualifying_input_claim_rejected(self):
        record,current,model=self.values();current['neural_inputs_matched_B_on_both_queries']=False
        with self.assertRaises(ValueError):self.validate(record,current,model)

    def test_missing_model_replay_rejected(self):
        record,current,model=self.values();model['records'][0]['actual_models_reopened']=2
        with self.assertRaises(ValueError):self.validate(record,current,model)

    def test_incomplete_preprocessing_rejected(self):
        record,current,model=self.values();model['records'][0]['all_normalized_input_bits_equal']=False
        with self.assertRaises(ValueError):self.validate(record,current,model)

    def test_qualifying_labels_in_development_rejected(self):
        record,current,model=self.values();current['qualifying_labels_read']=True
        with self.assertRaises(ValueError):self.validate(record,current,model)

    def test_development_cannot_claim_goal_completion(self):
        record,current,model=self.values();current['goal_complete']=True
        with self.assertRaises(ValueError):self.validate(record,current,model)

    def test_historical_tree_input_caveat_cannot_disappear(self):
        record,current,model=self.values();current['tree_control_retains_historical_A_B_realization_caveat']=False
        with self.assertRaises(ValueError):self.validate(record,current,model)

    def test_different_score_cannot_be_promoted(self):
        record,current,model=self.values();record['best_development_rmse']=.85
        with self.assertRaises(ValueError):self.validate(record,current,model)

    def assert_changed_bytes_rejected(self,relative):
        target=journey.ROOT/relative;original=Path.read_bytes
        def changed(path,*args,**kwargs):
            data=original(path,*args,**kwargs)
            return data+b'changed' if path==target else data
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaises(ValueError):self.validate(*self.values())

    def test_changed_actual_model_rejected(self):
        self.assert_changed_bytes_rejected('artifacts/upstream-own391-paired-h128-mps/nn/4/nn-13.pt')

    def test_changed_actual_scaler_rejected(self):
        self.assert_changed_bytes_rejected('artifacts/upstream-own391-paired-h128-mps/nn/0/scalers.npz')

    def test_changed_frozen_numerical_code_rejected(self):
        self.assert_changed_bytes_rejected('scripts/upstream_own391_paired_mps.py')

    def test_changed_canonical_prediction_rejected(self):
        self.assert_changed_bytes_rejected('artifacts/upstream-own391-paired-h128-mps/fixed-half-tree154.f32')


if __name__=='__main__':unittest.main()
