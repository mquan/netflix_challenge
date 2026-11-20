import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_research_journey as journey

FIXTURE=json.loads((Path(__file__).parent/'fixtures/own330_status.json').read_text())


class Own330Publication(unittest.TestCase):
    def reader(self,target=None,change=None):
        original=Path.read_text
        def read(path,*args,**kwargs):
            if path==journey.ROOT/'experiments/fresh-reproduction-status.json':return json.dumps(FIXTURE)
            text=original(path,*args,**kwargs)
            if path==target:
                data=json.loads(text);change(data);return json.dumps(data)
            return text
        return read

    def test_actual_complete_full_model_and_separate_score_proofs(self):
        with patch.object(Path,'read_text',self.reader()):result=journey.validate_fresh_status()
        self.assertEqual(result['best_development_rmse'],0.8589474951024617)
        self.assertFalse(result['goal_achieved'])

    def test_missing_actual_saved_model_replay_is_rejected(self):
        path=journey.ROOT/'experiments/upstream-own330-mps/independent-verification.json'
        with patch.object(Path,'read_text',self.reader(path,lambda d:d['records'][2].update(actual_models_reopened=2))):
            with self.assertRaises(ValueError):journey.validate_fresh_status()

    def test_incomplete_original_normalization_replay_is_rejected(self):
        path=journey.ROOT/'experiments/upstream-own330-mps/independent-verification.json'
        with patch.object(Path,'read_text',self.reader(path,lambda d:d['records'][1].update(all_original_normalized_input_bits_equal=False))):
            with self.assertRaises(ValueError):journey.validate_fresh_status()

    def test_smaller_than_original_model_is_rejected(self):
        path=journey.ROOT/'experiments/upstream-own330-mps/manifest.json'
        with patch.object(Path,'read_text',self.reader(path,lambda d:d['config'].update(nn_hidden=64))):
            with self.assertRaises(ValueError):journey.validate_fresh_status()


if __name__=='__main__':unittest.main()
