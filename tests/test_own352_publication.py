import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_research_journey as journey


class Own352Publication(unittest.TestCase):
    canonical = journey.ROOT/'experiments/upstream-own352-bounded-replay/independent-canonical-verification.json'
    model = canonical.parent/'independent-verification.json'

    def validate(self):
        status=json.loads((journey.EXPERIMENTS/'fresh-reproduction-status.json').read_text())
        journey.validate_own352_development(status,self.canonical,json.loads(self.canonical.read_text()),
                                           self.model,json.loads(self.model.read_text()))

    def test_complete_actual_original_models_and_bounded_replay(self):
        self.validate()

    def changed_reader(self,target,change):
        original=Path.read_text
        def read(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path==target:
                data=json.loads(text);change(data);return json.dumps(data)
            return text
        return read

    def test_inexact_original_statistics_rejected(self):
        with patch.object(Path,'read_text',self.changed_reader(self.model,lambda d:d.update(complete_original_statistics_bitwise_equal=False))):
            with self.assertRaises(ValueError):self.validate()

    def test_missing_actual_fold_model_replay_rejected(self):
        with patch.object(Path,'read_text',self.changed_reader(self.model,lambda d:d['records'][2].update(actual_models_reopened=2))):
            with self.assertRaises(ValueError):self.validate()

    def test_changed_model_training_during_recovery_rejected(self):
        target=self.canonical.parent/'plan.json'
        with patch.object(Path,'read_text',self.changed_reader(target,lambda d:d.update(original_model_fitting_restarted=True))):
            with self.assertRaises(ValueError):self.validate()


if __name__=='__main__':unittest.main()
