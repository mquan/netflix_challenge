"""Actual complete synthetic fold reconstruction and adversarial receipts."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import upstream_full_context_catboost_fold as fitting
import upstream_full_context_catboost_artifacts as artifacts
import verify_upstream_full_context_catboost_fold as replay


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(); cls.root = Path(cls.temporary.name).resolve()
        cls.users = np.arange(40, dtype=np.int64); cls.groups = (cls.users % 10).astype(np.int8)
        cls.y = (cls.users % 5 + 1).astype(np.float64)
        cls.x = np.zeros((40, 713), dtype='<f4'); cls.x[:, 0] = cls.users; cls.x[:, 1] = cls.y
        cls.baselines = np.full((2, 40), 2.5, dtype=np.float64); cls.models = {}
        def save(model, seed, count):
            cls.models[str(seed)] = artifacts.save_model(cls.root/('%s.cbm'%seed), model,
                seed=seed, iterations=count, width=713, root=cls.root, guard=lambda **kw: None)
        cls.receipt = fitting.run(cls.x, cls.baselines, cls.y, cls.users, cls.groups, 0,
            save_model=save, save_population=lambda *a: None, release=lambda: None, guard=lambda **kw: None)
        cls.prediction = cls.receipt.pop('prediction')
        assert cls.models

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def check(self, *, x=None, b=None, receipt=None, models=None, prediction=None):
        return replay.verify(self.x.copy() if x is None else x,
            self.baselines if b is None else b, self.users, self.groups,
            self.receipt if receipt is None else receipt, self.models if models is None else models,
            self.prediction if prediction is None else prediction, lambda **kw: None)

    def test_every_actual_saved_positive_slot_and_all_held_bits_reconstructed(self):
        result = self.check()
        self.assertEqual(result['actual_saved_models_reopened'], len(self.models))
        self.assertEqual(result['seed_slots'], 3)
        self.assertTrue(result['every_held_prediction_bit_verified'])

    def test_model_digest_inventory_and_slot_tampering_rejected(self):
        with self.assertRaises(ValueError): self.check(models={})
        models = copy.deepcopy(self.models); key = next(iter(models)); models[key]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'digest'): self.check(models=models)
        models = copy.deepcopy(self.models); models[key]['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'slot'): self.check(models=models)

    def test_one_bit_prediction_or_raw_operator_or_baseline_mutation_rejected(self):
        p = self.prediction.copy(); p[0] = np.nextafter(p[0], np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, 'held prediction'): self.check(prediction=p)
        x = self.x.copy(); x[0, 4] = .01
        with self.assertRaisesRegex(ValueError, 'raw input'): self.check(x=x)
        b = self.baselines.copy(); b[1, 0] += .001
        with self.assertRaisesRegex(ValueError, 'raw input'): self.check(b=b)

    def test_zero_slots_use_baseline_without_model_and_invalid_selection_rejected(self):
        receipt = copy.deepcopy(self.receipt); receipt['chosen_iterations'] = [0, 0, 0]
        for proof in receipt['inner_selection']:
            proof.update(selected_iterations=0, best_validation_RMSE=100., zero_correction_validation_MSE=1.)
        expected = self.baselines[1, self.groups // 2 == 0].astype('<f4')
        result = self.check(receipt=receipt, models={}, prediction=expected)
        self.assertEqual(result['actual_saved_models_reopened'], 0)
        receipt['inner_selection'][0]['selected_iterations'] = 1
        with self.assertRaisesRegex(ValueError, 'selection'): self.check(receipt=receipt, models={}, prediction=expected)


if __name__ == '__main__': unittest.main()
