"""Canonical arithmetic/selection fixtures; no production device acceptance."""
import json
import unittest
import numpy as np
import upstream_rbm400_neural_canonical_reference_v2 as canonical
import upstream_rbm400_neural_study_reference_v2 as study
import test_verify_upstream_rbm400_neural_study_reference_v2 as replay_fixtures


class CanonicalTests(unittest.TestCase):
    def test_float64_reclip_compensated_scoring_and_invalid_domains(self):
        values=np.array([1.05,4.95,3.],dtype='<f4');labels=np.array([1.,5.,3.])
        clipped=np.clip(values.astype(np.float64),1.05,4.95)
        self.assertEqual(canonical.score(values,labels),float(np.sqrt(sum(float(x)**2 for x in clipped-labels)/3)))
        self.assertNotEqual(canonical.score(values,labels),float(np.sqrt(np.mean((values.astype(np.float64)-labels)**2))))
        with self.assertRaises(ValueError):canonical.score(values,np.array([0.,5.,3.]))

    def test_predeclared_complete_known_scores_and_scalar_tie_rule(self):
        self.assertEqual(canonical.choose({'scalar':.86,'latent':.859}),'latent')
        self.assertEqual(canonical.choose({'scalar':.86,'latent':.86}),'scalar')
        self.assertEqual(canonical.choose({'scalar':.86,'latent':.86-5e-13}),'scalar')
        self.assertEqual(canonical.choose({'scalar':.86,'latent':.86-2e-12}),'latent')
        for scores in ({'scalar':.86},{'scalar':.86,'latent':float('nan')},{'scalar':-.1,'latent':.86}):
            with self.assertRaises(ValueError):canonical.choose(scores)

    def test_complete_both_arm_canonical_and_known_only_selection_after_actual_file_replays(self):
        # Reuses full CPU preprocessing/forward/file replays. Source, device
        # and fitting gates remain explicitly mocked by the temporary fixture.
        helper=replay_fixtures.ReplayTests()
        with helper.complete_fixture(arms=study.ARMS) as (root,results):
            scalar=canonical.canonical('scalar',lambda **kw:None)
            self.assertEqual(scalar['state'],canonical.STATE);self.assertFalse(scalar['goal_complete'])
            with self.assertRaises(FileNotFoundError):canonical.select(lambda **kw:None)
            self.assertFalse((root/'experiments/upstream-own-rbm400-neural-selection-reference-v2').exists())
            latent=canonical.canonical('latent',lambda **kw:None)
            self.assertEqual(scalar['fixed_half_tree154_rmse'],latent['fixed_half_tree154_rmse'])
            selected=canonical.select(lambda **kw:None)
            self.assertEqual(selected['selected_arm'],'scalar');self.assertTrue(selected['both_arms_complete_before_selection'])
            self.assertFalse(selected['qualifying_labels_read']);self.assertFalse(selected['qualifying_labels_used_for_selection'])
            self.assertEqual(selected['actual_saved_model_counts'],{'scalar':10,'latent':10})
            with self.assertRaises(ValueError):canonical.select(lambda **kw:None)
            with self.assertRaises(ValueError):canonical.canonical('scalar',lambda **kw:None)


if __name__=='__main__':unittest.main()
