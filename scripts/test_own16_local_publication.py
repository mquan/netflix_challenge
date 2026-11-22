"""Recorded metadata mutation checks; separate full source admission is required."""
import copy
from pathlib import Path
import unittest
import own16_local_publication as g

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    report=ROOT/g.REPORT
    latest=g.read(report/'independent-canonical-verification.json')
    parts={name:g.read(report/name) for name in g.METADATA}
    hashes={name:g.digest(report/name) for name in g.METADATA}
    return latest,parts,hashes


class MetadataTests(unittest.TestCase):
    def test_original_complete_metadata(self):g.validate_metadata(*fixture())

    def test_original_outcome_selected_primary_rejected(self):
        args=fixture();latest,parts,_=args
        for row in (latest,parts['plan.json'],parts['frozen.json'],parts['independent-recovered-audit.json']):row['primary']=g.CANDIDATES[1]
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_mixed_primary_group_vector_rejected(self):
        args=fixture();args[0]['groups']['source0']['rmse']=.84
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_incomplete_supports_equations_or_family_inventory_rejected(self):
        for field,value in [('simplex_supports_examined',65534),('ridge_equations',16)]:
            args=fixture();args[0][field]=value
            with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[1]['calibration.json']['families'].pop()
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_actual_KKT_precision_cannot_be_relaxed(self):
        for field,value in [('absolute_gap',1e-8),('minimum_reduced_gradient',-1e-8),('complementarity_error',1e-8)]:
            args=fixture();args[1]['independent-preaudit-verification.json']['actual_simplex_optimality'][field]=value
            with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_qualifying_outcomes_cannot_precede_preaudit(self):
        args=fixture();args[1]['calibration.json']['qualifying_labels_read']=True
        with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[1]['independent-preaudit-verification.json']['observed']=args[0]['observed']
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_apparent_known_fit_and_adaptive_audit_labels_remain_explicit(self):
        args=fixture();args[0]['known_score_is_nested_cross_validation']=True
        with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[0]['official_or_untouched_test']=True
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_changed_vector_or_missing_group_population_rejected(self):
        args=fixture();args[1]['frozen.json']['outputs'][g.CANDIDATES[0]]['sha256']='0'*64
        with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[0]['results'][0]['groups']['source0']['rows']-=1
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_no_target_claim_from_tiny_improvement(self):
        args=fixture();args[0]['both_groups_below_target']=True
        with self.assertRaises(ValueError):g.validate_metadata(*args)


if __name__=='__main__':unittest.main()
