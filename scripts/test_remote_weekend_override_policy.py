"""Reject numerical or provenance changes hidden behind a resource override."""
import copy
import json
import unittest
import upstream_remote_weekend_override_policy as policy


class PolicyVariant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.original=policy.bundle()

    def test_actual_pinned_variant(self):
        v=policy.validate(*self.original)
        self.assertFalse(v['numerical_returns_admitted'])
        self.assertEqual(sum(j['named_pairs'] for j in v['jobs']),67)
        self.assertTrue(v['coordinator_local_guards_unchanged'])

    def test_recipe_data_and_deadline_changes(self):
        for kind in ('epochs','seed','pairs','data','deadline'):
            with self.subTest(kind=kind):
                args=list(self.original);plan=json.loads(args[2])
                if kind=='epochs':plan['jobs'][0]['epochs_per_phase']=14
                elif kind=='seed':plan['jobs'][0]['environment']['SEED']='7'
                elif kind=='pairs':plan['jobs'][0]['published_columns'].pop()
                elif kind=='data':plan['row_map_sha256']='0'*64
                else:plan['training_deadline_utc']='2026-11-24T14:00:00Z'
                args[2]=json.dumps(plan).encode()
                with self.assertRaises(ValueError):policy.validate(*args)

    def test_additional_controller_edit(self):
        args=list(self.original);args[3]+=b'\n# additional change\n'
        with self.assertRaises(ValueError):policy.validate(*args)

    def test_missing_or_outcome_using_preparation(self):
        for key in ('missing','qualifying_labels_read','qualifying_targets_zero','plan_sha256'):
            with self.subTest(key=key):
                args=list(self.original);args[4]=copy.deepcopy(args[4]);job=next(iter(args[4]))
                if key=='missing':del args[4][job]
                else:
                    p=json.loads(args[4][job]);p[key]={'qualifying_labels_read':True,
                        'qualifying_targets_zero':False,'plan_sha256':'0'*64}[key];args[4][job]=json.dumps(p).encode()
                with self.assertRaises(ValueError):policy.validate(*args)


if __name__=='__main__':unittest.main()
