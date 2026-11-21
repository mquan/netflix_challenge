"""Private semantic/source fixtures, never actual numerical audit admission."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import own15_publication as g


def hashed(value):return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    base=dict(family_ids=['family%d'%i for i in range(14)],families=[dict(id='family%d'%i) for i in range(14)])
    source={'scripts/known_only.py':hashed('source')};full_source=dict(source,**{'proof.json':hashed('proof')})
    hashes={name:hashed(name) for name in g.METADATA};pred={name:hashed(name) for name in g.KNOWN+g.CANDIDATES}
    p=dict(primary=g.CANDIDATES[0],families=15,candidates=list(g.CANDIDATES),ridge_lambda=10.,source_sha256=source,
        qualifying_labels_read=False,known_score_is_nested_cross_validation=False,created='2026-11-21T20:00:00+00:00',verifier_sha256=hashed('verifier'))
    new=dict(id='nn-complete-trajectory-fold-bag',family='nn',development='upstream-complete-trajectory-h128-mps',
        known_prediction='artifacts/upstream-complete-trajectory-h128-mps/nn.f32',
        qualifying_prediction='artifacts/upstream-complete-trajectory-fold-bag/nn-bag.f32')
    c=dict(family_ids=base['family_ids']+[new['id']],families=base['families']+[new],simplex_weights=[1/15]*15,
        ridge_weights=[.1]*15,ridge_intercept=0.,ridge_lambda=10.,source_sha256=source,
        qualifying_labels_read=False,known_score_is_nested_cross_validation=False)
    known=dict(state='complete_original15_known_only_coefficients_before_candidate_freeze',observed='2026-11-21T20:01:00+00:00',
        plan_sha256=hashes['plan.json'],calibration_sha256=hashes['calibration.json'],known_predictions={k:pred[k] for k in g.KNOWN},qualifying_labels_read=False)
    f=dict(state='complete_all_three_own15_candidates_frozen_before_any_new_audit',primary=g.CANDIDATES[0],observed='2026-11-21T20:02:00+00:00',
        plan_sha256=hashes['plan.json'],calibration_sha256=hashes['calibration.json'],known_completion_sha256=hashes['known-completion.json'],
        predictions=pred,source_sha256=full_source,qualifying_labels_read=False)
    pre=dict(state='complete_independent_all32767_supports_ridge16_and_six_vectors_before_audit',observed='2026-11-21T20:03:00+00:00',
        freeze_sha256=hashes['frozen.json'],calibration_sha256=hashes['calibration.json'],verifier_sha256=p['verifier_sha256'],
        independent_simplex_weights=[1/15]*15,simplex_supports_examined=32767,ridge_equations=16,ridge_lambda=10.,
        actual_simplex_optimality=dict(objective=0.,independent_optimum=0.,absolute_gap=0.,minimum_reduced_gradient=0.,
            complementarity_error=0.,nonunique_coefficients_allowed=True),all_six_candidate_vectors_bitwise=True,
        source_sha256=full_source,qualifying_labels_read=False,known_score_is_nested_cross_validation=False)
    results=[dict(candidate=name,groups={'source0':dict(rows=1408789,rmse=.86+i/1000),
        'source1':dict(rows=1408342,rmse=.861+i/1000)},prediction_sha256=pred[name],both_groups_below_target=False)
        for i,name in enumerate(g.CANDIDATES)]
    a=dict(state='complete_all_three_original_own15_frozen_candidate_audits',observed='2026-11-21T20:04:00+00:00',
        primary=g.CANDIDATES[0],results=copy.deepcopy(results),preaudit_verification_sha256=hashes['independent-preaudit-verification.json'],
        label_git_blob='47ed4788ffba11aad029854a8e60d4dee04fc899',qualifying_labels_used_for_fitting_or_selection=False,official_or_untouched_test=False)
    latest=dict(state='complete_independent_own15_full_source_models_coefficients_and_frozen_group_scores',observed='2026-11-21T20:05:00+00:00',
        primary=g.CANDIDATES[0],results=results,groups=copy.deepcopy(results[0]['groups']),prediction_sha256=pred[g.CANDIDATES[0]],
        both_groups_below_target=False,simplex_supports_examined=32767,ridge_equations=16,all_six_vectors_independently_verified=True,
        full_inherited_source_closure_rehashed=True,actual_new_qualifying_models_reopened=15,source_sha256=full_source,
        label_git_blob=a['label_git_blob'],qualifying_labels_used_for_fitting_or_selection=False,
        known_score_is_nested_cross_validation=False,official_or_untouched_test=False)
    return latest,dict(zip(g.METADATA,[p,c,known,f,pre,a])),base,hashes


class Metadata(unittest.TestCase):
    def test_complete_private_schema(self):g.validate_metadata(*fixture())

    def test_nonunique_optimal_coefficients_remain_valid(self):
        args=fixture();pre=args[1]['independent-preaudit-verification.json']
        pre['independent_simplex_weights']=[1.]+[0.]*14
        g.validate_metadata(*args)  # Full actual KKT/objective proof, not coefficient distance.

    def test_outcome_selected_primary_or_mixed_group_vector_rejected(self):
        args=fixture();latest,parts,_,_=args
        for value in (latest,parts['plan.json'],parts['frozen.json'],parts['independent-recovered-audit.json']):value['primary']=g.CANDIDATES[1]
        with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[0]['groups']['source0']['rmse']=.84
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_fewer_families_models_supports_or_equations_rejected(self):
        cases=[('latest','actual_new_qualifying_models_reopened',14),('latest','simplex_supports_examined',32766),
            ('latest','ridge_equations',15),('plan.json','families',14)]
        for target,key,value in cases:
            args=fixture();obj=args[0] if target=='latest' else args[1][target];obj[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[1]['calibration.json']['families'].pop()
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_fitting_labels_official_claim_or_nested_claim_rejected(self):
        for target,key in [('plan.json','qualifying_labels_read'),('frozen.json','qualifying_labels_read'),
            ('independent-preaudit-verification.json','qualifying_labels_read'),('latest','qualifying_labels_used_for_fitting_or_selection'),
            ('latest','official_or_untouched_test'),('calibration.json','known_score_is_nested_cross_validation')]:
            args=fixture();obj=args[0] if target=='latest' else args[1][target];obj[key]=True
            with self.subTest(target=target,key=key),self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_wrong_known_freeze_preaudit_order_or_hash_rejected(self):
        args=fixture();args[1]['independent-preaudit-verification.json']['observed']='2026-11-21T20:06:00+00:00'
        with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[1]['known-completion.json']['observed']='2026-11-21T20:01:00'
        with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[1]['frozen.json']['known_completion_sha256']='0'*64
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_bad_actual_KKT_gap_or_partial_vectors_rejected(self):
        for key,value in [('absolute_gap',1e-9),('minimum_reduced_gradient',-1e-6),('complementarity_error',1e-6)]:
            args=fixture();args[1]['independent-preaudit-verification.json']['actual_simplex_optimality'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();del args[1]['frozen.json']['predictions'][g.CANDIDATES[2]]
        with self.assertRaises(ValueError):g.validate_metadata(*args)

    def test_complete_groups_labels_and_strict_same_candidate_target_required(self):
        args=fixture();args[0]['results'][1]['groups']['source1']['rows']-=1
        with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[0]['label_git_blob']='0'*40
        with self.assertRaises(ValueError):g.validate_metadata(*args)
        args=fixture();args[0]['results'][0]['both_groups_below_target']=True;args[0]['both_groups_below_target']=True
        with self.assertRaises(ValueError):g.validate_metadata(*args)


class Sources(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.names=['scripts/calibration.py','artifacts/upstream-own391-paired-h128-mps/nn/0/model.pt',
            'artifacts/upstream-complete-trajectory-h128-mps/nn/0/nn-2026.pt',
            'artifacts/upstream-complete-trajectory-h128-mps/nn/0/scalers.npz',
            'artifacts/upstream-complete-trajectory-fold-bag/nn-bag.f32',
            'experiments/upstream-complete-trajectory-fold-bag/independent-verification.json']
        self.bindings={}
        for name in self.names:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode());self.bindings[name]=g.digest(p)

    def test_all_categories_and_actual_byte_tampering(self):
        self.assertEqual(sum(g.verify_sources(self.root,self.bindings).values()),len(self.names))
        for name in self.names:
            p=self.root/name;before=p.read_bytes();p.write_bytes(before+b'changed')
            try:
                with self.subTest(name=name),self.assertRaises(ValueError):g.verify_sources(self.root,self.bindings)
            finally:p.write_bytes(before)

    def test_missing_bag_or_external_binding_rejected(self):
        missing={k:v for k,v in self.bindings.items() if 'fold-bag/' not in k}
        with self.assertRaises(ValueError):g.verify_sources(self.root,missing)
        self.bindings['../external.py']='0'*64
        with self.assertRaises(ValueError):g.verify_sources(self.root,self.bindings)


if __name__=='__main__':unittest.main()
