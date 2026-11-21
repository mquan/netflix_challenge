"""Pure arithmetic/dispatch rejection fixtures, without real MPS admission."""
import unittest
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import upstream_rbm400_latent_worker as worker


class WorkerTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(128)
        self.raw=dict(users=rng.normal(size=(9,50)).astype('<f4'),
                      items=rng.normal(size=(170,50)).astype('<f4'),covered=np.ones(9,dtype=bool))
        self.raw['covered'][3]=False;self.raw['users'][3]=np.nan
        self.query=dict(users=np.array([8,0,8,1,6]),items=np.array([169,0,169,31,22]))
        basis=np.linalg.qr(rng.normal(size=(50,50)))[0]
        self.known_items=(self.raw['items'].astype(np.float64)@basis).astype('<f4')

    def test_qualifying_alignment_preserves_uncovered_users_and_all_item_bits(self):
        value,maps=worker.aligned(self.raw,self.known_items)
        self.assertTrue(np.isnan(value['users'][3]).all())
        self.assertTrue(worker.equal_bits(value['covered'],self.raw['covered']))
        self.assertTrue(worker.equal_bits(value['items'],(self.raw['items'].astype(np.float64)@maps['item_map']).astype('<f4')))
        for user in np.flatnonzero(self.raw['covered']):
            np.testing.assert_allclose(value['users'][user],self.raw['users'][user].astype(np.float64)@maps['user_map'],rtol=1e-7,atol=1e-7)

    def test_every_query_feature_bit_and_measured_dot_bound(self):
        for aligned in (False,True):
            value,maps=worker.aligned(self.raw,self.known_items) if aligned else (self.raw,None)
            context=worker.math.Context(value['users'],value['items'],value['covered'])
            proof=worker.features_and_dots(context,self.raw,maps,self.query,lambda:None)
            self.assertEqual(proof['query_rows'],len(self.query['users']))
            self.assertTrue(proof['all_query_feature_bits_match'])
            self.assertLessEqual(proof['max_dot_product_error'],proof['max_dot_product_error_bound'])

    def test_corrupted_virtual_feature_bit_fails(self):
        context=worker.math.Context(self.raw['users'],self.raw['items'],self.raw['covered'])
        original=context.values
        def changed(*args):
            result=original(*args);result[0,149]=np.nextafter(result[0,149],np.float32(np.inf));return result
        with patch.object(context,'values',side_effect=changed):
            with self.assertRaises(ValueError):
                worker.features_and_dots(context,self.raw,None,self.query,lambda:None)

    def test_measured_cast_bound_does_not_replace_complete_cache_bit_check(self):
        value,maps=worker.aligned(self.raw,self.known_items)
        value['users'][0]*=2
        context=worker.math.Context(value['users'],value['items'],value['covered'])
        # Cache-bit verification separately rejects this alteration. Invariance
        # still bounds its measured cast discrepancy rather than claiming exact
        # equality for altered stored coordinates.
        proof=worker.features_and_dots(context,self.raw,maps,self.query,lambda:None)
        self.assertGreater(proof['max_dot_product_error'],0)
        self.assertFalse(worker.equal_bits(value['users'],worker.aligned(self.raw,self.known_items)[0]['users']))

    def test_cpu_fixture_cannot_publish_actual_extraction_or_verification(self):
        fixture=dict(self.raw,metadata=dict(actual_MPS_predictions_replayed=False))
        with patch.object(worker.provider,'extract_phase',return_value=(fixture,{})),patch.object(worker,'save',side_effect=AssertionError('Must not write CPU fixture')):
            with self.assertRaises(ValueError):worker.extract('known','1'*64,lambda:None)

    def test_literal_cache_bits_include_nan_payloads_and_signed_zero(self):
        a=np.array([0.,np.nan],dtype='<f4');b=a.copy()
        self.assertTrue(worker.equal_bits(a,b))
        b[0]=-0.
        self.assertFalse(worker.equal_bits(a,b))
        b=a.copy();b.view('<u4')[1]^=1
        self.assertTrue(np.isnan(b[1]));self.assertFalse(worker.equal_bits(a,b))

    def test_paired_cache_IO_both_independent_replays_and_tamper_rejection(self):
        # This fixture mocks the separately tested actual-model adapter. Its
        # device booleans are dispatch fixtures, never evidence of actual MPS.
        ex=worker.execution
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder).resolve();study=root/'study';art=root/'art';report=root/'control'
            for p in (study,art,report,root/'query'):p.mkdir()
            (report/'plan.json').write_text('{}');(report/'first-scientific-baseline.json').write_text('{}')
            (root/'fixture.json').write_text('{"synthetic":true}')
            bindings={'fixture.json':ex.inputs.sha(root/'fixture.json')}
            paths={}
            for key in ('users','items'):
                relative='query/'+('user_idxs' if key=='users' else 'item_idxs')+'.npy'
                np.save(root/relative,self.query[key]);paths[relative]=ex.inputs.sha(root/relative)
            records=ex.inputs.verify_files(root,paths,lambda:None)
            (study/'data-admission.json').write_text(json.dumps({'bound_file_records':records}))
            declaration={'phases':{p:{'training':'unused','query':'query'} for p in ('known','qualifying')}}
            def fixture_extract(phase,guard):
                value=copy.deepcopy(self.raw)
                if phase=='known':value['items']=self.known_items.copy()
                value['metadata']=dict(actual_MPS_predictions_replayed=True,
                    original_full_source_and_own16_lifecycle_verified=True,all_native_queries_verified=True,
                    synthetic_fixture=True)
                return value,dict(bindings)
            for name,value in [('ROOT',root),('REPORT',report),('STUDY',study),('ART',art),
                               ('USERS',9),('ITEMS',170),('COUNTS',{'known':5,'qualifying':5})]:
                stack.enter_context(patch.object(ex,name,value))
            stack.enter_context(patch.object(worker.provider.original,'ROOT',root))
            stack.enter_context(patch.object(worker.provider.rbm,'STUDY',study))
            stack.enter_context(patch.object(worker.provider,'extract_phase',side_effect=fixture_extract))
            stack.enter_context(patch.object(worker.original_worker,'data_receipt',return_value=({},declaration,'1'*64)))
            guard=lambda **kwargs:None;plan_sha='2'*64
            for phase in ('known','qualifying'):
                receipt=worker.extract(phase,plan_sha,guard)
                ex.validate_result(phase,receipt,plan_sha)
                ex.exclusive_json(study/ex.RESULTS[phase][0],receipt,guard)
            for phase in ('known','qualifying'):
                proof=worker.verify(phase,plan_sha,guard)
                self.assertTrue(proof['all_actual_cache_bits_match'])
                self.assertTrue(proof['all_query_feature_bits_match'])
                ex.exclusive_json(study/ex.RESULTS['verify_'+phase][0],proof,guard)
            final=worker.assemble(plan_sha,guard)
            self.assertFalse(final['production_provider_admitted']);self.assertEqual(final['models_fitted'],0)
            self.assertEqual(final['independent_full_phase_replays'],2)
            with (art/'qualifying-users.npy').open('ab') as stream:stream.write(b'tampered')
            with self.assertRaises(ValueError):worker.verify('qualifying',plan_sha,guard)


if __name__=='__main__':unittest.main()
