"""Independent data replay tests, with explicit mocked production admission."""
from contextlib import contextmanager,ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import verify_upstream_rbm400_context_data_reference_v2 as verification
import upstream_rbm400_context_data_reference_v2 as data
import test_upstream_rbm400_context_data_reference_v2 as fixtures


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.DataTests.setUpClass()

    def fixture(self):return fixtures.DataTests()

    def test_canonical_float64_reclip_after_float32_storage(self):
        predictions=np.array([1.05,4.95],dtype='<f4');labels=np.array([1.,5.])
        self.assertLess(float(predictions[0]),1.05)
        clipped=[max(1.05,min(4.95,float(v))) for v in predictions]
        expected=((clipped[0]-1.)**2+(clipped[1]-5.)**2)**.5/np.sqrt(2.)
        self.assertAlmostEqual(verification.known_score(predictions,labels),float(expected),places=15)
        raw=np.sqrt(np.mean((predictions.astype(np.float64)-labels)**2))
        self.assertNotEqual(verification.known_score(predictions,labels),raw)

    def test_independent_full_streams_and_statistics_without_producer_scan(self):
        fixture=self.fixture();reader=fixture.reader()
        expected,g,r=data.scan_query(reader,True,lambda:None,(fixture.y,fixture.u,fixture.groups))
        with patch.object(data,'scan_query',side_effect=AssertionError('Verifier must not call producer scanner')):
            actual,gram,rhs=verification.replay_query(reader,True,lambda:None,fixture.y,fixture.u,fixture.groups)
        self.assertEqual(expected,actual);self.assertEqual(g.tobytes(),gram.tobytes());self.assertEqual(r.tobytes(),rhs.tobytes())

    def test_every_solved_coefficient_bit_and_equation(self):
        fixture=self.fixture()
        _,g,r=data.scan_query(fixture.reader(),True,lambda:None,(fixture.y,fixture.u,fixture.groups))
        weights=data.weights(g,r)
        residuals=verification.coefficients(g,r,weights);self.assertEqual(len(residuals),5)
        changed=weights.copy();changed[2,561]=np.nextafter(changed[2,561],np.inf)
        with self.assertRaises(ValueError):verification.coefficients(g,r,changed)

    def test_complete_scaled_admit_independent_verify_and_output_tamper(self):
        # Real file copies, statistics, solves, saved outputs and verifier;
        # source admission and query-file ownership are mocked ONLY in this
        # temporary synthetic fixture. No real provider/MPS claim follows.
        rng=np.random.default_rng(128);counts={'probex':100,'qual':103};raw={}
        context=data.inputs.cache.Context(rng.normal(size=(100,50)).astype('<f4'),
            rng.normal(size=(170,50)).astype('<f4'),np.ones(100,dtype=bool))
        for q,n in counts.items():
            z=rng.normal(size=(n,562));z[:,-1]=1
            raw[q]=dict(z=z,f=rng.normal(size=(n,712)).astype('<f4'),
                        p=rng.normal(3.6,1,size=n).astype('<f4'),u=(np.arange(n)%100).astype('<i4'),
                        i=(np.arange(n)%170).astype('<i4'))
        candidate_rows=[dict(id='candidate:'+str(i),paired_files={'probex':'k'+str(i),'qual':'q'+str(i)}) for i in range(128)]
        clipped={};streams={}
        for q,values in raw.items():
            hs=[hashlib.sha256(np.clip(values['z'][:,c],0,6).astype('<f4').tobytes()).hexdigest() for c in data.RAW_COLUMNS]
            clipped[q]=dict(original=hs[:391],candidates=hs[:33]+hs[391:])
            streams[q]=[hashlib.sha256(values[key].tobytes()).hexdigest() for key in ('z','f')]
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder).resolve();prepared=root/'source';prepared.mkdir();(root/'scripts').mkdir()
            for name in ('upstream_rbm400_context_data_reference_v2.py','upstream_rbm400_context_inputs_reference_v2.py','verify_upstream_rbm400_context_data_reference_v2.py'):
                source=Path(data.__file__).parent/name;(root/'scripts'/name).write_bytes(source.read_bytes())
            for name,value in [('labels.npy',rng.integers(1,6,size=100).astype('<f8')),
                               ('users.npy',raw['probex']['u']),('groups.npy',(raw['probex']['u']%10).astype('i1'))]:np.save(prepared/name,value)
            for q in counts:np.save(root/(q+'-scalar.npy'),raw[q]['p'])
            base_bindings={str(p.relative_to(root)):data.inputs.extraction.inputs.sha(p) for p in list(prepared.iterdir())+list((root/'scripts').iterdir())+[root/(q+'-scalar.npy') for q in counts]}
            declaration=dict(source_sha256=base_bindings,parent_plan={'candidates':candidate_rows},
                parent={'retained':candidate_rows[33:],'virtual_c_order_sha256':streams},
                input_manifest={'clipped_sha256':clipped},scalar_paths={q:q+'-scalar.npy' for q in counts})
            def sources(guard):
                value=copy.deepcopy(declaration)
                return value,data.inputs.extraction.inputs.verify_files(root,value['source_sha256'],guard)
            @contextmanager
            def open_query(q,decl,records,guard,*,retain_scalar,latent):
                values=raw[q]
                class Parent:
                    def block(self,lo,hi):return values['z'][lo:hi],values['f'][lo:hi]
                yield data.inputs.QueryInputs(Parent(),values['p'],values['u'],values['i'],context if latent else None,
                                             guard,retain_scalar=retain_scalar,latent=latent)
            for module,name,value in [(data,'ROOT',root),(data,'REPORT',root/'report'),(data,'ART',root/'art'),
                                      (data.inputs,'ROOT',root),(data.inputs,'COUNTS',counts),
                                      (data.inputs.extraction,'ROOT',root),(data.inputs.provider.original,'ROOT',root),
                                      (data.inputs.preceding,'RUN',prepared)]:stack.enter_context(patch.object(module,name,value))
            stack.enter_context(patch.object(data.inputs,'source_declaration',side_effect=sources))
            stack.enter_context(patch.object(data.inputs,'open_query',side_effect=open_query))
            guard=lambda **kwargs:None
            manifest=data.admit(guard);self.assertTrue(manifest['alias']['retain_scalar'])
            result=verification.verify(guard)
            self.assertEqual(result['state'],verification.STATE)
            self.assertTrue(result['every_both_arm_operator_bit_verified']);self.assertEqual(result['ridge_equations'],2815)
            self.assertFalse(result['qualifying_labels_read']);self.assertEqual(result['neural_models_fitted'],0)
            # The verifier output is exclusively created: never replace it.
            with (data.ART/'ridge.npy').open('ab') as stream:stream.write(b'tampered')
            with self.assertRaises(ValueError):verification.verify(guard)


if __name__=='__main__':unittest.main()
