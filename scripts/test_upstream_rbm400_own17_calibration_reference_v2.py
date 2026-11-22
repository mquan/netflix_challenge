"""Full seventeen-family CPU fit/preaudit/control-role fixtures; no production source admission."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_rbm400_own17_calibration_reference_v2 as study
import verify_upstream_rbm400_own17_calibration_reference_v2 as verify


class FullSyntheticStudyTests(unittest.TestCase):
    def test_active_faces_add_and_remove_coordinates_without_relaxing_KKT(self):
        gram=np.eye(17);guard=lambda:None
        vertex=np.zeros(17);vertex[0]=1.
        actual,proof=study.polish_simplex(gram,np.zeros(17),vertex,guard)
        np.testing.assert_allclose(actual,np.full(17,1/17),atol=1e-14,rtol=0.)
        self.assertEqual(proof['active_face_solves'],17)
        rhs=np.zeros(17);rhs[0]=10.
        actual,proof=study.polish_simplex(gram,rhs,np.full(17,1/17),guard)
        np.testing.assert_allclose(actual,vertex,atol=1e-14,rtol=0.)
        self.assertLessEqual(proof['complementarity_error'],1e-10)
        actual,proof=study.polish_simplex(np.zeros((17,17)),np.zeros(17),np.full(17,1/17),guard)
        self.assertAlmostEqual(actual.sum(),1.,places=14)
        self.assertLessEqual(proof['complementarity_error'],1e-10)
        with self.assertRaises(ValueError):study.polish_simplex(gram,np.zeros(16),vertex,guard)

    def test_known_scoring_reclips_float32_lower_bound_before_error(self):
        value=np.array([1.05],dtype='<f4')
        self.assertLess(float(value[0]),1.05)
        self.assertAlmostEqual(verify.known_rmse(value,np.array([1.])),.05,places=14)

    def test_all17_real_fit_full131071_support_replay_freeze_and_separate_audits(self):
        rng=np.random.default_rng(828);nk,nq=80,61
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder).resolve();report=root/'study';art=root/'art';old=root/'old';known=root/'known'
            for path in (report,art,old,known):path.mkdir()
            base=root/'base';base.mkdir();(base/'fresh-training-plan.json').write_text(json.dumps({'synthetic_qualifying_payload_sha256':'a'*64}))
            rows=[];bindings={}
            for index,name in enumerate(study.BASE_IDS+[study.FAMILY_ID]):
                values={'known':rng.normal(3,.7,nk).astype('<f4'),'qualifying':rng.normal(3,.7,nq).astype('<f4')}
                row={'id':name,'family':'synthetic'}
                for query,value in values.items():
                    filename='%s-%s.%s'%(index,query,'npy' if index==16 else 'f32');path=root/filename
                    if index==16:
                        with path.open('wb') as stream:np.save(stream,value,allow_pickle=False)
                        row['format']='npy'
                    else:value.tofile(path)
                    row[query+'_prediction']=filename;bindings[filename]=study.inputs.sha(path)
                rows.append(row)
            y=rng.integers(1,6,nk,dtype=np.int8)
            with (known/'labels.npy').open('wb') as stream:np.save(stream,y,allow_pickle=False)
            bindings['known/labels.npy']=study.inputs.sha(known/'labels.npy')
            np.save(old/'known-half.npy',rng.uniform(1.05,4.95,nk).astype('<f4'))
            np.save(old/'qual-half.npy',rng.uniform(1.05,4.95,nq).astype('<f4'))
            for path in old.iterdir():bindings[str(path.relative_to(root))]=study.inputs.sha(path)
            for obj,name,value in [(study,'ROOT',root),(study.rbm,'ROOT',root),(study,'REPORT',report),(study,'ART',art),(study,'COUNTS',{'known':nk,'qualifying':nq}),(study,'KNOWN',known),(study.preceding.preceding.u,'BASE',base),(verify,'GROUP_COUNTS',{0:30,1:31})]:
                stack.enter_context(patch.object(obj,name,value))
            stack.enter_context(patch.object(study,'source_declaration',return_value=(rows,bindings)))
            stack.enter_context(patch.object(study,'controls',return_value={
                'known':{'path':str(old/'known-half.npy')},'qualifying':{'path':str(old/'qual-half.npy')}}))
            guard=lambda **kwargs:None
            outcomes=np.column_stack([rng.integers(1,6,nq,dtype=np.int16),np.r_[np.zeros(30,dtype=np.int16),np.ones(31,dtype=np.int16)]])
            def gated_outcomes(guard):
                self.assertTrue((report/'independent-preaudit-verification.json').is_file())
                return outcomes,'synthetic-fixture-only'
            label_reader=stack.enter_context(patch.object(verify,'labels',side_effect=gated_outcomes))
            study.known(guard);study.frozen(guard)
            with self.assertRaises(FileNotFoundError):verify.audit(guard)
            label_reader.assert_not_called()
            verify.verify(guard)
            pre=study.rbm.read(report/'independent-preaudit-verification.json')
            self.assertEqual(pre['simplex_supports_examined'],131071);self.assertEqual(pre['ridge_equations'],18)
            self.assertFalse(pre['qualifying_labels_read']);label_reader.assert_not_called()
            verify.audit(guard);verify.canonical(guard);self.assertEqual(label_reader.call_count,2)
            result=study.rbm.read(report/'independent-canonical-verification.json')
            self.assertEqual(result['primary'],study.CANDIDATES[0]);self.assertFalse(result['qualifying_labels_used_for_fitting_or_selection'])
            self.assertFalse(result['official_or_untouched_test']);self.assertEqual(result['groups']['source0']['rows'],30)
            self.assertEqual(result['groups']['source1']['rows'],31);self.assertEqual(len(result['results']),3)
            # Alter a vector and its recorded hash: actual independent arithmetic
            # must still detect the change, even with newly authenticated bytes.
            path=art/'known-simplex.f32';changed=np.fromfile(path,dtype='<f4');changed[0]+=np.float32(.125);changed.tofile(path)
            freeze=study.rbm.read(report/'frozen.json');freeze['outputs']['known-simplex']['sha256']=study.inputs.sha(path)
            (report/'frozen.json').write_text(json.dumps(freeze))
            with self.assertRaisesRegex(ValueError,'Every actual stored arithmetic bit'):
                verify.verify(guard)

    def test_exact_population_schema_and_file_identity_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/'values.npy'
            with path.open('wb') as stream:np.save(stream,np.array([1,2,3],dtype='<f4'),allow_pickle=False)
            records=study.inputs.verify_files(root,{'values.npy':study.inputs.sha(path)},lambda:None)
            with patch.object(study,'ROOT',root):
                with study.vector('values.npy',3,records,'npy') as value:np.testing.assert_array_equal(value,[1,2,3])
                with self.assertRaises(ValueError):
                    with study.vector('values.npy',2,records,'npy'):pass
                before=path.read_bytes();path.unlink();path.write_bytes(before)
                with self.assertRaises(ValueError):
                    with study.vector('values.npy',3,records,'npy'):pass


if __name__=='__main__':unittest.main()
