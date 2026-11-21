"""Admission I/O contracts on temporary synthetic arrays; not actual-data admission."""
from pathlib import Path
from dataclasses import asdict
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_factored_rbm_admission as admission
from upstream_factored_rbm_admission import arrays,header_record,phase_paths,statistics
from upstream_factored_rbm_inputs import verify_files,sha,COMMON,MAPS,REPORT
from upstream_factored_rbm_training import Config
from upstream_factored_rbm_rows import UserRows
from test_upstream_factored_rbm_rows import fixture
from test_upstream_factored_rbm_data_audit import projection_fixture


class AdmissionTests(unittest.TestCase):
    def test_exact_native_header_and_payload_end_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.npy';np.save(path,np.arange(4,dtype=np.int16))
            r=header_record(path,'int16',(4,));self.assertEqual(r['payload_bytes'],8)
            with self.assertRaises(ValueError):header_record(path,'int32',(4,))
            with self.assertRaises(ValueError):header_record(path,'int16',(3,))
            with path.open('ab') as stream:stream.write(b'junk')
            with self.assertRaisesRegex(ValueError,'trailing'):header_record(path,'int16',(4,))

    def test_phase_query_field_whitelist_has_no_outcomes(self):
        plan=dict(phases={'known':dict(training='tr',query='probe'),'qualifying':dict(training='full',query='qual')})
        for phase in plan['phases']:
            tr,q=phase_paths(plan,phase)
            self.assertIn('ratings',tr);self.assertIn('is_test',tr)
            self.assertEqual(set(q),{'user_idxs','item_idxs','dates','user_cnts'})
        with self.assertRaises(ValueError):phase_paths(plan,'recovered')

    def test_memory_maps_close_on_success_and_guard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);np.save(root/'a.npy',np.arange(4,dtype=np.int16))
            with arrays(root,{'a':'a.npy'},lambda:None) as opened:
                observed=opened['a'];self.assertFalse(observed._mmap.closed)
            self.assertTrue(observed._mmap.closed)
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with arrays(root,{'a':'a.npy'},lambda:None) as opened:
                    observed=opened['a'];raise RuntimeError('synthetic fatal verification')
            self.assertTrue(observed._mmap.closed)
            with self.assertRaises(ValueError):
                with arrays(root,{'a':'../a.npy'},lambda:None):pass

    def test_real_tensor_shape_storage_counts_are_distinct_from_peak_admission(self):
        tr,q=fixture();reader=UserRows(tr,q,4)
        result=statistics(reader,dict(factors=2,hidden=3))
        expected=4*(4*5*2+2*3+4*3+4*5+3)+4*5*(4+3)
        self.assertEqual(result['predictive_parameter_bytes'],expected)
        self.assertEqual(result['model_plus_momentum_bytes'],2*expected)
        self.assertGreater(result['retained_reader_metadata_bytes'],0)
        self.assertIn('peaks still require measurement',result['scope'])

    def test_replaced_paths_cannot_substitute_for_fully_hashed_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'a.npy';np.save(path,np.arange(4,dtype=np.int16))
            records=verify_files(root,{'a.npy':sha(path)},lambda:None)
            np.save(root/'new.npy',np.full(4,99,dtype=np.int16));(root/'new.npy').replace(path)
            with self.assertRaisesRegex(ValueError,'fully hashed identity'):
                with arrays(root,{'a':'a.npy'},lambda:None,records):pass

    def test_map_uses_original_descriptor_and_rejects_changes_or_object_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'a.npy';original=np.arange(4,dtype=np.int16);np.save(path,original)
            records=verify_files(root,{'a.npy':sha(path)},lambda:None)
            with self.assertRaisesRegex(ValueError,'identity changed'):
                with arrays(root,{'a':'a.npy'},lambda:None,records) as opened:
                    observed=opened['a'];path.rename(root/'old.npy');np.save(path,np.full(4,99,dtype=np.int16))
                    np.testing.assert_array_equal(observed,original)
            self.assertTrue(observed._mmap.closed)
            np.save(path,np.array([{}],dtype=object))
            with self.assertRaisesRegex(ValueError,'primitive'):
                with arrays(root,{'a':'a.npy'},lambda:None):pass

    def test_complete_orchestration_uses_only_temporary_synthetic63_array_fixture(self):
        # Only the canonical provider/row-count boundary is replaced for this
        # tiny fixture. All actual hashing, headers, readers and independent
        # semantic/zero/disjointness checks below execute unmodified.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tr,q,known,mapping=projection_fixture()
            def dataset(relative,raw,rating,has_maps):
                folder=root/relative;folder.mkdir(parents=True)
                names=COMMON|(MAPS if has_maps else set());n=len(raw['user_idxs'])
                values=dict(user_idxs=raw['user_idxs'].astype(np.int32),item_idxs=raw['item_idxs'].astype(np.int16),
                    dates=raw['dates'].astype(np.int16),ratings=rating.astype(np.int8),
                    is_test=raw.get('is_test',np.zeros(n,dtype=np.int8)),
                    user_cnts=np.bincount(raw['user_idxs'],minlength=4).astype(np.int32),
                    item_cnts=np.bincount(raw['item_idxs'],minlength=4).astype(np.int32),
                    item_years=np.full(4,2000,dtype=np.int32),uidx_to_uid=np.arange(1,5,dtype=np.int32),
                    iidx_to_iid=np.arange(1,5,dtype=np.int16))
                for name in names:np.save(folder/name,values[Path(name).stem])
            for split in ('train','probe','probex','fulltrain','qual'):
                raw=tr if split in ('train','fulltrain') else known if split=='probex' else q
                ratings=np.zeros(len(q['user_idxs']),dtype=np.int8) if split=='qual' else raw['ratings']
                dataset('artifacts/upstream-repro/fresh/data/'+split,raw,ratings,split in ('train','fulltrain'))
            for split,raw in [('train_3',tr),('probe_3',q)]:
                dataset('artifacts/upstream-split3-pilot/source/data/'+split,raw,raw['ratings'],split=='train_3')
            np.save(root/'artifacts/upstream-split3-pilot/probex-row-map.npy',mapping)
            paths=list((root/'artifacts/upstream-repro/fresh/data').rglob('*.npy'))+list((root/'artifacts/upstream-split3-pilot/source/data').rglob('*.npy'))
            paths.append(root/'artifacts/upstream-split3-pilot/probex-row-map.npy')
            bound={str(p.relative_to(root)):sha(p) for p in paths};self.assertEqual(len(bound),63)
            plan=dict(raw_input_sha256=bound,source_code_sha256={},original_reference_sha256={},provider_sha256={},prototype_proof_sha256={},
                phases={'known':dict(training='artifacts/upstream-split3-pilot/source/data/train_3',query='artifacts/upstream-split3-pilot/source/data/probe_3',training_rows=6,query_rows=6),
                        'qualifying':dict(training='artifacts/upstream-repro/fresh/data/fulltrain',query='artifacts/upstream-repro/fresh/data/qual',training_rows=6,query_rows=6)},
                known_rows=3,users=4,items=4,exact_config=asdict(Config()))
            (root/REPORT).mkdir(parents=True);(root/REPORT/'plan.json').write_text(json.dumps(plan))
            counts={k:3 if k=='probex' else 6 for k in admission.ROW_COUNTS}
            with patch.object(admission,'load_declared_plan',return_value=plan),patch.object(admission,'ROW_COUNTS',counts):
                result=admission.admit(root,lambda:None)
                self.assertEqual(len(result['raw_array_schemas']),63)
                self.assertEqual(result['phases']['known']['reader']['positive_ratings'],4)
                self.assertTrue(result['phases']['known']['projection']['native_holdout_identities_absent_from_positive_training'])
                self.assertTrue(result['phases']['qualifying']['zero_targets']['zero_targets'])
                target=root/'artifacts/upstream-repro/fresh/data/qual/ratings.npy';np.save(target,np.ones(6,dtype=np.int8))
                # Changed expectation is permitted only by this isolated mock
                # to exercise the semantic zero check after a passing hash.
                bound[str(target.relative_to(root))]=sha(target)
                with self.assertRaisesRegex(ValueError,'not zero'):admission.admit(root,lambda:None)
