"""Source-expectation isolation and guarded full-byte hashing, synthetic files."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from upstream_factored_rbm_inputs import COMMON,MAPS,PIN,compile_records,verify_files


def fixture():
    raw={s+'/'+n:'a'*64 for s in ('train','probe','probex','fulltrain','qual') for n in COMMON|(MAPS if s in ('train','fulltrain') else set())}
    split={s+'/'+n:'b'*64 for s in ('train_3','probe_3') for n in COMMON|(MAPS if s=='train_3' else set())}
    weekend=dict(source_pin=PIN,qualifying_labels_read=False,public_prediction_values_read=False,data_sha256=raw,split3_sha256=split,row_map_sha256='c'*64)
    pilot=dict(source_pin=PIN,state='complete',rows=1128164,source_rows=1760494,known_rows_absent_from_training=True,
        known_labels_match=True,qualifying_labels_read=False,data_sha256={'source/data/'+n:s for n,s in split.items()},row_map_sha256='c'*64)
    known=dict(state='complete_original_holdout_projection',known_rows=1128164,source_rows=1408395,unique_exact_identity_mapping=True,
        known_labels_equal=True,qualifying_labels_read=False,input_sha256={'artifacts/upstream-repro/fresh/data/'+s+'/'+n+'.npy':'a'*64 for s in ('probe','probex') for n in ('user_idxs','item_idxs','dates','ratings')})
    return weekend,pilot,known


class InputTests(unittest.TestCase):
    def test_exact_original_expectations_and_map_survive_declaration(self):
        w,p,k=fixture();out=compile_records(w,p,k)
        self.assertEqual(len(out),63)
        self.assertEqual(out['artifacts/upstream-repro/fresh/data/qual/ratings.npy'],'a'*64)
        self.assertEqual(out['artifacts/upstream-split3-pilot/source/data/train_3/ratings.npy'],'b'*64)
        self.assertEqual(out['artifacts/upstream-split3-pilot/probex-row-map.npy'],'c'*64)

    def test_conflicting_incomplete_or_outcome_read_providers_are_rejected(self):
        for change in (
            lambda w,p,k:p['data_sha256'].update({'source/data/train_3/ratings.npy':'d'*64}),
            lambda w,p,k:w['data_sha256'].pop('qual/ratings.npy'),
            lambda w,p,k:w['data_sha256'].update({'../outcomes.npy':'d'*64}),
            lambda w,p,k:w.update(qualifying_labels_read=True),
            lambda w,p,k:k['input_sha256'].update({'artifacts/upstream-repro/fresh/data/probex/ratings.npy':'d'*64}),
            lambda w,p,k:p.update(row_map_sha256='d'*64),
        ):
            w,p,k=fixture();change(w,p,k)
            with self.assertRaises(ValueError):compile_records(w,p,k)

    def test_all_payload_bytes_and_fixed_expected_digest_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'raw.npy';payload=b'x'*(1048576+17);path.write_bytes(payload)
            digest=hashlib.sha256(payload).hexdigest();calls=[]
            got=verify_files(root,{'raw.npy':digest},lambda:calls.append(True))
            self.assertEqual(got['raw.npy']['bytes'],len(payload));self.assertGreaterEqual(len(calls),5)
            path.write_bytes(payload[:-1]+b'y')
            with self.assertRaisesRegex(ValueError,'expected hash mismatch'):verify_files(root,{'raw.npy':digest},lambda:None)

    def test_guard_failure_path_escape_and_midread_identity_change_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'raw.npy';payload=b'x'*(1048576+17);path.write_bytes(payload)
            digest=hashlib.sha256(payload).hexdigest()
            with self.assertRaises(ValueError):verify_files(root,{'../raw.npy':digest},lambda:None)
            def fail():raise RuntimeError('synthetic resource failure')
            with self.assertRaises(RuntimeError):verify_files(root,{'raw.npy':digest},fail)
            count=0
            def change():
                nonlocal count
                count+=1
                if count==3:
                    with path.open('r+b') as stream:stream.seek(-1,2);stream.write(b'y')
            with self.assertRaisesRegex(ValueError,'identity changed'):verify_files(root,{'raw.npy':digest},change)


if __name__=='__main__':unittest.main()
