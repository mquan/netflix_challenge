"""Corruption, completeness, native side values and same-process admission checks."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import numpy as np
import upstream_remote_weekend_return as run
import upstream_remote_weekend_return_queue as queue
import verify_upstream_remote_weekend_return as independent


def npy(values):
    out=io.BytesIO();np.save(out,values,allow_pickle=False);return out.getvalue()


def bundle(payloads, names=None, link=False):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as archive:
        for rel,payload in payloads.items():
            info=tarfile.TarInfo((names or {}).get(rel,'tiny/'+rel));info.size=len(payload)
            if link:info.type=tarfile.SYMTYPE;info.linkname='/tmp/elsewhere';info.size=0
            archive.addfile(info,None if link else io.BytesIO(payload))
    blob=out.getvalue();expected={p:run.digest(v) for p,v in payloads.items()}
    return blob,dict(file='tiny-part01.tar.gz',bytes=len(blob),sha256=run.digest(blob),members_sha256=expected),expected


class ReturnTests(unittest.TestCase):
    def extract(self,payloads,*,shapes=None,names=None,link=False,part_change=None,blob_change=None):
        blob,part,expected=bundle(payloads,names,link)
        if part_change:part_change(part)
        if blob_change:blob=blob_change(blob)
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(run,'native_shapes',return_value=shapes or {}):
            root=Path(tmp);seen=run.extract_part(io.BytesIO(blob),part,{'name':'tiny'},root,expected,lambda:None)
            return seen,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}

    def test_all_real_native_shapes_and_epoch_pair_counts(self):
        plan=run.read(run.PLAN)
        self.assertEqual([len(j['published_columns']) for j in plan['jobs']],[16,25,26])
        for job in plan['jobs']:
            shapes=run.native_shapes(job)
            self.assertEqual(len(shapes),2*len(job['published_columns'])+2+(2 if job['kind']=='rbm' else 0))
            self.assertEqual(shapes['preds_3/'+job['name']+'.train_3.npy'],(98720013,))
        self.assertEqual(run.native_shapes(plan['jobs'][0])['preds_3/s3-rbm400pf.ifeat.fulltrain.npy'],(17770,250))
        self.assertEqual(run.native_shapes(plan['jobs'][2])['preds_3/s3-rbm800p.ifeat.fulltrain.npy'],(17770,4000))

    def test_stream_all_values_preserve_query_bytes_discard_only_archived_sides(self):
        payloads={'native/preds_3/tiny.probe_3.npy':npy(np.array([1,2,3,4],dtype='<f4')),
                  'native/preds_3/tiny.train_3.npy':npy(np.arange(10000,dtype='<f4')),
                  'source/src/main.rs':b'original source','tiny.npz':b'compact payload'}
        shapes={p[len('native/'):]:((4,) if 'probe' in p else (10000,)) for p in payloads if p.startswith('native/')}
        seen,saved=self.extract(payloads,shapes=shapes)
        self.assertEqual(set(seen),set(payloads));self.assertNotIn('native/preds_3/tiny.train_3.npy',saved)
        for p,value in saved.items():self.assertEqual(value,payloads[p])
        self.assertTrue(seen['native/preds_3/tiny.train_3.npy']['native_finite_checked'])

    def test_side_array_nonfinite_is_rejected_even_when_not_materialized(self):
        for value in [np.inf,np.nan,-np.inf]:
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'Nonfinite'):
                self.extract({'native/preds_3/tiny.train_3.npy':npy(np.array([value],dtype='<f4'))},
                             shapes={'preds_3/tiny.train_3.npy':(1,)})

    def test_shape_dtype_and_truncated_npy_rejected(self):
        for payload in [npy(np.ones(2,dtype='<f8')),npy(np.ones(3,dtype='<f4')),npy(np.ones(2,dtype='<f4'))[:-1]]:
            with self.subTest(length=len(payload)),self.assertRaises(ValueError):
                self.extract({'native/preds_3/tiny.train_3.npy':payload},shapes={'preds_3/tiny.train_3.npy':(2,)})

    def test_unsafe_member_and_link_rejected(self):
        for name in ['/tmp/escape','tiny/../escape','other/file','tiny//file','tiny/./file']:
            with self.subTest(name=name),self.assertRaises(ValueError):self.extract({'file':b'abc'},names={'file':name})
        with self.assertRaises(ValueError):self.extract({'file':b'abc'},link=True)
        for name in ['.','../file','/tmp/file','a\\b','a//b']:
            with self.assertRaises(ValueError):run.safe_name(name)

    def test_plaintext_and_compressed_hashes_both_required(self):
        def wrong_member(p):p['members_sha256']['file']='0'*64
        def wrong_archive(p):p['sha256']='0'*64
        for change in [wrong_member,wrong_archive]:
            with self.assertRaises(ValueError):self.extract({'file':b'abc'},part_change=change)
        with self.assertRaises((EOFError,ValueError,tarfile.ReadError)):
            self.extract({'file':b'abc'},blob_change=lambda b:b[:-6])

    def test_declared_missing_member_rejected(self):
        def missing(p):p['members_sha256']['missing']=hashlib.sha256(b'').hexdigest()
        with self.assertRaises(ValueError):self.extract({'file':b'abc'},part_change=missing)

    def test_compact_headers_reject_wrong_shape_before_array_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pair.npz';counts={'probex':2,'qual':3}
            np.savez_compressed(path,probex=np.ones(2,dtype='<f4'),qual=np.ones(3,dtype='<f4'))
            independent.compact_headers(path,counts)
            for known in [np.ones(2,dtype='<f8'),np.ones((2,1),dtype='<f4'),np.ones(20,dtype='<f4')]:
                np.savez_compressed(path,probex=known,qual=np.ones(3,dtype='<f4'))
                with self.assertRaises(ValueError):independent.compact_headers(path,counts)

    def test_gzip_crc_is_verified_even_if_compressed_digest_is_updated(self):
        blob,part,expected=bundle({'file':b'abc'})
        broken=blob[:-8]+bytes([blob[-8]^1])+blob[-7:]
        part.update(bytes=len(broken),sha256=run.digest(broken))
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(run,'native_shapes',return_value={}):
            with self.assertRaises((OSError,EOFError)):
                run.extract_part(io.BytesIO(broken),part,{'name':'tiny'},Path(tmp),expected,lambda:None)

    def test_actual_completed_metadata_closure_is_not_current_hash_replacement(self):
        job=run.job_spec('s3-tx300r');name=job['name'];plan=run.read(run.PLAN)
        log=('\n'.join('Epoch %02d | RMSE .9'%i for i in range(1,25))+'\n').encode()*2
        dumps=lambda x:(json.dumps(x,indent=2)+'\n').encode()
        launch=dumps(dict(job=name,plan_sha256=run.ops.sha(run.PLAN),command=job['command'],environment=job['environment']))
        outputs={p:'a'*64 for p in run.native_shapes(job)};outputs['preds_3/'+name+'.out']=run.digest(log)
        receipt=dumps(dict(state='native_complete_pending_export',job=name,outputs=outputs,
                           log_sha256=run.digest(log),launch_sha256=run.digest(launch)))
        files={column+'.npz':'b'*64 for column in job['published_columns']}
        files.update({'plan.json':run.ops.sha(run.PLAN),name+'-receipt.json':run.digest(receipt),name+'-launch.json':run.digest(launch),
                      name+'.log':run.digest(log),'preparation.json':'c'*64,'build.log':'c'*64,'split3.log':'c'*64,
                      'upstream_remote_weekend.py':plan['controller_sha256']['scripts/upstream_remote_weekend.py']})
        source={'source/'+p:s for p,s in plan['source_files_sha256'].items()}
        source.update({'source/target/release/fit':'d'*64,'source/target/release/split_3':'e'*64})
        proof=dict(state='verified_complete_original_remote_weekend_producer',job=name,source_pin=plan['source_pin'],
                   source_data_binary_rehashed=True,both_full_original_epoch_traces=True,independent_known_projection=True,
                   qualifying_labels_read=False,public_prediction_values_read=False,
                   all_native_output_sha256={'native/'+p:s for p,s in outputs.items()},original_source_and_binary_sha256=source,
                   files_sha256=files,records=[dict(column=c,file=c+'.npz',sha256=files[c+'.npz'],
                        native_probe_sha256=outputs['preds_3/'+c+'.probe_3.npy'],native_qual_sha256=outputs['preds_3/'+c+'.qual.npy']) for c in job['published_columns']])
        encoded=dumps(proof);expected={**files,**proof['all_native_output_sha256'],**source,'verification.json':run.digest(encoded)}
        index=dict(state='complete_all_original_native_outputs_and_all_epoch_pairs_lossless_return',job=name,
                   plan_sha256=run.ops.sha(run.PLAN),qualifying_labels_read=False,raw_dataset_uploaded=False,all_original_files_retained=True,
                   source_verification_sha256=run.digest(encoded),parts=[dict(file=name+'-part01.tar.gz',bytes=100,sha256='f'*64,members_sha256=expected)])
        encoded_index=dumps(index)
        transfer=dict(job=name,release_tag=run.TAG,release_draft=True,numeric_arrays_in_git=False,
                      assets=[dict(name=name+'-part01.tar.gz',bytes=100,sha256='f'*64),
                              dict(name=name+'-index.json',bytes=len(encoded_index),sha256=run.digest(encoded_index))])
        blobs={'archive-index.json':encoded_index,'export-verification.json':encoded,name+'-receipt.json':receipt,
               name+'-launch.json':launch,name+'.log':log,'transfer.json':dumps(transfer)}
        run.validate_metadata(job,blobs)
        for mutation in ['epoch','known_only_alias','source_hash','recipe']:
            changed=copy.deepcopy(blobs)
            if mutation=='epoch':changed[name+'.log']=log.replace(b'Epoch 09',b'Epoch 08')
            if mutation=='known_only_alias':
                obj=json.loads(encoded);obj['records'].pop();changed['export-verification.json']=dumps(obj)
            if mutation=='source_hash':
                obj=json.loads(encoded);obj['original_source_and_binary_sha256']['source/Cargo.toml']='0'*64;changed['export-verification.json']=dumps(obj)
            if mutation=='recipe':
                obj=json.loads(launch);obj['environment']['F']='30';changed[name+'-launch.json']=dumps(obj)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):run.validate_metadata(job,changed)

    def test_queue_requires_actual_canonical_own15_path_and_same_child(self):
        script=(run.u.ROOT/'scripts/upstream_own15_complete_trajectory_queue.py').read_text()
        expected=queue.PREREQUISITES[0]
        self.assertIn("independent-canonical-verification.json",expected[3]);self.assertIn(expected[1],script)
        captured=dict(pid=10,pgid=10,started='original',command='bootstrap',worker_command='worker --job tiny')
        self.assertTrue(queue.same_child(captured,dict(captured,command=captured['worker_command'])))
        self.assertFalse(queue.same_child(captured,dict(captured,started='reused PID')))


if __name__=='__main__':unittest.main()
