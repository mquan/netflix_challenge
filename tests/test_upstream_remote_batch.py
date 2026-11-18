from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_remote_batch as b


class RemoteBatchChecks(unittest.TestCase):
    def fixture(self,root):
        data=root/'data';data.mkdir()
        for name in ['ratings.npy','user_idxs.npy','item_idxs.npy','dates.npy']:
            np.save(data/name,np.tile(np.arange(1,5,dtype=np.int32),2))
        job='bknbrx-100';directory=root/'experiments/upstream-remote-12h'/job
        output=directory.parent/'outputs';output.mkdir(parents=True);directory.mkdir()
        (directory/'native.out').write_text('fresh native job complete\n')
        receipt=dict(job=job,source_commit=b.u.PIN,command='native',
                     outputs={f'preds_new/{job}.out':b.u.sha(directory/'native.out')})
        entries=[];rows=[];y=np.load(data/'ratings.npy').astype(float)
        for suffix in ['', '_ep01','_ep05']:
            value=(np.arange(8,dtype=np.float32)+int(suffix=='_ep01'))
            source=f'preds_new/{job}{suffix}.probex.npy'
            path=output/(Path(source).name+'.npz');np.savez_compressed(path,prediction=value)
            digest=b.x.array_sha(value);source_digest=digest
            receipt['outputs'][source]=source_digest
            entries.append(dict(source=source,source_npy_sha256=source_digest,receipt_bound_sha256=source_digest,
                npz=str(path.relative_to(root)),npz_sha256=b.u.sha(path),prediction_content_sha256=digest,
                diagnostic_standalone_probex_rmse=float(np.sqrt(np.mean((value.astype(float)-y)**2)))))
        rp=directory/'native-receipt.json';b.u.write(rp,receipt)
        for e in entries:
            e['owning_native_receipt_sha256']=b.u.sha(rp)
            rows.append(dict(job=job,npz=e['npz'],npz_sha256=e['npz_sha256'],
                             content_sha256=e['prediction_content_sha256'],receipt_sha256=b.u.sha(rp)))
        alignment={p.name:b.u.sha(p) for p in data.iterdir()}
        manifest=dict(job=job,source_commit=b.u.PIN,native_receipt=str(rp.relative_to(root)),
            native_receipt_sha256=b.u.sha(rp),native_command='native',probex_rows=8,
            alignment_sha256=alignment,outputs=entries,all_receipt_output_hashes_verified=len(receipt['outputs']),
            fresh_data_receipt=dict(source_commit=b.u.PIN,qualifying_outcomes_zero=True,
                                    ratings_sha256=alignment['ratings.npy'],rows=8))
        mp=directory/'transfer-manifest.json';b.u.write(mp,manifest)
        ip=directory.parent/'transfer-index.json';b.u.write(ip,dict(rows=rows,unique_vectors=2))
        return ip,data,mp,manifest,{'bknbrx-100':{'command':'native'}}

    def test_all_exports_verified_and_duplicate_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ip,data,_,_,registry=self.fixture(root)
            arrays,report,diagnostics=b.receive(ip,root,data,registry)
            self.assertEqual(len(arrays),2);self.assertEqual(report['exports'],3)
            self.assertEqual(report['duplicate_of'],{'bknbrx-100_ep05':'bknbrx-100'})
            self.assertEqual(len(diagnostics),3)
            self.assertIn('experiments/upstream-remote-12h/bknbrx-100/native.out',report['source_sha256'])

    def test_inventory_command_alignment_and_native_log_rejected_when_changed(self):
        for damage,pattern in [('inventory','inventory'),('command','command'),
                               ('alignment','alignment'),('log','output log'),('content','content')]:
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);ip,data,mp,manifest,registry=self.fixture(root)
                if damage=='inventory':
                    index=b.u.read(ip);index['rows'].pop();b.u.write(ip,index)
                elif damage=='command':registry['bknbrx-100']['command']='other'
                elif damage=='alignment':np.save(data/'dates.npy',np.zeros(8))
                elif damage=='log':(mp.parent/'native.out').write_text('changed')
                else:
                    index=b.u.read(ip);index['rows'][0]['content_sha256']='wrong';b.u.write(ip,index)
                with self.assertRaisesRegex(ValueError,pattern):b.receive(ip,root,data,registry)

    def test_fixed_bundles_have_disjoint_main_and_secondary_inputs(self):
        self.assertEqual(len(b.MAIN),19);self.assertEqual(len(b.ALL),22);self.assertEqual(len(b.JOBS),16)
        self.assertFalse(set(b.MAIN)&set(b.INTERMEDIATES))
        self.assertEqual(len(set(b.MAIN)),len(b.MAIN))

    def test_legacy_and_canonical_fixture_report_keys_are_explicit(self):
        value={'rms_diff':.02}
        self.assertEqual(b.sender_entry({'ep01':value},'bknbrx-100','bknbrx-100_ep01'),(value,'ep01'))
        key='bknbrx-100_ep01.probex'
        self.assertEqual(b.sender_entry({key:value,'ep01':{'wrong':True}},'bknbrx-100','bknbrx-100_ep01'),(value,key))
        self.assertEqual(b.sender_entry({'ep01':value},'other','other_ep01'),(None,'other_ep01.probex'))

    def test_shared_screen_preserves_intercept_and_excludes_scoring_labels(self):
        rng=np.random.default_rng(44)
        raw=rng.uniform(0,6,(100,3));added=rng.uniform(0,6,(100,3))
        old=np.column_stack([raw,np.ones(100)]);y=rng.uniform(1,5,100)
        groups=np.tile(np.arange(10),10);arms=dict(control=[],single=[1],all=[0,1,2])
        result=b.screen(old,added,y,groups,arms)
        for name,columns in arms.items():
            expected=b.x.crossfit(np.column_stack([raw,added[:,columns]]),y,groups//2)
            np.testing.assert_allclose(result[name]['prediction'],expected,atol=1e-12)
        poisoned=y.copy();poisoned[groups//2==2]+=100
        after=b.screen(old,added,poisoned,groups,arms)
        for name in arms:
            np.testing.assert_array_equal(result[name]['prediction'][groups//2==2],
                                          after[name]['prediction'][groups//2==2])

    def test_engine_paths_and_code_are_distinct_from_parent(self):
        with patch.multiple(b.n,RUN=b.PARENT_RUN,REPORT=b.PARENT_REPORT,code_hashes=b.parent.code_hashes):
            b.configure()
            self.assertEqual(b.n.RUN,b.RUN)
            self.assertIn('scripts/upstream_remote_batch.py',b.n.code_hashes())


if __name__=='__main__':unittest.main()
