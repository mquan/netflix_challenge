import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import upstream_remote_integration as r


class RemoteIntegrationChecks(unittest.TestCase):
    def make_transfer(self, root):
        data=root/'data';data.mkdir()
        for name in ['ratings.npy','user_idxs.npy','item_idxs.npy','dates.npy']:
            np.save(data/name,np.arange(8,dtype=np.int32))
        dest=root/'experiments/upstream-remote-12h';(dest/'outputs').mkdir(parents=True)
        receipt=dict(source_commit=r.u.PIN,job='bknbrx-100',command='native',outputs={})
        entries=[]
        for suffix in ['', '_ep01','_ep05']:
            value=np.arange(8,dtype=np.float32)+(1 if suffix=='_ep01' else 0)
            source=f'preds_new/bknbrx-100{suffix}.probex.npy'
            native=root/(Path(source).name);np.save(native,value)
            receipt['outputs'][source]=r.u.sha(native)
            path=dest/'outputs'/(Path(source).name+'.npz');np.savez_compressed(path,prediction=value)
            entries.append(dict(source=source,source_npy_sha256=r.u.sha(native),
                receipt_bound_sha256=r.u.sha(native),npz=str(path.relative_to(root)),npz_sha256=r.u.sha(path),
                prediction_content_sha256=r.x.array_sha(value)))
        rp=dest/'native-receipt.json';r.u.write(rp,receipt)
        for e in entries:e['owning_native_receipt_sha256']=r.u.sha(rp)
        alignment={p.name:r.u.sha(p) for p in data.iterdir()}
        manifest=dict(source_commit=r.u.PIN,job='bknbrx-100',native_command='native',
            native_receipt=str(rp.relative_to(root)),native_receipt_sha256=r.u.sha(rp),
            alignment_sha256=alignment,outputs=entries,probex_rows=8,
            fresh_data_receipt=dict(source_commit=r.u.PIN,qualifying_outcomes_zero=True,
                                    ratings_sha256=alignment['ratings.npy'],rows=8))
        mp=dest/'transfer-manifest.json';r.u.write(mp,manifest)
        return mp,data,manifest

    def test_receives_exact_vectors_and_deduplicates_terminal_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp,data,_=self.make_transfer(root)
            arrays,record=r.receive(mp,root,data)
            self.assertEqual(list(arrays),['bknbrx-100','bknbrx-100_ep01'])
            self.assertEqual(record['duplicate_of'],{'bknbrx-100_ep05':'bknbrx-100'})
            np.testing.assert_array_equal(arrays['bknbrx-100'],np.arange(8,dtype=np.float32))

    def test_rejects_alignment_receipt_archive_and_content_changes(self):
        for damage,pattern in [('alignment','alignment'),('receipt','receipt'),
                               ('archive','compressed'),('content','content')]:
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);mp,data,manifest=self.make_transfer(root)
                if damage=='alignment':np.save(data/'dates.npy',np.zeros(8))
                elif damage=='receipt':(root/manifest['native_receipt']).write_text('{}')
                else:
                    entry=manifest['outputs'][0];path=root/entry['npz']
                    np.savez_compressed(path,prediction=np.ones(8,dtype=np.float32))
                    if damage=='content':
                        entry['npz_sha256']=r.u.sha(path);r.u.write(mp,manifest)
                with self.assertRaisesRegex(ValueError,pattern):r.receive(mp,root,data)

    def test_augmented_ridge_matches_direct_fit_and_excludes_outer_labels(self):
        rng=np.random.default_rng(81)
        raw=rng.uniform(0,6,(100,5));z=np.column_stack([raw,np.ones(100)])
        groups=np.tile(np.arange(10),10);y=rng.uniform(1,5,100)
        actual,_,_=r.crossfit(z,y,groups)
        np.testing.assert_allclose(actual,r.x.crossfit(raw,y,groups//2),atol=1e-12)
        poisoned=y.copy();poisoned[groups//2==3]+=100
        changed,_,_=r.crossfit(z,poisoned,groups)
        np.testing.assert_array_equal(actual[groups//2==3],changed[groups//2==3])

    def test_adapter_uses_new_paths_and_freezes_its_own_code(self):
        with patch.multiple(r.n,RUN=r.PARENT_RUN,REPORT=r.PARENT_REPORT,code_hashes=r.BASE_CODE_HASHES):
            r.configure_engine()
            self.assertEqual(r.n.RUN,r.RUN)
            self.assertEqual(r.n.REPORT,r.REPORT)
            hashes=r.n.code_hashes()
            self.assertIn('scripts/upstream_remote_integration.py',hashes)
            self.assertEqual(hashes['scripts/upstream_nonlinear.py'],r.BASE_CODE_HASHES()['scripts/upstream_nonlinear.py'])


if __name__=='__main__':unittest.main()
