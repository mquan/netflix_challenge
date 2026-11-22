"""CPU full-width stream/statistic/copy checks; no source/provider admission."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_rbm400_context_data as module


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(128);n=8201
        cls.z=rng.normal(size=(n,562));cls.z[:,-1]=1
        cls.f=rng.normal(size=(n,712)).astype('<f4')
        cls.pred=rng.normal(3.6,1,size=n).astype('<f4')
        cls.u=np.arange(n,dtype=np.int32)%100
        cls.i=np.arange(n,dtype=np.int32)%170
        cls.context=module.inputs.cache.Context(rng.normal(size=(100,50)).astype('<f4'),
            rng.normal(size=(170,50)).astype('<f4'),np.ones(100,dtype=bool))
        cls.y=rng.integers(1,6,size=n).astype(np.float64);cls.groups=(cls.u%10).astype(np.int8)

    def reader(self,keep=True):
        outer=self
        class Parent:
            def block(self,lo,hi):return outer.z[lo:hi],outer.f[lo:hi]
        return module.inputs.QueryInputs(Parent(),self.pred,self.u,self.i,self.context,lambda:None,
                                         retain_scalar=keep,latent=True)

    def test_full8201row_scan_all486_columns_both_operators_and_context_arms(self):
        reader=self.reader();proof,gram,rhs=module.scan_query(reader,True,lambda:None,(self.y,self.u,self.groups))
        self.assertEqual(proof['rows'],8201)
        self.assertEqual(proof['original_operator_sha256'],[hashlib.sha256(v.tobytes()).hexdigest() for v in (self.z,self.f)])
        self.assertEqual(proof['original_scalar_sha256'],[hashlib.sha256(np.clip(self.z[:,c],0,6).astype('<f4').tobytes()).hexdigest() for c in module.RAW_COLUMNS])
        self.assertEqual(proof['new_scalar_sha256'],hashlib.sha256(np.clip(self.pred,0,6).astype('<f4').tobytes()).hexdigest())
        z,f=reader.block(0,8201)
        for arm,width in [('scalar',713),('latent',863)]:
            self.assertEqual(proof['arm_operator_sha256'][arm],[hashlib.sha256(z.tobytes()).hexdigest(),hashlib.sha256(np.ascontiguousarray(f[:,:width]).tobytes()).hexdigest()])
        # Independent whole-group contractions differ from block accumulation
        # only by float64 roundoff, not omitted rows or altered eligibility.
        for group in range(10):
            a=z[self.groups==group];target=self.y[self.groups==group]
            np.testing.assert_allclose(gram[group],a.T@a,rtol=1e-12,atol=1e-10)
            np.testing.assert_allclose(rhs[group],a.T@target,rtol=1e-12,atol=1e-10)

    def test_alias_arm_keeps_old_design_exactly_and_no_qualifying_statistics(self):
        reader=self.reader(False);proof,gram,rhs=module.scan_query(reader,False,lambda:None)
        self.assertEqual(gram.shape,(10,562,562));self.assertFalse(np.any(gram));self.assertFalse(np.any(rhs))
        self.assertEqual(proof['original_operator_sha256'][0],hashlib.sha256(self.z.tobytes()).hexdigest())
        self.assertEqual(proof['arm_operator_sha256']['scalar'][1],hashlib.sha256(self.f.tobytes()).hexdigest())

    def test_canonical_user_coordinate_mismatch_rejected(self):
        wrong=self.u.copy();wrong[8199]+=1
        with self.assertRaises(ValueError):module.scan_query(self.reader(),True,lambda:None,(self.y,wrong,self.groups))

    def test_all_five_solutions_unpenalized_intercept_and_no_held_label_effect(self):
        _,gram,rhs=module.scan_query(self.reader(),True,lambda:None,(self.y,self.u,self.groups))
        actual=module.weights(gram,rhs);reg=np.eye(563)*1000.;reg[-1,-1]=0
        for fold in range(5):
            eligible=np.arange(10)//2!=fold
            matrix=gram[eligible].sum(axis=0)+reg;target=rhs[eligible].sum(axis=0)
            np.testing.assert_allclose(matrix@actual[fold],target,rtol=1e-10,atol=1e-8)
        changed=rhs.copy();changed[:2]+=123.
        alternate=module.weights(gram,changed)
        self.assertEqual(actual[0].tobytes(),alternate[0].tobytes())
        self.assertGreater(np.max(np.abs(actual[1:]-alternate[1:])),1e-6)

    def test_prepared_bytes_copy_preserves_original_inode_ctime_and_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();art=root/'output';art.mkdir();source=root/'users.npy'
            np.save(source,np.array([1,2,3],dtype=np.int32));before=source.stat()
            records=module.inputs.extraction.inputs.verify_files(root,{'users.npy':module.inputs.extraction.inputs.sha(source)},lambda:None)
            with patch.object(module,'ROOT',root),patch.object(module,'ART',art):
                result=module.copy_bound(source,'users.npy',records['users.npy'],lambda **kw:None)
            self.assertEqual(source.read_bytes(),(art/'users.npy').read_bytes())
            self.assertEqual(result['sha256'],records['users.npy']['sha256'])
            after=source.stat()
            self.assertEqual((before.st_ino,before.st_ctime_ns,before.st_nlink),(after.st_ino,after.st_ctime_ns,after.st_nlink))
            self.assertNotEqual(after.st_ino,(art/'users.npy').stat().st_ino)


if __name__=='__main__':unittest.main()
