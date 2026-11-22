"""Full-width join and both-query alias fixtures; no real model admission."""
import copy
import hashlib
import unittest
import numpy as np
import upstream_rbm400_context_inputs_reference_v2 as module


def digest(value):return hashlib.sha256(value.encode()).hexdigest()


class InputTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(128)
        self.z=rng.normal(size=(9,562));self.z[:,-1]=1
        self.f=rng.normal(size=(9,712)).astype('<f4');self.f[0,0]=-0.
        self.prediction=np.array([-1.,0.,1.,2.,3.,4.,5.,6.,8.],dtype='<f4')
        self.users=np.array([3,0,3,2,2,1,0,1,3]);self.items=np.array([8,0,8,6,4,2,7,4,8])
        self.context=module.cache.Context(rng.normal(size=(4,50)).astype('<f4'),
                                          rng.normal(size=(11,50)).astype('<f4'),np.ones(4,dtype=bool))
        outer=self
        class Parent:
            def block(self,lo,hi):return outer.z[lo:hi],outer.f[lo:hi]
        self.parent=Parent();self.calls=0

    def guard(self):self.calls+=1

    def reader(self,keep,latent):
        return module.QueryInputs(self.parent,self.prediction,self.users,self.items,
            self.context if latent else None,self.guard,retain_scalar=keep,latent=latent)

    def test_both_arms_all_alias_widths_preserve_every_parent_bit(self):
        before=[self.z.tobytes(),self.f.tobytes(),self.prediction.tobytes()]
        for keep in (False,True):
            for latent in (False,True):
                with self.subTest(keep=keep,latent=latent):
                    z,f=self.reader(keep,latent).block(0,9)
                    self.assertEqual(z.shape,(9,562+keep));self.assertEqual(f.shape,(9,712+keep+150*latent))
                    self.assertEqual(z[:,:561].tobytes(),self.z[:,:561].tobytes())
                    self.assertEqual(f[:,:712].tobytes(),self.f.tobytes())
                    self.assertEqual(z[:,-1].tobytes(),self.z[:,-1].tobytes())
                    if keep:
                        expected=np.clip(self.prediction,0,6)
                        self.assertEqual(z[:,-2].astype('<f4').tobytes(),expected.tobytes())
                        self.assertEqual(f[:,712].tobytes(),(expected-np.float32(3.6)).tobytes())
                    if latent:
                        expected=self.context.values(self.users,self.items)
                        self.assertEqual(f[:,-150:].tobytes(),expected.tobytes())
        self.assertEqual(before,[self.z.tobytes(),self.f.tobytes(),self.prediction.tobytes()])
        self.assertEqual(self.calls,8)

    def test_block_partition_duplicates_and_uncovered_addresses(self):
        reader=self.reader(True,True);z,f=reader.block(0,9)
        a,b=reader.block(0,4);c,d=reader.block(4,9)
        self.assertEqual(np.concatenate([a,c]).tobytes(),z.tobytes())
        self.assertEqual(np.concatenate([b,d]).tobytes(),f.tobytes())
        self.context.covered[3]=False
        with self.assertRaises(ValueError):reader.block(0,1)

    def test_invalid_row_contract_and_nonfinite_scalar_fail(self):
        for lo,hi in ((-1,2),(1,1),(0,10),(0.,2)):
            with self.assertRaises(ValueError):self.reader(True,True).block(lo,hi)
        with self.assertRaises(ValueError):
            module.QueryInputs(self.parent,self.prediction,self.users,self.items,None,self.guard,
                               retain_scalar=True,latent=True)
        self.prediction[2]=np.inf
        with self.assertRaises(ValueError):self.reader(True,False).block(0,4)

    def test_complete_clip_hash_preserves_scalar_order_and_signed_zero(self):
        expected=hashlib.sha256(np.clip(self.prediction,0,6).astype('<f4').tobytes()).hexdigest()
        self.assertEqual(module.clipped_hash(self.prediction,self.guard),expected)
        values=np.array([-0.,0.],dtype='<f4')
        self.assertEqual(module.clipped_hash(values,self.guard),hashlib.sha256(np.clip(values,0,6).astype('<f4').tobytes()).hexdigest())
        with self.assertRaises(ValueError):module.clipped_hash(np.array([np.inf],dtype='<f4'),self.guard)

    def test_alias_requires_both_complete_queries_and_first_original_wins(self):
        existing={q:[digest(q+str(i)) for i in range(486)] for q in module.COUNTS}
        names=['original:'+str(i) for i in range(486)]
        pair={q:existing[q][17] for q in module.COUNTS}
        result=module.paired_alias(existing,pair,names)
        self.assertFalse(result['retain_scalar']);self.assertEqual(result['alias_of'],names[17])
        pair['qual']=existing['qual'][18]
        self.assertTrue(module.paired_alias(existing,pair,names)['retain_scalar'])
        for q in existing:existing[q][18]=existing[q][17]
        pair={q:existing[q][18] for q in existing}
        self.assertEqual(module.paired_alias(existing,pair,names)['alias_of'],names[17])
        with self.assertRaises(ValueError):module.paired_alias(existing,{'probex':pair['probex']},names)

    def test_parent_hashes_preserve_all519_to486_declared_order(self):
        rows=[dict(id='candidate:'+str(i),paired_files={'probex':'k'+str(i),'qual':'q'+str(i)}) for i in range(128)]
        retained=rows[33:]
        hashes={q:dict(original=[digest(q+'o'+str(i)) for i in range(391)],
                       candidates=[digest(q+'c'+str(i)) for i in range(128)]) for q in module.COUNTS}
        declaration=dict(parent_plan={'candidates':rows},parent={'retained':retained},input_manifest={'clipped_sha256':hashes})
        result,names=module.parent_pairs(declaration)
        self.assertEqual(len(names),486);self.assertEqual(names[-95:],[r['id'] for r in retained])
        for q in result:self.assertEqual(result[q],hashes[q]['original']+hashes[q]['candidates'][33:])
        bad=copy.deepcopy(declaration);bad['parent']['retained'][0]['paired_files']['qual']='changed'
        # deepcopy retains list element aliasing; replace the whole retained row.
        bad['parent']['retained'][0]=dict(id=rows[33]['id'],paired_files={'probex':'k33','qual':'changed'})
        bad['parent_plan']['candidates'][33]=rows[33]
        with self.assertRaises(ValueError):module.parent_pairs(bad)
        bad=copy.deepcopy(declaration);bad['parent']['retained'][0]=bad['parent']['retained'][1]
        with self.assertRaises(ValueError):module.parent_pairs(bad)
        bad=copy.deepcopy(declaration);bad['parent']['retained'].reverse()
        with self.assertRaises(ValueError):module.parent_pairs(bad)


if __name__=='__main__':unittest.main()
