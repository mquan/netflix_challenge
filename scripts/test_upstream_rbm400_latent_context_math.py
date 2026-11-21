"""Independent scientific arithmetic checks; no real models or outcomes read."""
import unittest
import numpy as np
import upstream_rbm400_latent_context_math as m


class RepresentationTests(unittest.TestCase):
    def setUp(self):self.rng=np.random.default_rng(128)

    def test_full50factor_contrast_against_independent_category_dot(self):
        p=self.rng.normal(size=(73,5,50)).astype(np.float32)
        actual=m.item_contrast(p)
        reference=np.empty((73,50),dtype=np.float64)
        weights=np.array([-2,-1,0,1,2],dtype=np.float64)/np.sqrt(10.)
        for item in range(73):
            for factor in range(50):reference[item,factor]=np.dot(weights,p[item,:,factor].astype(np.float64))
        self.assertTrue(np.allclose(actual,reference,rtol=1e-7,atol=5e-7))
        self.assertEqual(actual.dtype,np.dtype('<f4'))
        self.assertTrue(np.array_equal(actual,m.item_contrast(p)))

    def test_full_alignment_normal_equations_and_independent_inverse(self):
        qual=self.rng.normal(size=(170,50));known=self.rng.normal(size=(170,50))
        maps=m.alignment(known,qual)
        gram=qual.T@qual+1e-6*np.eye(50);rhs=qual.T@known
        self.assertLess(np.max(np.abs(gram@maps['item_map']-rhs)),1e-11)
        # Independently solve for the dual map rather than reusing inv().
        independent=np.linalg.solve(maps['item_map'].T,np.eye(50))
        self.assertTrue(np.allclose(maps['user_map'],independent,rtol=1e-10,atol=1e-10))

    def test_contragredient_alignment_preserves_query_dot_products(self):
        qual=self.rng.normal(size=(170,50));basis=np.linalg.qr(self.rng.normal(size=(50,50)))[0]
        known=qual@basis;users=self.rng.normal(size=(13,50))
        maps=m.alignment(known,qual);u,i=m.align_qualifying(users,qual,maps)
        # Compare all2210 products using separate per-pair arithmetic.
        before=np.array([[np.dot(a,b) for b in qual] for a in users])
        after=u.astype(np.float64)@i.astype(np.float64).T
        self.assertLess(np.max(np.abs(before-after)),2e-6)
        self.assertTrue(np.array_equal(u,m.align_qualifying(users,qual,maps)[0]))

    def test_rank_deficient_or_ill_conditioned_alignment_fails(self):
        qual=self.rng.normal(size=(100,50));known=qual.copy();known[:,-1]=known[:,0]
        with self.assertRaises(ValueError):m.alignment(known,qual)
        known=qual.copy();known[:,-1]*=1e-9
        with self.assertRaises(ValueError):m.alignment(known,qual)

    def test_virtual_rows_preserve_order_duplicates_and_full150coordinates(self):
        users=self.rng.normal(size=(7,50)).astype(np.float32);items=self.rng.normal(size=(11,50)).astype(np.float32)
        c=m.Context(users,items,np.ones(7,dtype=bool))
        u=np.array([5,0,5,1]);i=np.array([3,8,3,0]);actual=c.values(u,i)
        expected=np.array([list(users[a])+list(items[b])+list(users[a]*items[b]) for a,b in zip(u,i)],dtype=np.float32)
        self.assertTrue(np.array_equal(actual,expected));self.assertEqual(actual.shape,(4,150))

    def test_uncovered_query_user_never_becomes_a_zero_posterior(self):
        users=np.ones((3,50),dtype=np.float32);users[1]=np.nan
        covered=np.array([True,False,True]);c=m.Context(users,np.ones((5,50)),covered)
        with self.assertRaises(ValueError):c.values(np.array([1]),np.array([0]))
        self.assertTrue(np.isfinite(c.values(np.array([2]),np.array([0]))).all())

    def test_wrong_dimensions_addresses_or_nonfinite_state_fail(self):
        with self.assertRaises(ValueError):m.item_contrast(np.zeros((10,5,49)))
        with self.assertRaises(ValueError):m.alignment(np.zeros((20,50)),np.zeros((20,50)))
        c=m.Context(np.ones((3,50)),np.ones((4,50)),np.ones(3,dtype=bool))
        for users,items in [(np.array([-1]),np.array([0])),(np.array([0]),np.array([4])),(np.array([0.]),np.array([0]))]:
            with self.assertRaises(ValueError):c.values(users,items)

    def test_finite_inputs_cannot_produce_infinite_float32_caches(self):
        p=np.zeros((3,5,50),dtype=np.float32)
        p[:,:2,:]=-np.float32(3e38);p[:,3:,:]=np.float32(3e38)
        self.assertTrue(np.isfinite(p).all())
        with self.assertRaises(ValueError):m.item_contrast(p)
        maps=dict(item_map=np.eye(50),user_map=np.eye(50))
        with self.assertRaises(ValueError):m.align_qualifying(np.full((2,50),1e39),np.ones((3,50)),maps)


if __name__=='__main__':unittest.main()
