import pathlib
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path('scripts').resolve()))
from gbdt_blend import blend_fold
from moonshot_distribution import fit_fold, calibrated_centers


class DistributionTests(unittest.TestCase):
    def test_prior_preserves_expected_rating(self):
        base=torch.tensor([-1.,1.,1.1,2.,3.,3.7,4.9,5.,7.])
        center=calibrated_centers(base); levels=torch.arange(1,6)
        predicted=((-0.5*(levels[None,:]-center[:,None]).square()).softmax(1)*levels).sum(1)
        np.testing.assert_allclose(predicted.numpy(),base.clamp(1.0001,4.9999).numpy(),atol=2e-5)

    def test_held_labels_do_not_fit_or_select_head(self):
        rng=np.random.default_rng(19); users=np.repeat(np.arange(1000,dtype=np.uint32),3)
        pred=rng.uniform(1.5,4.5,size=(len(users),3)).astype(np.float32)
        context=np.column_stack([np.ones(len(users)),rng.normal(size=len(users))]).astype(np.float32)
        metadata=rng.normal(size=(len(users),2)).astype(np.float32)
        # Strong metadata signal ensures this exercises a trained head, not fallback.
        y=np.where(metadata[:,0]>0,5.,1.)
        held=blend_fold(users)==0; changed=y.copy(); changed[held]=6-changed[held]
        for head,calibrated in [('mean',False),('distribution',False),('distribution',True)]:
            first,epoch=fit_fold(pred,context,metadata,y,users,0,head=head,epochs=12,calibrated=calibrated)
            second,epoch2=fit_fold(pred,context,metadata,changed,users,0,head=head,epochs=12,calibrated=calibrated)
            self.assertGreater(epoch,0)
            np.testing.assert_array_equal(first,second); self.assertEqual(epoch,epoch2)
            if head=='distribution': self.assertTrue(((first>=1)&(first<=5)).all())


if __name__=='__main__': unittest.main()
