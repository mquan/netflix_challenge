"""Full-width partition/scaler invariants and unchanged established architecture."""
import unittest
import numpy as np
import upstream_complete_trajectory_mps as study
import upstream_exact_trajectory_scaler as scaler
import upstream_exact391_scaler as original_scaler
import verify_upstream_complete_trajectory_mps as independent


class CompleteTrajectoryNeuralTests(unittest.TestCase):
    def test_every_dynamic_width_keeps_every_signal_and_context(self):
        for width in range(618,747):
            signals=study.signal_indices(width)
            self.assertEqual(signals[:-1][:392],study.established.SIGNALS[:-1])
            self.assertEqual(signals[-1],width-1)
            self.assertEqual(sorted(signals+study.CONTEXT),list(range(width)))
            self.assertEqual(len(study.CONTEXT),225)
        for width in [617,747]:
            with self.assertRaises(ValueError):study.signal_indices(width)

    def test_original_budget_and_zero_addition_model_state_are_unchanged(self):
        import torch
        original=study.established.CONFIGS['h128'];actual=study.CONFIGS['h128']
        self.assertEqual({k:v for k,v in original.items() if k!='architecture'},
                         {k:v for k,v in actual.items() if k!='architecture'})
        self.assertEqual(actual['nn_hidden'],128)
        self.assertEqual(actual['context_rank'],32)
        self.assertEqual(actual['nn_epochs'],40)
        self.assertEqual(len(actual['seeds']),3)
        torch.manual_seed(420)
        old=study.established.model_factory(618,original)
        torch.manual_seed(420)
        new=study.model_factory(618,actual)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for key,tensor in old.state_dict().items():
            self.assertTrue(torch.equal(tensor,new.state_dict()[key]),key)
        wide=study.model_factory(746,actual)
        self.assertEqual(wide.base[0].in_features,746)
        self.assertEqual(wide.project.in_features,521)
        self.assertEqual(wide.gate[0].in_features,225)

    def test_full_dynamic_scaler_matches_independent_ordered_statistics(self):
        rng=np.random.default_rng(440)
        for width in [618,746]:
            raw=rng.normal(size=(67,width)).astype('f4')
            raw[:,0]=-0.;raw[:,1]=np.arange(67)%2
            raw[:,2]=np.where(np.arange(67)%2,1e20,-1e20)
            raw[:,3]=np.where(np.arange(67)%3,1e-30,-1e-30)
            mask=np.arange(67)%5!=0
            expected=independent.independent_scaler(raw,mask)
            for block in [1,7,8192]:
                actual=scaler.fit_scaler(raw,mask,block_rows=block)
                for key in expected:
                    np.testing.assert_array_equal(actual[key].view('u8'),expected[key].view('u8'))
            if width==618:
                old=original_scaler.fit_scaler(raw,mask)
                for key in expected:np.testing.assert_array_equal(old[key].view('u8'),expected[key].view('u8'))


if __name__=='__main__':unittest.main()
