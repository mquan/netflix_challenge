"""CPU construction and inherited-budget checks; no actual NN fitting or MPS."""
import unittest
import upstream_complete_trajectory_mps as original
import upstream_rbm400_latent_neural as neural
import upstream_rbm400_latent_operators as ops


class NeuralFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def test_all_original_hyperparameters_except_architecture_identity_unchanged(self):
        expected={k:v for k,v in original.CONFIGS['h128'].items() if k!='architecture'}
        for cfg in neural.CONFIGS.values():self.assertEqual({k:v for k,v in cfg.items() if k!='architecture'},expected)
        self.assertEqual(expected['seeds'],[2026,7,13])
        self.assertEqual((expected['nn_epochs'],expected['nn_patience'],expected['nn_batch']),(40,4,8192))

    def test_both_full_arms_and_alias_layouts_keep_all_inputs_and_zero_residual_start(self):
        t=self.torch
        for arm,cfg in neural.CONFIGS.items():
            for retain in (False,True):
                with self.subTest(arm=arm,retain=retain):
                    width=ops.dimensions(retain,arm=='latent')['nn_inputs']
                    model=neural.model_factory(width,cfg);model.eval()
                    self.assertEqual(model.base[0].in_features,width)
                    self.assertEqual(model.gate[0].in_features,375 if arm=='latent' else 225)
                    self.assertEqual(model.project.out_features,32)
                    self.assertEqual(sorted(model.signals.tolist()+model.context.tolist()),list(range(width)))
                    self.assertTrue(t.equal(model(t.randn(3,width)),t.zeros((3,1))))

    def test_scalar_alias_factory_matches_every_original_initial_model_bit(self):
        t=self.torch;t.manual_seed(128);old=original.model_factory(713,original.CONFIGS['h128'])
        t.manual_seed(128);new=neural.model_factory(713,neural.CONFIGS['scalar'])
        a,b=old.state_dict(),new.state_dict();self.assertEqual(set(a),set(b))
        for key in a:self.assertTrue(t.equal(a[key],b[key]),key)

    def test_invented_or_reduced_recipe_and_invalid_widths_reject(self):
        for change in (dict(nn_epochs=1),dict(nn_hidden=64),dict(seeds=[2026]),dict(nn_batch=4096)):
            with self.assertRaises(ValueError):neural.model_factory(864,dict(neural.CONFIGS['latent'],**change))
        for width in (712,714,865):
            with self.assertRaises(ValueError):neural.model_factory(width,neural.CONFIGS['latent'])


if __name__=='__main__':unittest.main()
