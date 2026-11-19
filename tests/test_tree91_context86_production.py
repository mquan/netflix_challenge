import copy
import unittest
import upstream_tree91_context86_production as p


class NeuralReuseGateTest(unittest.TestCase):
    def fixtures(self):
        config=dict(seeds=[2026,7,13],architecture='context-weighted-residual-cf12-v1',context_rank=16)
        plan=dict(config=config,seeds=config['seeds'],iterations=dict(nn=[19,20,20]),
                  qualifying_rows=2817131,qualifying_labels_read=False)
        outputs={name:name for name in ['qualifying.f32','scaler.npz','nn-2026.pt','nn-7.pt','nn-13.pt']}
        receipt=dict(outputs=outputs,plan_sha256='plan',qualifying_labels_read=False)
        replay=dict(bitwise_equal=True,plan_sha256='plan',family_receipt_sha256='receipt',prediction_sha256='qualifying.f32')
        return plan,receipt,replay,config

    def test_exact_recipe_and_complete_replay_are_required(self):
        plan,receipt,replay,config=self.fixtures()
        p.check_neural_origin(plan,receipt,replay,config,[19,20,20],'plan','receipt')
        changes=[('config',dict(config,context_rank=8)),('iterations',dict(nn=[19,20,21])),
                 ('qualifying_rows',2817130),('qualifying_labels_read',True)]
        for key,value in changes:
            changed=copy.deepcopy(plan);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                p.check_neural_origin(changed,receipt,replay,config,[19,20,20],'plan','receipt')
        for key,value in [('bitwise_equal',False),('family_receipt_sha256','other'),('prediction_sha256','other')]:
            changed=dict(replay,**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):
                p.check_neural_origin(plan,receipt,changed,config,[19,20,20],'plan','receipt')

    def test_missing_model_or_other_plan_is_rejected(self):
        plan,receipt,replay,config=self.fixtures()
        for filename in receipt['outputs']:
            changed=copy.deepcopy(receipt);del changed['outputs'][filename]
            with self.subTest(filename=filename),self.assertRaises(ValueError):
                p.check_neural_origin(plan,changed,replay,config,[19,20,20],'plan','receipt')
        with self.assertRaises(ValueError):
            p.check_neural_origin(plan,dict(receipt,plan_sha256='other'),replay,config,[19,20,20],'plan','receipt')


if __name__=='__main__':unittest.main()
