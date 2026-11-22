"""Full-width fold/order/memory and CPU realization checks; no MPS admission."""
import hashlib
import unittest
import weakref
import numpy as np
import upstream_rbm400_neural_fold as scheduler


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(128);self.users=np.arange(100,dtype=np.int32)
        self.groups=(self.users%10).astype(np.int8);self.labels=np.full(100,5.,dtype=np.float64)
        self.raw=self.rng.normal(size=(100,864)).astype('<f4');self.weights=np.ones(563,dtype=np.float64)

    def build(self,stage,weights):
        values=self.raw.copy();baseline=np.full(100,1.5 if stage=='inner' else 2.,dtype=np.float64)
        values[:,-1]=baseline
        return values,baseline,dict(raw_neural_input_sha256=hashlib.sha256(values.tobytes()).hexdigest())

    def call(self,train,predict,build=None,save_model=lambda *a:None,labels=None):
        return scheduler.run('latent',2,self.labels if labels is None else labels,self.users,self.groups,
            self.weights,self.weights,build=build or self.build,train_nn=train,predict=predict,
            save_preprocessing=lambda *a:None,save_model=save_model,release=lambda:None,guard=lambda:None)

    def test_all_seed_slots_original_budgets_zero_count_and_single_matrix_lifetime(self):
        traces=[];saved=[];references=[]
        counts={2028:0,9:2,15:1}
        def build(stage,weights):
            if stage=='full':self.assertIsNone(references[0]())
            value,baseline,proof=self.build(stage,weights);references.append(weakref.ref(value));return value,baseline,proof
        def train(x,y,cfg,seed,epochs,valid=None):
            traces.append((seed,epochs,valid is not None,len(x),len(valid[0]) if valid is not None else None,y.copy()))
            self.assertEqual(cfg['nn_patience'],4);self.assertEqual(cfg['nn_batch'],8192)
            return dict(seed=seed),counts[seed] if valid is not None else epochs
        result=self.call(train,lambda model,x:np.full(len(x),model['seed']*1e-5,dtype='<f4'),build,
                         lambda model,seed,count:saved.append((seed,count)))
        self.assertEqual(result['chosen_iterations'],[0,2,1]);self.assertEqual(result['seed_order'],[2026,7,13])
        self.assertEqual([t[:3] for t in traces],[(2028,40,True),(9,40,True),(15,40,True),(9,2,False),(15,1,False)])
        self.assertEqual(saved,[(7,2),(13,1)])
        self.assertTrue(all(r() is None for r in references));self.assertEqual(result['full_representation_matrices_resident_at_once'],1)

    def test_all_folds_keep_exact_masks_and_reject_user_inner_group_leakage(self):
        for fold in range(5):
            m=scheduler.masks(self.users,self.groups,fold)
            self.assertTrue(np.array_equal(m['train'],self.groups//2!=fold))
            self.assertTrue(np.array_equal(m['fit'],m['train']&(self.groups%2==0)))
            self.assertTrue(np.array_equal(m['valid'],m['train']&(self.groups%2==1)))
        broken=self.users.copy();broken[1]=broken[0]
        with self.assertRaises(ValueError):scheduler.masks(broken,self.groups,2)

    def test_held_labels_never_change_inner_or_outer_training_targets(self):
        def targets(labels):
            captured=[]
            def train(x,y,cfg,seed,epochs,valid=None):
                captured.append(y.tobytes())
                if valid is not None:captured.append(valid[1].tobytes())
                return object(),1
            self.call(train,lambda model,x:np.zeros(len(x),dtype='<f4'),labels=labels)
            return captured
        changed=self.labels.copy();changed[self.groups//2==2]=1.
        self.assertEqual(targets(self.labels),targets(changed))

    def test_two_stage_order_matches_alternating_full_budget_CPU_models_bitwise(self):
        # A full-width CPU realization verifies seed-reset independence and all
        # original40/patience4 budgets. It establishes no actual MPS parity.
        import torch
        torch.set_num_threads(1);cfg=scheduler.architecture.CONFIGS['latent'];actual_states={}
        def predict(model,x):
            model.eval()
            with torch.no_grad():return model(torch.from_numpy(np.ascontiguousarray(x))).squeeze(1).numpy().copy()
        def train(x,y,config,seed,epochs,valid=None):
            torch.manual_seed(seed);model=scheduler.architecture.model_factory(x.shape[1],config)
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['nn_lr'],weight_decay=config['nn_weight_decay'])
            tx=torch.from_numpy(x);ty=torch.from_numpy(y.astype('<f4'))
            best,count=(float(np.mean(valid[1]**2)),0) if valid is not None else (None,epochs)
            for epoch in range(1,epochs+1):
                model.train()
                for batch in torch.randperm(len(x)).split(config['nn_batch']):
                    optimizer.zero_grad(set_to_none=True)
                    (model(tx[batch]).squeeze(1)-ty[batch]).square().mean().backward();optimizer.step()
                if valid is not None:
                    loss=float(np.mean((predict(model,valid[0])-valid[1])**2))
                    if loss<best:best,count=loss,epoch
                    if epoch-count>=config['nn_patience']:break
            return model,count
        def save(model,seed,count):actual_states[seed]={k:v.detach().clone() for k,v in model.state_dict().items()}
        result=self.call(train,predict,save_model=save)
        masks=scheduler.masks(self.users,self.groups,2);representations={}
        for stage,eligible in [('inner',masks['fit']),('full',masks['train'])]:
            values,baseline,_=self.build(stage,self.weights)
            scaler=scheduler.scaling.fit_scaler(values,eligible,guard=lambda:None)
            scheduler.scaling.apply_scaler(values,scaler,guard=lambda:None,output=values)
            representations[stage]=(values,baseline)
        inner,ib=representations['inner'];full,fb=representations['full'];correction=np.zeros(int(masks['held'].sum()));counts=[]
        for seed in cfg['seeds']:
            _,count=train(inner[masks['fit']],self.labels[masks['fit']]-ib[masks['fit']],cfg,seed+2,40,
                          valid=(inner[masks['valid']],self.labels[masks['valid']]-ib[masks['valid']]))
            counts.append(count)
            if count:
                model,_=train(full[masks['train']],self.labels[masks['train']]-fb[masks['train']],cfg,seed+2,count)
                for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,actual_states[seed][name]),(seed,name))
                correction+=predict(model,full[masks['held']])/3
        expected=np.clip(fb[masks['held']]+correction,1.05,4.95).astype('<f4')
        self.assertTrue(any(counts));self.assertEqual(counts,result['chosen_iterations'])
        self.assertEqual(result['prediction'].tobytes(),expected.tobytes())


if __name__=='__main__':unittest.main()
