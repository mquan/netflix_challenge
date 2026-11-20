#!/usr/bin/env python3
"""Compare actual saved LightGBM streaming inference with full-array inference."""
from pathlib import Path
import tempfile
import numpy as np
import lightgbm as lgb
import upstream_item154_tree_production_recovered as model

def check():
    rng=np.random.default_rng(419);rows=137
    f=rng.normal(size=(rows,397)).astype(np.float32);z=np.zeros((rows,183));z[:,0]=1
    w=np.zeros(183);w[0]=3.6;base=np.clip(z@w,1.05,4.95);x=np.column_stack([f,base]).astype(np.float32)
    y=.1*f[:,0]-.2*f[:,11]+.05*f[:,396]
    plan=dict(seeds=[2026,7,13],iterations={'tree':[7,8,9]},qualifying_rows=rows)
    with tempfile.TemporaryDirectory(prefix='tree154-stream-fixture-') as tmp:
        root=Path(tmp);np.save(root/'ridge.npy',w);plan['ridge_sha256']=model.u.sha(root/'ridge.npy')
        expected=np.zeros(rows)
        for seed,count in zip(plan['seeds'],plan['iterations']['tree']):
            fitted=lgb.train(dict(objective='regression',verbosity=-1,num_threads=2,seed=seed,num_leaves=7,min_data_in_leaf=3,deterministic=True,force_col_wise=True),lgb.Dataset(x,label=y),num_boost_round=count)
            fitted.save_model(str(root/f'tree-{seed}.txt'));reloaded=lgb.Booster(model_file=str(root/f'tree-{seed}.txt'))
            expected+=reloaded.predict(x,num_threads=2)/3
        expected=np.clip(base+expected,1.05,4.95).astype('<f4')
        original=(model.RUN,model.bank.qual_chunk,model.core.CHUNK)
        try:
            model.RUN=root;model.bank.qual_chunk=lambda lo,hi:(z[lo:hi],f[lo:hi])
            for chunk in [4,17,64,137]:
                model.core.CHUNK=chunk;actual=model.infer('tree',plan,root)
                if not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)):raise ValueError('Saved-tree streaming chunk arithmetic differs')
        finally:model.RUN,model.bank.qual_chunk,model.core.CHUNK=original
    report=model.u.ROOT/'experiments/upstream-item154-tree-stream-fixture-recovered';report.mkdir(parents=True,exist_ok=True)
    model.u.write(report/'verification.json',dict(state='complete_actual_saved_tree_stream_fixture',observed=model.binding.stamp(),
        actual_saved_tree_streaming_bitwise_equal=True,rows=rows,tree_feature_width=398,ridge_width=183,seeds=plan['seeds'],chunks=[4,17,64,137],
        code_sha256=model.code_hashes(),scope='Synthetic arithmetic fixture only, three actual saved/reloaded LightGBM models; original full-data production models/configuration/budgets are not reduced.'))
    print('PASS: three actual saved trees,398 features,183 ridge columns and four streaming chunk boundaries')

if __name__=='__main__':check()
