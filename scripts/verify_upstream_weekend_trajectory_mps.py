#!/usr/bin/env python3
"""Full independent complete-trajectory operators, eligible statistics and actual NN replay."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import gc
import math
from pathlib import Path
import numpy as np
import upstream_weekend_trajectory_mps as study
import verify_upstream_weekend_trajectory_data as operators

u,n,data,mps=study.u,study.n,study.data,study.mps


def score(values,labels):
    return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(np.asarray(values,dtype=np.float64),1.05,4.95),labels))/len(labels))


def independent_scaler(raw,eligible):
    # Independently use complete individual columns and full-column prefix
    # accumulation, rather than the fitting helper's bounded row blocks.
    if raw.dtype!=np.float32 or not 618<=raw.shape[1]<=813 or not raw.flags.c_contiguous:raise ValueError('Complete dynamically expanded original float32 matrix required')
    lo=np.empty(raw.shape[1]);hi=np.empty(raw.shape[1]);mean=np.empty(raw.shape[1]);std=np.empty(raw.shape[1]);count=int(eligible.sum())
    if count<=0:raise ValueError('Complete nonempty original eligibility required')
    for column in range(raw.shape[1]):
        values=raw[eligible,column];lo[column],hi[column]=np.percentile(values,[1,99])
        clipped=np.clip(values.astype(np.float64),lo[column],hi[column]);clipped[0]+=np.float64(0.)
        total=np.add.accumulate(clipped)[-1];mean[column]=total/count
        squared=np.clip(values.astype(np.float64),lo[column],hi[column])-mean[column]
        np.multiply(squared,squared,out=squared);squared[0]+=np.float64(0.)
        std[column]=max(float(np.sqrt(np.add.accumulate(squared)[-1]/count)),1e-6)
    return dict(lo=lo,hi=hi,mean=mean,std=std)


def main(variant):
    import torch
    run,report=study.paths(variant);cfg=study.CONFIGS[variant];output=report/'independent-verification.json'
    if output.exists() or (run/'nn.f32').exists():raise ValueError('Existing complete/partial actual neural replay retained')
    manifest=u.read(report/'manifest.json')
    if (u.read(report/'fit-completion.json')['state']!='complete_all_five_original_budget_complete_trajectory_neural_fits' or
        manifest['config']!=cfg or manifest['code_sha256']!=study.hashes() or manifest['versions']!=n.versions() or
        not manifest['matched_B_known_and_qualifying']):raise ValueError('Original complete matched neural recipe and full fits required')
    data.verify_bindings(manifest['source_sha256']);m=u.read(data.REPORT/'manifest.json');mps.MODEL_FACTORY=study.model_factory
    parent_meta=u.read(data.parent.REPORT/'manifest.json')
    parent=operators.legacy.Independent('probex',parent_meta)
    width=study.dimensions();design=m['design_columns']
    labels=np.load(run/'labels.npy');groups=np.load(run/'groups.npy');users=np.load(run/'users.npy');order=np.argsort(users,kind='stable')
    if (labels.shape!=(1128164,) or set(np.unique(groups))!=set(range(10)) or
        np.any((users[order][1:]==users[order][:-1])&(groups[order][1:]//2!=groups[order][:-1]//2))):raise ValueError('Complete canonical held-user folds required')
    gram=np.zeros((10,design,design));rhs=np.zeros((10,design))
    def block(lo,hi):
        z,f=parent.block(lo,hi)
        return operators.extend(z,f,m['retained'],'probex',lo,hi)
    for lo in range(0,len(labels),8192):
        hi=min(lo+8192,len(labels));z,_=block(lo,hi)
        for group in range(10):
            keep=groups[lo:hi]==group;a=z[keep];gram[group]+=a.T@a;rhs[group]+=a.T@labels[lo:hi][keep]
    with np.load(run/'stats.npz') as saved:
        if not np.array_equal(gram.view('u8'),saved['gram'].view('u8')) or not np.array_equal(rhs.view('u8'),saved['rhs'].view('u8')):
            raise ValueError('Every complete original group statistic bit must match')
    prediction=np.empty(len(labels),dtype='<f4');records=[]
    for fold in range(5):
        root=run/'nn'/str(fold);receipt_path=report/'nn'/('%s.json'%fold);receipt=u.read(receipt_path);counts=receipt['chosen_iterations']
        if (receipt['manifest_sha256']!=data.ops.sha(report/'manifest.json') or receipt['fold']!=fold or
            len(counts)!=3 or any(type(c)!=int or not 0<=c<=40 for c in counts)):
            raise ValueError('Every original known-selected seed budget required')
        expected={'ridge.npz','scalers.npz','prediction.f32'}|{'nn-%s.pt'%s for s,c in zip(cfg['seeds'],counts) if c}
        if set(receipt['outputs'])!=expected or {p.name for p in root.iterdir()}!=expected:raise ValueError('Actual saved original neural inventory differs')
        for name,digest in receipt['outputs'].items():
            if data.ops.sha(root/name)!=digest:raise ValueError('Actual saved model/scaler/prediction changed')
        train=groups//2!=fold;held=~train;fit=train&(groups%2==0);eligible=np.arange(10)//2!=fold
        penalty=np.eye(design)*cfg['ridge_lambda'];penalty[-1,-1]=0
        selectors=[eligible&(np.arange(10)%2==0),eligible]
        weights=[np.linalg.solve(gram[s].sum(0)+penalty,rhs[s].sum(0)) for s in selectors]
        with np.load(root/'ridge.npz') as saved:
            if not np.array_equal(weights[0].view('u8'),saved['inner'].view('u8')) or not np.array_equal(weights[1].view('u8'),saved['full'].view('u8')):
                raise ValueError('All actual original eligible ridge coefficient bits must match')
        full_held=None;baseline=None
        for ident,w,mask in [('inner',weights[0],fit),('full',weights[1],train)]:
            raw=np.empty((len(labels),width),dtype=np.float32);base=np.empty(len(labels))
            for lo in range(0,len(labels),8192):
                hi=min(lo+8192,len(labels));z,f=block(lo,hi);base[lo:hi]=np.clip(z@w,1.05,4.95)
                raw[lo:hi,:-1]=f;raw[lo:hi,-1]=base[lo:hi]
            scaler=independent_scaler(raw,mask)
            with np.load(root/'scalers.npz') as saved:
                if any(not np.array_equal(value.view('u8'),saved[ident+'_'+key].view('u8')) for key,value in scaler.items()):
                    raise ValueError('Every complete eligible percentile/ordered mean/std bit must match actual scaler')
            if ident=='full':full_held=np.empty((int(held.sum()),width),dtype=np.float32);baseline=base[held].copy()
            cursor=0
            for lo in range(0,len(labels),8192):
                hi=min(lo+8192,len(labels));reference=np.asarray((np.clip(raw[lo:hi],scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std'],dtype=np.float32,order='C')
                actual=study.scaling.apply_scaler(raw[lo:hi],scaler)
                if not np.array_equal(reference.view('u4'),actual.view('u4')):raise ValueError('Every original normalized input bit must match')
                if ident=='full':
                    keep=held[lo:hi];count=int(keep.sum());full_held[cursor:cursor+count]=reference[keep];cursor+=count
            if ident=='full' and cursor!=len(full_held):raise ValueError('All held canonical rows required')
            del raw,base;gc.collect()
        correction=np.zeros(int(held.sum()));reopened=0
        for seed,count in zip(cfg['seeds'],counts):
            if not count:continue
            path=root/('nn-%s.pt'%seed);saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['epochs']!=count or saved['seed']!=seed+fold or saved['feature_width']!=width or saved['config']!=cfg:
                raise ValueError('Original actual selected seed/epoch/width/config differs')
            model=mps.load_model(path,cfg);correction+=mps.predict(model,full_held)/3
            del model,saved;torch.mps.empty_cache();reopened+=1
        replay=np.clip(baseline+correction,1.05,4.95).astype('<f4');actual=np.fromfile(root/'prediction.f32',dtype='<f4')
        if not np.array_equal(replay.view('u4'),actual.view('u4')) or receipt['canonical_rows']!=len(replay):raise ValueError('Every complete saved-model held prediction must replay')
        prediction[held]=replay
        records.append(dict(fold=fold,rows=len(replay),actual_models_reopened=reopened,receipt_path=str(receipt_path.relative_to(u.ROOT)),
            receipt_sha256=data.ops.sha(receipt_path),outputs=receipt['outputs'],both_complete_eligible_scalers_independently_recomputed=True,
            all_normalized_input_bits_equal=True,actual_saved_pipeline_bitwise_equal=True,rmse=score(replay,labels[held])))
        del full_held,baseline;gc.collect();torch.mps.empty_cache();n.event(stage='independent_complete_trajectory_fold_verified',variant=variant,**records[-1])
    parent.close();prediction.tofile(run/'nn.f32')
    tree=manifest['controls'][0];tree_p=np.fromfile(u.ROOT/tree['prediction'],dtype='<f4')
    if tree_p.shape!=prediction.shape or not np.isfinite(tree_p).all():raise ValueError('Complete original tree154 OOF control required')
    half=np.clip(.5*prediction.astype(np.float64)+.5*tree_p.astype(np.float64),1.05,4.95).astype('<f4');half.tofile(run/'fixed-half-tree154.f32')
    data.verify_bindings(manifest['source_sha256'])
    for row in records:
        if data.ops.sha(u.ROOT/row['receipt_path'])!=row['receipt_sha256']:raise ValueError('Actual fitted receipt changed during replay')
    u.write(output,dict(state='complete_independent_complete_trajectory_actual_models_full_eligible_statistics_and_predictions',
        variant=variant,manifest_sha256=data.ops.sha(report/'manifest.json'),verifier_sha256=data.ops.sha(Path(__file__)),records=records,
        complete_original_statistics_bitwise_equal=True,all_original_models_preserved_without_refitting=True,
        matched_B_known_and_qualifying=True,nn_rmse=score(prediction,labels),fixed_half_tree154_rmse=score(half,labels),
        nn_sha256=data.ops.sha(run/'nn.f32'),fixed_half_sha256=data.ops.sha(run/'fixed-half-tree154.f32'),
        declared_global_reference_rmse=manifest['declared_global_reference_rmse'],
        fixed_half_below_declared_global_reference=score(half,labels)<manifest['declared_global_reference_rmse'],
        qualifying_labels_read=False,production_started=False,goal_complete=False))
    print('Complete matched391',variant,'NN',score(prediction,labels),'fixed tree154 half',score(half,labels),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=study.CONFIGS,default='h128');args=parser.parse_args();main(args.variant)
