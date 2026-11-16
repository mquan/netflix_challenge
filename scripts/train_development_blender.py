#!/usr/bin/env python3
"""Fit/freeze blender models using development inputs only; infer queries later.

This separates training from slow base-model qualifying exports. A models_ready
report is deliberately not an auditable qualifying prediction report.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from fit_qual_blend import solve,nn_features
from gbdt_blend import load,blend_fold
from streaming_blend import digest,predict_ridge
from streaming_nn_blend import standardized,train_device
from feature_adaptation import relative_unrated


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True);ap.add_argument('--features')
    ap.add_argument('--output',required=True);ap.add_argument('--device',choices=['cpu','mps'],default='cpu')
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--seeds',nargs='+',type=int,default=[2026,7,13])
    ap.add_argument('--hidden',type=int,default=128);ap.add_argument('--layers',type=int,default=2)
    ap.add_argument('--ridge-alpha',type=float,default=1e-5)
    ap.add_argument('--nn-only',action='store_true')
    ap.add_argument('--drop-unrated',action='store_true')
    ap.add_argument('--relative-unrated',action='store_true')
    ap.add_argument('--drop-support',action='store_true',help='suppress latent/voting supports, retain generic metadata')
    ap.add_argument('--predictions-only',action='store_true',help='suppress every NN side feature; retain prediction inputs')
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    if a.drop_unrated and a.relative_unrated:raise ValueError('choose only one unrated feature adaptation')
    if a.predictions_only and a.relative_unrated:
        raise ValueError('prediction-only inputs cannot restore an unrated side feature')
    if not a.ridge_alpha>0:raise ValueError('ridge alpha must be positive')
    torch.set_num_threads(a.threads)
    b=json.loads(Path(a.bank+'.inputs.json').read_text())
    if b['config'].get('sample_weights') and not a.nn_only:
        raise ValueError('weighted development currently requires --nn-only')
    if digest(b['config']['data'])!=b['data_sha256'] or digest(b['config']['meta'])!=b['meta_sha256']:
        raise ValueError('development inputs changed')
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);users=d['probe']['user'].copy();del d
    if not np.all((y>=1)&(y<=5)):raise ValueError('development ratings required')
    sample_weight=None;weight_provenance=None
    if b['config'].get('sample_weights'):
        weight_path=b['config']['sample_weights']
        if digest(weight_path)!=b['sample_weights_sha256']:raise ValueError('training weights changed')
        sample_weight=np.load(weight_path,allow_pickle=False)
        if (sample_weight.shape!=y.shape or not np.isfinite(sample_weight).all() or
                np.any(sample_weight<=0)):raise ValueError('invalid training weights')
        weight_report=Path(weight_path).with_suffix('.json')
        if weight_report.exists():
            wr=json.loads(weight_report.read_text())
            if wr.get('weights_sha256')!=b['sample_weights_sha256'] or wr.get('qualifying_labels_read') is not False:
                raise ValueError('unverified weight provenance')
            weight_provenance=dict(path=str(weight_report),sha256=digest(weight_report),kind=wr.get('kind'),
                unlabeled_target_covariates_used=wr.get('kind')=='unlabeled_domain_density_ratio')
    raw=np.load(a.bank+'.x.npy',mmap_mode='r');c=np.load(b['config']['meta'])['context']
    inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    stats_file=np.load(a.bank+'.stats.npz');stats=tuple(stats_file[k] for k in ('gram','rhs','count'))
    report=dict(state='starting',config=vars(a),bank_sha256=digest(a.bank+'.inputs.json'),
                stats_sha256=digest(a.bank+'.stats.npz'),development_rows=len(y),feature_width=raw.shape[1],
                blend_rule=(f'Seed-averaged NN{a.layers} residuals' if a.nn_only else
                            f'Fixed equal average of seed-averaged tree and NN{a.layers} residuals'),
                qualifying_inputs_provided=False,qualifying_labels_read=False,
                sample_weights_sha256=b.get('sample_weights_sha256'),
                sample_weight_provenance=weight_provenance,
                torch_version=torch.__version__,numpy_version=np.__version__)
    start=time.monotonic()

    def status(state,**values):
        report.update(state=state,elapsed_seconds=time.monotonic()-start,**values)
        temp=root/'report.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(root/'report.json')
        print(json.dumps(dict(state=state,**values)),flush=True)

    status('ridge')
    full=solve(stats,np.ones(10,dtype=bool),a.ridge_alpha);iw=solve(stats,np.arange(10)%2==0,a.ridge_alpha)
    np.savez(root/'ridge-weights.npz',full=full,inner=iw)
    base=predict_ridge(raw[:,:b['columns']],c,full);ib=predict_ridge(raw[:,:b['columns']],c,iw)
    np.save(root/'development-base.npy',base,allow_pickle=False)
    np.save(root/'development-inner-base.npy',ib,allow_pickle=False)
    del stats,stats_file
    plan=dict(bank=a.bank,output=str(root),seeds=a.seeds,threads=a.threads)
    (root/'tree-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    tree=dict(tree_rounds=[])
    if not a.nn_only:
        status('tree_worker')
        subprocess.run([sys.executable,str(Path(__file__).with_name('train_tree_models.py')),
                        '--plan',str(root/'tree-plan.json')],check=True)
        tree=json.loads((root/'tree-report.json').read_text())
        if tree['state']!='complete' or tree['torch_loaded']:raise ValueError('invalid isolated tree training')
    status('normalization',tree_rounds=tree['tree_rounds'])
    feature_path=a.features or str(root/'nn-features.npy')
    features=(np.load(feature_path,mmap_mode='r') if a.features else
              nn_features(raw,b['columns'],feature_path))
    rows=np.linspace(0,len(raw)-1,min(256,len(raw)),dtype=int)
    if features.shape!=(len(raw),raw.shape[1]+1) or not np.array_equal(
            features[rows],nn_features(np.array(raw[rows],dtype=np.float32,order='F'),b['columns'])):
        raise ValueError('cached NN encodings differ')
    report['nn_features_sha256']=digest(feature_path)
    zero_columns=[]
    if a.drop_unrated:
        names=b['extra_meta_names']
        zero_columns=[features.shape[1]-len(names)+names.index('log_unrated_n')]
    if a.predictions_only:
        zero_columns+=list(range(b['columns']+1,features.shape[1]))
    elif a.drop_support:
        if not b.get('support'):raise ValueError('bank does not declare support columns')
        zero_columns+=list(range(b['columns']+1,b['columns']+1+b['support']))
    zero_columns=sorted(set(zero_columns))
    report['suppressed_features']=['log_unrated_n'] if a.drop_unrated else []
    report['suppressed_support_count']=b.get('support',0) if (a.drop_support or a.predictions_only) else 0
    report['prediction_only_neural_inputs']=bool(a.predictions_only)
    replacements={}
    if a.relative_unrated:
        index,values=relative_unrated(features,b);replacements[index]=values
    report['relative_unrated_feature']=bool(a.relative_unrated)
    normal,scaler=standardized(features,np.ones(len(y),dtype=bool),
                              output=str(root/'nn-normalized.npy'),return_scaler=True,zero_columns=zero_columns,
                              replacement_columns=replacements)
    np.savez(root/'nn-scaler.npz',**scaler)
    # Full CPU refits gather many random rows: retain a contiguous matrix once.
    training=np.ascontiguousarray(normal) if a.device=='cpu' else normal
    cfg=SimpleNamespace(hidden=a.hidden,layers=a.layers,dropout=.2,lr=.0003,weight_decay=.01,
                        batch=4096,patience=5,device=a.device,resident_features=a.device=='mps')
    epochs_used=[]
    for seed in a.seeds:
        valid=(training[inner],y[inner]-ib[inner])
        if sample_weight is not None:valid=valid+(sample_weight[inner],)
        _,epochs=train_device(training[~inner],y[~inner]-ib[~inner],cfg,seed,60,
                              valid=valid,
                              sample_weights=None if sample_weight is None else sample_weight[~inner])
        status('nn_refit',seed=seed,epochs=epochs)
        model,_=train_device(training,y-base,cfg,seed,epochs,sample_weights=sample_weight)
        torch.save(dict(state=model.state_dict(),config=vars(cfg),seed=seed,epochs=epochs),root/f'nn-{seed}.pt')
        epochs_used.append(epochs);del model
        status('nn_seed_complete',seed=seed,nn_epochs=epochs_used)
    hashes={p.name:digest(p) for p in root.iterdir() if p.suffix in ('.pt','.npz','.txt')}
    status('models_ready',outputs=hashes,nn_epochs=epochs_used)


if __name__=='__main__':main()
