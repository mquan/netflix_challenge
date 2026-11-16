#!/usr/bin/env python3
"""Development-only symmetric-tree residual blending, separate query inference."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import catboost
from catboost import CatBoostRegressor
from blend_features import nn_features
from gbdt_blend import load,blend_fold
from streaming_blend import digest
from qual_matrix import verified_matrix,predict_matrix_ridge


def write_report(root,report):
    tmp=root/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(root/'report.json')


def verified_neural_source(path):
    root=Path(path);m=json.loads((root/'report.json').read_text())
    if m['state']!='models_ready' or m['qualifying_labels_read'] is not False or m['qualifying_inputs_provided'] is not False:
        raise ValueError('expected development-only frozen neural models')
    if not m['config'].get('nn_only'):raise ValueError('reference must be neural-only')
    for name,h in m['outputs'].items():
        if digest(root/name)!=h:raise ValueError('neural source changed')
    bank=m['config']['bank'];b=json.loads(Path(bank+'.inputs.json').read_text())
    if digest(bank+'.inputs.json')!=m['bank_sha256']:raise ValueError('source bank changed')
    return root,m,bank,b


def train(a):
    source,m,bank,b=verified_neural_source(a.neural_models)
    if digest(b['config']['data'])!=b['data_sha256'] or digest(b['config']['meta'])!=b['meta_sha256']:
        raise ValueError('development data changed')
    features=np.load(a.features,mmap_mode='r');raw=np.load(bank+'.x.npy',mmap_mode='r')
    if digest(a.features)!=m['nn_features_sha256']:raise ValueError('frozen feature encoding changed')
    rows=np.linspace(0,len(raw)-1,min(256,len(raw)),dtype=int)
    if features.shape!=(len(raw),raw.shape[1]+1) or not np.array_equal(features[rows],nn_features(np.array(raw[rows],dtype=np.float32,order='F'),b['columns'])):
        raise ValueError('feature encoding differs')
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);users=d['probe']['user'].copy();del d
    inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    base=np.load(source/'development-base.npy');ib=np.load(source/'development-inner-base.npy')
    c=np.load(b['config']['meta'])['context'];w=np.load(source/'ridge-weights.npz')
    if not np.allclose(base,predict_matrix_ridge(raw,c,b['columns'],w['full']),rtol=0,atol=1e-10) or not np.allclose(ib,predict_matrix_ridge(raw,c,b['columns'],w['inner']),rtol=0,atol=1e-10):
        raise ValueError('cached ridge baselines differ from frozen weights')
    ignored=[features.shape[1]-len(b['extra_meta_names'])+b['extra_meta_names'].index('log_unrated_n')]
    params=dict(loss_function='RMSE',depth=6,learning_rate=.03,l2_leaf_reg=10,rsm=.25,
                bootstrap_type='Bernoulli',subsample=.8,border_count=128,thread_count=a.threads,
                ignored_features=ignored,verbose=100)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    report=dict(state='training',config=vars(a),bank=bank,bank_sha256=m['bank_sha256'],
        neural_models=str(source),neural_report_sha256=digest(source/'report.json'),
        feature_encoding_sha256=digest(a.features),qualifying_inputs_provided=False,qualifying_labels_read=False,
        catboost_version=catboost.__version__,params=params,seeds=a.seeds,rounds=[],outputs={},torch_loaded='torch' in sys.modules)
    if report['torch_loaded']:raise ValueError('run CatBoost in an isolated non-Torch process')
    write_report(root,report);start=time.monotonic()
    for seed in a.seeds:
        model=CatBoostRegressor(**params,iterations=a.max_rounds,random_seed=seed,train_dir=str(root/f'inner-{seed}'))
        model.fit(features[~inner],y[~inner]-ib[~inner],eval_set=(features[inner],y[inner]-ib[inner]),
            early_stopping_rounds=a.patience,use_best_model=True,save_snapshot=True,
            snapshot_file=str((root/f'inner-{seed}.snapshot').resolve()),snapshot_interval=300)
        rounds=model.get_best_iteration()+1
        if not 1<=rounds<=a.max_rounds:raise ValueError('invalid stopping iteration')
        model.save_model(str(root/f'inner-{seed}.cbm'));del model;gc.collect()
        report.update(state='refitting',seed=seed,selected_rounds=rounds);write_report(root,report)
        model=CatBoostRegressor(**params,iterations=rounds,random_seed=seed,train_dir=str(root/f'full-{seed}'))
        model.fit(features,y-base,save_snapshot=True,snapshot_file=str((root/f'full-{seed}.snapshot').resolve()),snapshot_interval=300)
        if model.tree_count_!=rounds or np.any(model.feature_importances_[ignored]!=0):
            raise ValueError('iteration budget or ignored feature violated')
        path=root/f'cat-{seed}.cbm';model.save_model(str(path));del model;gc.collect()
        report['rounds'].append(rounds);report['outputs'][path.name]=digest(path)
        report.update(state='seed_complete',elapsed_seconds=time.monotonic()-start);write_report(root,report)
    report.update(state='models_ready');write_report(root,report)


def predict(a):
    root=Path(a.models);m=json.loads((root/'report.json').read_text())
    if m['state']!='models_ready' or m['qualifying_labels_read'] is not False or m['qualifying_inputs_provided'] is not False:
        raise ValueError('CatBoost models not frozen')
    for name,h in m['outputs'].items():
        if digest(root/name)!=h:raise ValueError('CatBoost model changed')
    source,n,bank,b=verified_neural_source(m['neural_models'])
    if digest(source/'report.json')!=m['neural_report_sha256']:raise ValueError('neural source report changed')
    q=json.loads(Path(a.qual+'.json').read_text())
    if q['state']!='complete' or q['bank_sha256']!=m['bank_sha256'] or q['qualifying_labels_read'] is not False:
        raise ValueError('unverified qualifying feature schema')
    if digest(a.qual+'.context.npy')!=q['context_sha256']:raise ValueError('query context changed')
    raw=verified_matrix(a.qual,q);c=np.load(a.qual+'.context.npy',mmap_mode='r')
    base=predict_matrix_ridge(raw,c,b['columns'],np.load(source/'ridge-weights.npz')['full'])
    models=[]
    for seed in m['seeds']:
        model=CatBoostRegressor();model.load_model(str(root/f'cat-{seed}.cbm'));models.append(model)
    out=np.zeros(len(raw))
    for lo in range(0,len(raw),65536):
        hi=min(lo+65536,len(raw));f=nn_features(np.array(raw[lo:hi],dtype=np.float32,order='F'),b['columns'])
        for model in models:out[lo:hi]+=model.predict(f,thread_count=a.threads)/len(models)
    cat=base+out;reference=Path(a.neural_reference);nr=json.loads((reference/'report.json').read_text())
    if nr['state']!='complete' or nr['qualifying_labels_read'] is not False or nr['models_report_sha256']!=m['neural_report_sha256']:
        raise ValueError('neural prediction reference differs')
    if nr['qual_manifest_sha256']!=digest(a.qual+'.json'):raise ValueError('neural query matrix differs')
    if digest(reference/'qualifying-final.f32')!=nr['frozen_prediction_sha256']:raise ValueError('neural final changed')
    neural=np.fromfile(reference/'nn.f32',dtype='<f4')
    if len(neural)!=len(cat) or not np.isfinite(neural).all() or not np.array_equal(np.clip(neural,1,5),np.fromfile(reference/'qualifying-final.f32',dtype='<f4')):
        raise ValueError('raw neural prediction differs from verified final')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    # Freeze BOTH rules before any audit is permitted by the outer pipeline.
    for name,values in [('cat-only',cat),('hybrid',(cat+neural.astype(np.float64))/2)]:
        path=output/name;path.mkdir();final=np.clip(values,1,5).astype('<f4')
        if not np.isfinite(final).all():raise ValueError('nonfinite predictions')
        final.tofile(path/'qualifying-final.f32')
        report=dict(state='complete',qualifying_labels_read=False,qualifying_rows=len(final),metric=None,
            config=vars(a),catboost_report_sha256=digest(root/'report.json'),qual_manifest_sha256=digest(a.qual+'.json'),
            neural_reference_report_sha256=digest(reference/'report.json'),raw_neural_sha256=digest(reference/'nn.f32'),
            rule='Seed-averaged CatBoost residuals' if name=='cat-only' else 'Fixed equal average of CatBoost and NN2 predictions, clipped after averaging',
            frozen_prediction_sha256=digest(path/'qualifying-final.f32'))
        write_report(path,report)
    write_report(output,dict(state='complete',qualifying_labels_read=False,variants=['cat-only','hybrid']))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('train');t.add_argument('--neural-models',required=True);t.add_argument('--features',required=True)
    t.add_argument('--output',required=True);t.add_argument('--threads',type=int,default=2);t.add_argument('--seeds',type=int,nargs='+',default=[2026,7,13])
    t.add_argument('--max-rounds',type=int,default=4000);t.add_argument('--patience',type=int,default=100)
    p=sub.add_parser('predict');p.add_argument('--models',required=True);p.add_argument('--qual',required=True)
    p.add_argument('--neural-reference',required=True);p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();(train if a.mode=='train' else predict)(a)


if __name__=='__main__':main()
