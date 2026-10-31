#!/usr/bin/env python3
"""Fit the fixed H8 CPU column or cache label-free query-view features.

Training uses P2 labels only. Inference takes prediction/feature arrays, never
query ratings. The matching N1 full-data base column is supplied after its
existing queue completes. There is no epoch/feature/parameter search here.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from gbdt_blend import blend_fold,load
from headroom_choices import features,read_joint_factors
from headroom_map import crossfit
from train_holdout import digest

SOURCES = ('choice_affinity_column.py','headroom_choices.py','headroom_map.py',
           'gbdt_blend.py','local_context_chain.py','train_holdout.py')


def fit_models(bank,choices,y,users,params,iterations):
    if bank.ndim != 2 or choices.ndim != 2 or len(bank) != len(y) or len(choices) != len(y):
        raise ValueError('training shapes differ')
    if not all(np.isfinite(v).all() for v in (bank,choices,y)):
        raise ValueError('nonfinite training input')
    if set(iterations) != {'base_tree','choice_tree'} or min(iterations.values()) < 1:
        raise ValueError('fixed positive round counts required')
    folds=blend_fold(users)
    if not all(np.any(folds==f) for f in range(5)):
        raise ValueError('all five training folds required')
    base=crossfit(bank,y,folds,penalty=10.)
    design=np.column_stack([np.ones(len(y)),bank])
    weights=np.linalg.solve(design.T@design+10.*np.eye(design.shape[1]),design.T@y)
    models={}
    for name,extra in [('base_tree',choices[:,:0]),('choice_tree',choices)]:
        x=np.column_stack([base,extra])
        models[name]=lgb.train(params,lgb.Dataset(x,y-base),int(iterations[name]))
    return weights,models


def predict_models(bank,choices,weights,models):
    if (bank.ndim != 2 or choices.ndim != 2 or len(bank) != len(choices)
            or weights.shape != (bank.shape[1]+1,)):
        raise ValueError('query column shapes differ')
    if not all(np.isfinite(v).all() for v in (bank,choices,weights)):
        raise ValueError('nonfinite query input')
    base=np.column_stack([np.ones(len(bank)),bank])@weights
    result={}
    for name,extra in [('base_tree',choices[:,:0]),('choice_tree',choices)]:
        result[name]=base+models[name].predict(np.column_stack([base,extra]),num_threads=models[name].params.get('num_threads',1))
        if not np.isfinite(result[name]).all():
            raise FloatingPointError('nonfinite column output')
    return result


def load_bundle(directory,bank_names,feature_names):
    directory=Path(directory)
    report=json.loads((directory/'manifest.json').read_text())
    if (report['state'] != 'complete' or report['mode'] != 'train'
            or report['bank_names'] != bank_names or report['feature_names'] != feature_names):
        raise ValueError('bundle schema differs from query columns')
    for filename,h in report['files'].items():
        if digest(directory/filename) != h:
            raise ValueError('bundle payload changed')
    weights=np.load(directory/'ridge.npy')
    models={name:lgb.Booster(model_file=str(directory/(name+'.txt')))
            for name in ('base_tree','choice_tree')}
    return weights,models


def run(plan_path,output):
    plan_path,output=Path(plan_path),Path(output)
    p=json.loads(plan_path.read_text())
    if p['mode'] not in ('train','features'):
        raise ValueError('unknown column preparation mode')
    if output.exists():
        raise FileExistsError(output)
    required=[p['data'],p['joint_model']]
    required += [str(Path(__file__).resolve().with_name(name)) for name in SOURCES]
    if p['mode']=='train':
        required += [p['selection_result'],*p['bank']]
    if not set(required) <= set(p['frozen']):
        raise ValueError('column plan lacks input/source bindings')
    def verify():
        for path,h in p['frozen'].items():
            if digest(path)!=h:
                raise ValueError('frozen column input changed: '+path)
    verify()
    if p['libraries'] != dict(numpy=np.__version__,lightgbm=lgb.__version__):
        raise ValueError('column library version changed')
    d=load(p['data'])
    q,emb=read_joint_factors(p['joint_model'],d['items'])
    # The feature producer receives a new array with identity/date fields only.
    # Query rating bytes are unavailable through this interface.
    identities=np.empty(len(d['probe']),dtype=[('user','<u4'),('item','<u2'),('day','<u2')])
    for key in identities.dtype.names:identities[key]=d['probe'][key]
    choices,names=features(dict(d,probe=identities),q,emb)
    if not np.isfinite(choices).all():
        raise ValueError('nonfinite choice features')
    if names != p['feature_names']:
        raise ValueError('column feature schema changed')
    report=dict(state='complete',mode=p['mode'],plan_sha256=digest(plan_path),
        data_sha256=p['frozen'][p['data']],joint_sha256=p['frozen'][p['joint_model']],
        feature_names=names,rows=len(identities),libraries=p['libraries'])
    if p['mode']=='train':
        selected=json.loads(Path(p['selection_result']).read_text())
        if selected['selected_iterations']!=p['iterations']:
            raise ValueError('fixed rounds differ from nested selection')
        if len(p['bank_names'])!=len(p['bank']) or len(set(p['bank_names']))!=len(p['bank']):
            raise ValueError('unique ordered bank names required')
        bank=np.column_stack([np.fromfile(path,dtype='<f4').astype(float) for path in p['bank']])
        weights,models=fit_models(bank,choices,d['probe']['rating'].astype(float),
                                  d['probe']['user'],p['tree_params'],p['iterations'])
        report.update(bank_names=p['bank_names'],iterations=p['iterations'],tree_params=p['tree_params'],
                      scope='P2-trained fixed model bundle. No original-probe or hidden-test score.')
    else:
        report['scope']='Feature generation only; query rating fields were not passed to the producer.'
    verify()
    output.mkdir(parents=True)
    if p['mode']=='train':
        np.save(output/'ridge.npy',weights)
        for name,model in models.items():model.save_model(str(output/(name+'.txt')))
        files=['ridge.npy','base_tree.txt','choice_tree.txt']
    else:
        np.save(output/'features.npy',choices)
        files=['features.npy']
    report['files']={name:digest(output/name) for name in files}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.plan,args.output)
