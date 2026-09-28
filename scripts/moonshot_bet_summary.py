#!/usr/bin/env python3
"""Score predeclared fixed averages of completed, independently cross-fitted bets."""
import argparse
import hashlib
import json
import pathlib
import numpy as np
from gbdt_blend import load, blend_fold

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',required=True)
p.add_argument('--output',default='experiments/BETS-combinations.json')
a=p.parse_args()
d=load(a.data); y=d['probe']['rating'].astype(np.float64)
fold=blend_fold(d['probe']['user']); predictions={}; hashes={}; results=[]

def read(name,path):
    raw=pathlib.Path(path).read_bytes()
    values=np.frombuffer(raw,dtype='<f4').astype(np.float64)
    if len(values)!=len(y) or not np.isfinite(values).all(): raise ValueError(path)
    hashes[path]=hashlib.sha256(raw).hexdigest(); predictions[name]=values

def score(name,values,components=None):
    error=(np.clip(values,1,5)-y)**2
    rec=dict(name=name,rmse=float(np.sqrt(error.mean())),
             fold_rmse=[float(np.sqrt(error[fold==f].mean())) for f in range(5)])
    if components:
        rec['components']=components; rec['weights']=[1/len(components)]*len(components)
        path='artifacts/'+name+'.f32'; values.astype('<f4').tofile(path)
        rec['output']=path; rec['sha256']=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    results.append(rec)

read('reference','artifacts/UNION-combined-equal.f32')
read('neural','artifacts/UNION-union.f32')
for run in ['BOOST-direct-v2','BOOST-residual']:
    if not pathlib.Path('artifacts/'+run+'.json').exists(): raise ValueError('unfinished: '+run)
    for fit in ['selected','refit']:
        read(run+'-'+fit,'artifacts/'+run+'.'+fit+'.f32')
for name,values in predictions.items(): score(name,values)
for run in ['BOOST-direct-v2','BOOST-residual']:
    for fit in ['selected','refit']:
        components=['neural',run+'-'+fit]
        score('BETS-equal-'+run+'-'+fit,sum(predictions[n] for n in components)/2,components)
components=['neural','BOOST-direct-v2-refit','BOOST-residual-refit']
score('BETS-three-way',sum(predictions[n] for n in components)/3,components)
for name in ['DIST-mean','DIST-probability']:
    if not pathlib.Path('artifacts/'+name+'.json').exists(): raise ValueError('unfinished: '+name)
    read(name,'artifacts/'+name+'.f32'); score(name,predictions[name])
    components=[name,'BOOST-residual-refit']
    score('BETS-equal-'+name,sum(predictions[n] for n in components)/2,components)
report=dict(protocol='Predeclared equal weights; no fitting on CV outputs. Development comparison, not independent test.',
            inputs=hashes,results=results)
pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(results,indent=2))
