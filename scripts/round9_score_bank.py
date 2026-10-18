#!/usr/bin/env python3
"""Score one new column against the frozen promoted bank, then compare folds."""
import argparse
import json
import pathlib
import sys
from round9_latent_run import run


def score(name, prediction):
    c=json.loads(pathlib.Path('artifacts/r8/R8-B4-A2e-s3.json').read_text())['config']
    c['output']='artifacts/r9/'+name+'-s3';c['extra']=[*c['extra'],prediction];c['processes']=3
    args=['scripts/round4_boost.py']
    for key,value in c.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:args.append(flag)
        elif isinstance(value,list):args += [flag,*map(str,value)]
        else:args += [flag,str(value)]
    run(name+'-s3','Frozen promoted bank plus '+prediction,args)
    out='artifacts/r9/'+name+'-NN2'
    run(name+'-NN2','Fixed NN2 recipe and 50/50 average',
        ['scripts/round8_nn_blend.py','--data',c['data'],'--like',c['output']+'.json',
         '--output',out,'--lr','3e-4','--dropout','.2','--seeds','2026','7','13',
         '--epochs','60','--patience','5','--threads','2'])
    baseline=json.loads(pathlib.Path('artifacts/r8/R8-B4-A2e-NN2.json').read_text())
    result=json.loads(pathlib.Path(out+'.json').read_text())
    delta=[a-b for a,b in zip(result['avg_fold_rmse'],baseline['avg_fold_rmse'])]
    report=dict(control=baseline['avg_rmse'],treatment=result['avg_rmse'],
                delta=result['avg_rmse']-baseline['avg_rmse'],fold_delta=delta,promote=all(x<0 for x in delta))
    pathlib.Path('artifacts/r9/'+name+'-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--prediction',required=True)
    a=p.parse_args();score(a.name,a.prediction)
