#!/usr/bin/env python3
"""Matched full-P2 pilot: rank 16, 40 sweeps, frequency group off versus linear."""
import json
import pathlib
import re
import subprocess
import sys
import numpy as np
from gbdt_blend import load, blend_fold


def main():
    data='artifacts/validation/train-matched-20260927.bin'
    scores={}
    for mode,label in [(0,'control'),(1,'frequency')]:
        prefix='artifacts/r9/B5-pilot-'+label
        cmd=['build/netflix','bayes','--data',data,'--model',prefix+'.bin',
             '--predictions',prefix+'.f32','--factors','16','--samples','40','--burn-in','16',
             '--threads','2','--seed','29','--item-freq',str(mode),'--checkpoint-every','8']
        subprocess.run([sys.executable,'scripts/experiment.py','--name','R9-B5-'+label,
                        '--strategy','Matched full-P2 Bayesian frequency pilot, mode '+str(mode),'--',*cmd],check=True)
        d=load(data); y=d['probe']['rating'].astype('f8'); f=blend_fold(d['probe']['user'])
        pred=np.clip(np.fromfile(prefix+'.f32',dtype='<f4').astype('f8'),1,5)
        scores[label]=dict(rmse=float(np.sqrt(np.mean((pred-y)**2))),
                           folds=[float(np.sqrt(np.mean((pred[f==k]-y[f==k])**2))) for k in range(5)])
    scores['delta']=scores['frequency']['rmse']-scores['control']['rmse']
    scores['fold_delta']=[a-b for a,b in zip(scores['frequency']['folds'],scores['control']['folds'])]
    scores['note']='P2 pilot only; no ensemble gain or promotion established'
    pathlib.Path('artifacts/r9/bayes-pilot-comparison.json').write_text(json.dumps(scores,indent=2)+'\n')
    print(json.dumps(scores),flush=True)


if __name__=='__main__': main()
