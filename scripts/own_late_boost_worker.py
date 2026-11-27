"""Run LightGBM with its own OpenMP runtime in an isolated process."""
import sys
from pathlib import Path
import numpy as np
import lightgbm as lgb


def fitter(x,y,seed,extra):
    params=dict(objective='regression',learning_rate=.028,num_leaves=22,
        min_child_samples=900,reg_lambda=110,subsample=.80,subsample_freq=1,colsample_bytree=.7,
        extra_trees=extra,max_bin=95,num_threads=2,seed=seed,verbosity=-1,force_col_wise=True)
    return lgb.train(params,lgb.Dataset(x,label=y),num_boost_round=850)


def main(root,phase):
    x=np.load(root/'features.tmp.npy',mmap_mode='r')
    if phase=='known':
        f,e=np.load(root/'folds.npy'),np.load(root/'residuals.npy')
        held=np.zeros(len(f))
        for fold in range(5):
            tr=f!=fold;va=~tr
            for extra in (False,True):
                model=fitter(x[tr],e[tr],1419+fold+int(extra),extra)
                model.save_model(str(root/f'fold-{fold}-{int(extra)}.txt'))
                held[va]+=.5*model.predict(x[va],num_threads=2)
            print('fold',fold,float(np.mean((e[va]-held[va])**2)**.5),flush=True)
        np.save(root/'oof.npy',held)
    else:
        delta=np.zeros(len(x))
        for fold in range(5):
            for extra in (0,1):
                model=lgb.Booster(model_file=str(root/f'fold-{fold}-{extra}.txt'))
                for j in range(0,len(x),65536):
                    sl=slice(j,j+65536);delta[sl]+=.1*model.predict(x[sl],num_threads=2)
        np.save(root/'qual-correction.npy',delta)


if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2])
