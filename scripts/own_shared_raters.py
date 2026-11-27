"""Retrieve comparable users by weighted movie and star histories, then measure their actual shared ratings."""
import ctypes
import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_anchor_context as anchor
import own_full_context as full
import own_transition as transition
import own_retained_history as ret
OUT=ne.ROOT/'own-shared-raters-20261126'
PARENT=blend.OUT/'schema-01'


def data(train,phase,base,lib):
    e,_=transition.loo(train)
    i,r,d=[ne.history(train,k) for k in ('item_idxs','ratings','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    nu,nm=len(uc),len(ic);off=np.r_[0,uc.cumsum()].astype('i')
    u,m,dd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    qu=np.argsort(u,kind='stable').astype('i');count=np.bincount(u,minlength=nu);assert count.max()<=12
    qo=np.r_[0,count.cumsum()].astype('i')
    mu=((np.bincount(i,weights=r,minlength=nm)+30*3.6)/(ic+30)).astype('f')
    rarity=(1/(np.log1p(nu/(20+ic))+.3)).astype('f')
    sig=np.empty((12,nu),'u8')
    ptr=lambda *args:[a.ctypes.data for a in args]
    lib.signatures(nu,*ptr(off,i,r,e,rarity,sig));idx=np.sort(sig,axis=1)
    result=np.zeros((len(u),40),'f');args=ptr(off,i,r,d,e,mu,rarity,qo,qu,m,dd,sig,idx,result)
    def chunk(j):lib.neighbors(j,min(nu,j+512),nu,nm,*args)
    with ThreadPoolExecutor(2) as pool:
        for j,_ in enumerate(pool.map(chunk,range(0,nu,512))):
            if j%180==0:print(train,'users',j*512,flush=True)
    assert np.isfinite(result).all()
    return np.column_stack((result,base,full.cf96(phase)-base,ne.verified_rank(phase)-base,
                            np.log1p(uc[u]),np.log1p(ic[m]),dd/2000)).astype('f')


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_shared_raters.cpp','-o',str(OUT/'raters.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'raters.dylib').resolve()));it,pt=ctypes.c_int,ctypes.c_void_p
    lib.signatures.argtypes=[it]+[pt]*6;lib.neighbors.argtypes=[it]*4+[pt]*14
    lib.signatures.restype=lib.neighbors.restype=None
    y,f,_,_=anchor.history_control();rec=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    x=data('trainx','probex',base,lib)
    def model():return CatBoostRegressor(iterations=500,depth=5,learning_rate=.045,l2_leaf_reg=70,
            thread_count=2,random_seed=1279,verbose=False,allow_writing_files=False)
    choice,stats,held=anchor.crossfit(x,y,f,base,model)
    report=dict(parent=str(PARENT),selected=choice,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if choice!='control':
        qx=data('fulltrain','qual',qb,lib);delta=model().fit(x,y-base).predict(qx,thread_count=2)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');report['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
