"""Learn conditional user rating distributions around the target's two own-model forecasts."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_full_context as full
import own_history_context as hist
from own_kernel_family import control

OUT=ne.ROOT/'own-ordinal-profile-20261126'


def features(train,query,p):
    u,d=[ne.history(query,k) for k in ('user_idxs','dates')]
    qr,qc=ne.verified_rank(query),full.cf96(query)
    off=np.r_[0,ne.history(train,'user_cnts').cumsum()].astype('i')
    td,stars=[ne.history(train,k) for k in ('dates','ratings')]
    rank,cf=ne.verified_rank(train),full.cf96(train)
    profile=np.empty((len(u),78),'f')
    lib=ctypes.CDLL(str((OUT/'profile.dylib').resolve()))
    lib.ordinal_context.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*10
    lib.ordinal_context.restype=None
    args=[a.ctypes.data for a in (u,d,qr,qc,off,td,stars,rank,cf,profile)]
    def block(lo):lib.ordinal_context(lo,min(lo+8192,len(u)),*args)
    with ThreadPoolExecutor(2) as ex:list(ex.map(block,range(0,len(u),8192)))
    h=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    x=np.column_stack((p,qr-p,qc-p,h[:,2:5],h[:,7],h[:,13],profile)).astype('f')
    assert np.isfinite(x).all()
    np.save(OUT/(query+'.npy'),x)
    return x


def model():
    return CatBoostRegressor(iterations=950,depth=5,learning_rate=.035,l2_leaf_reg=50,
                             thread_count=2,random_seed=20261126,allow_writing_files=False,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC','scripts/own_ordinal_profile.cpp','-o',str(OUT/'profile.dylib')],check=True)
    y,f,p,q=control()
    x=features('trainx','probex',p)
    selected,stats,held=anchor.crossfit(x,y,f,p,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        qx=features('fulltrain','qual',q)
        delta=model().fit(x,y-p).predict(qx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
