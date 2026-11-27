"""Estimate within-user contrasts using other unlabeled query predictions."""
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_full_context as full
import own_overnight_blend as blend
import own_retained_history as retained
import own_session_latent as latent

OUT=ne.ROOT/'own-query-peers-20261126'
PARENT=blend.OUT/'stage-07'


def features(phase,p,base):
    known=phase=='probex'
    u,i,d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    rep=json.loads((latent.OUT/'graph.json').read_text())
    path=latent.OUT/'graph-coordinates.npy'
    assert ne.sha(path)==rep['embedding_sha256']
    geom=np.load(path)
    geom/=np.maximum(np.linalg.norm(geom,axis=1,keepdims=True),1e-7)
    assert geom.shape[1]==48
    masked=np.load(retained.OUT/(('known' if known else 'qual')+'.npy'),mmap_mode='r')
    rank,cf=ne.verified_rank(phase),full.cf96(phase)
    z=np.column_stack((base,base-p,masked[:,0]-base,masked[:,1:].mean(1)-base,rank-base,cf-base,
                       np.std(masked,axis=1),np.abs(rank-cf))).astype('f')
    qo=np.argsort(u,kind='stable').astype('i')
    n=len(ne.history('trainx','user_cnts'))
    starts=np.r_[0,np.bincount(u,minlength=n).cumsum()].astype('i')
    lib=ctypes.CDLL(str((OUT/'peers.dylib').resolve()))
    lib.peers.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*7
    lib.peers.restype=None
    peer=np.zeros((len(base),4*(z.shape[1]+2)+1),'f')
    lib.peers(n,z.shape[1],*[a.ctypes.data for a in (starts,qo,d,i,geom,z,peer)])
    raw=np.load(full.OUT/(phase+'.npy'),mmap_mode='r')
    ctx=np.column_stack((z,peer,raw[:,::6])).astype('f')
    assert np.isfinite(ctx).all()
    return ctx


def model():
    return CatBoostRegressor(iterations=1100,depth=5,learning_rate=.04,l2_leaf_reg=50,
                             thread_count=2,allow_writing_files=False,random_seed=20261126,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_query_peers.cpp',
                    '-o',str(OUT/'peers.dylib')],check=True)
    y,f,p,q=anchor.history_control()
    report=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==report['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy')
    qb=np.fromfile(PARENT/'selected.f32','<f4')
    x=features('probex',p,base)
    selected,stats,held=anchor.crossfit(x,y,f,base,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,
                known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        qx=features('qual',q,qb)
        delta=model().fit(x,y-base).predict(qx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(qb+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
