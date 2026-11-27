"""Re-encode epoch-95 CFNADE histories with temporal and movie-affinity weights."""
import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_full_context as full
import own_anchor_context as anchor
import own_history_context as hist
import own_overnight_blend as blend
import own_session_latent as latent
from own_kernel_family import gram

OUT=ne.ROOT/'own-cf-context-20261126'
UP=Path('../netflix_upstream_repro')
CK=UP/'artifacts/upstream-cf96-training-export/checkpoints'
MAN=UP/'experiments/upstream-cf96-complete-bank/manifest.json'


def checkpoint(train,prefix=''):
    assert ne.sha(MAN)=='be259a6f49f1b6d2fa2dca505988a9dc343fa7af1e467aa2c8c8a9a66a3c47c3'
    sig=json.loads(MAN.read_text())['source_sha256']
    root=CK/train/'epoch-00095'
    upstream=Path('artifacts/upstream-cfnade-checkpoint/parallel-checkpoints')/train/'epoch-00095'
    state=json.loads((root/'state.json').read_text())
    assert ne.sha(root/'state.json')==sig[str(upstream/'state.json')]
    assert state['epoch']==95 and state['swapped_in'] and state['identity']['train']==train
    result={}
    for key in ('a','p','enc_proj','dec_proj','enc_bias','b','user_bias','user_drift'):
        filename=prefix+key+'.npy';path=root/filename
        assert ne.sha(path)==sig[str(upstream/filename)]
        result[key]=np.load(path)
    return result


def decode(ctx,s,m,u,dev):
    n,modes,_=ctx.shape
    v=ctx.reshape(-1,96)
    h=np.tanh(v@s['enc_proj'].T+s['enc_bias'])
    z=(h@s['dec_proj'].T).reshape(n,modes,96)
    bias=(s['b'][:,m]+s['user_bias'][:,u]+s['user_drift'][:,u]*dev).T
    logits=np.einsum('nmk,tnk->nmt',z,s['p'][:,m])+bias[:,None,:]
    logits=np.cumsum(logits,axis=2)
    logits-=logits.max(2,keepdims=True)
    probs=np.exp(logits)
    probs/=probs.sum(2,keepdims=True)
    means=probs@np.arange(1,6,dtype='f')
    return probs,means


def features(train,phase,base):
    lib=ctypes.CDLL(str((OUT/'cf.dylib').resolve()))
    lib.cf_context.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*13
    lib.cf_context.restype=None
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    ti,td,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    s=checkpoint(train)
    a=np.ascontiguousarray(np.cumsum(s['a'],axis=0,dtype=float),dtype='f')
    co=gram(s['p'][2:].mean(0))
    path=latent.OUT/'graph-coordinates.npy'
    assert ne.sha(path)==json.loads((latent.OUT/'graph.json').read_text())['embedding_sha256']
    sg=gram(np.load(path))
    date_mean=np.float32(td.mean())
    day_sq=np.float32(sum(np.dot(td[j:j+1_000_000].astype(float),td[j:j+1_000_000].astype(float)) for j in range(0,len(td),1_000_000))/len(td))
    std=np.sqrt(day_sq-date_mean**2)
    expected=full.cf96(phase)
    args=[x.ctypes.data for x in (qu,qi,qd,off,ti,td,r,a,co,sg)]
    out=np.empty((len(qi),42),'f')
    maxerr=0.
    for j in range(0,len(qi),2048):
        sl=slice(j,min(j+2048,len(qi)));n=len(qi[sl])
        ctx=np.zeros((n,6,96),'f');effective=np.empty((n,5),'f');mday=np.empty(n,'f')
        lib.cf_context(j,j+n,len(ic),*args,ctx.ctypes.data,effective.ctypes.data,mday.ctypes.data)
        diff=qd[sl]-np.where(uc[qu[sl]]>0,mday,date_mean)
        dev=np.sign(diff)*np.power(np.abs(diff)/std,.4)
        probs,means=decode(ctx,s,qi[sl],qu[sl],dev)
        err=float(np.max(np.abs(expected[sl]-means[:,0])))
        maxerr=max(maxerr,err)
        assert err<.0003,(phase,j,err,expected[sl][:3],means[:3,0])
        out[sl]=np.column_stack((base[sl],means-base[sl,None],probs.reshape(n,-1),effective))
        if j%(2048*250)==0:print(phase,j,'max native replay discrepancy',maxerr,flush=True)
    assert np.isfinite(out).all()
    np.save(OUT/(phase+'.npy'),out)
    (OUT/(phase+'-replay.json')).write_text(json.dumps(dict(rows=len(qi),largest_absolute_error=maxerr,
        native_sha256=ne.sha(full.UP/'upstream-cf96-training-export/reference'/('cfnade-96.'+phase+'.npy'))),indent=2)+'\n')
    h=np.load(hist.OUT/(phase+'.npy'),mmap_mode='r')
    return np.column_stack((out,h[:,1:9],h[:,13:16])).astype('f')


def model():
    return CatBoostRegressor(iterations=1200,depth=5,learning_rate=.04,l2_leaf_reg=40,
                             thread_count=2,allow_writing_files=False,random_seed=20261126,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_cf_context.cpp','-o',str(OUT/'cf.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('trainx','probex',base)
    selected,stats,held=anchor.crossfit(x,y,f,base,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,parent=str(parent),qualifying_hashes={},qualifying_labels_used_for_fitting=False,
                known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        xx=features('fulltrain','qual',qb)
        delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32';ne.clipped(qb+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
