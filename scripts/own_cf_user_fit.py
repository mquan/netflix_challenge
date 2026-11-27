"""Fit each user's residual CFNADE preferences using leave-one-rating-out encodings."""
import ctypes
import json
import subprocess
import time
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_history_context as hist
import own_full_context as full
import own_cf_context as cf

OUT=ne.ROOT/'own-cf-user-fit-20261126'
PENS=(24.,120.,600.)


def projected_decoder(p,ic):
    cum=p.cumsum(0)
    nominal=np.arange(-2,3,dtype='f')*.2
    g=np.einsum('t,tik->ik',nominal,cum)
    gram=(g.T*(np.sqrt(ic+1)))@g
    _,v=np.linalg.eigh(gram)
    U=v[:,-20:]
    small=cum@U
    rms=np.maximum(np.sqrt(np.average((g@U)**2,axis=0,weights=ic)),.001)
    return np.ascontiguousarray((small/rms).transpose(1,0,2),'f')


def features(train,phase,base):
    lib=ctypes.CDLL(str((OUT/'equations.dylib').resolve()))
    it,pt=ctypes.c_int,ctypes.c_void_p
    lib.cf_totals.argtypes=[it,it]+[pt]*7
    lib.cf_equations.argtypes=[it]*3+[pt]*5
    lib.cf_totals.restype=lib.cf_equations.restype=None
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    qo=np.argsort(qu,kind='stable').astype('i')
    qoff=np.r_[0,np.bincount(qu,minlength=len(uc)).cumsum()].astype('i')
    s=cf.checkpoint(train)
    enc=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
    total=np.empty((len(uc),96),'f');mday=np.empty(len(uc),'f')
    lib.cf_totals(len(uc),len(ic),*[a.ctypes.data for a in (off,i,d,r,enc,total,mday)])
    daymean=np.float32(d.mean())
    dayvar=np.float32(sum(np.dot(d[j:j+1_000_000].astype(float),d[j:j+1_000_000].astype(float))
                   for j in range(0,len(d),1_000_000))/len(d))-daymean*daymean
    sd=np.sqrt(dayvar);mday[uc==0]=daymean
    small=projected_decoder(s['p'],ic)
    rank=ne.verified_rank(train);trained=full.cf96(train);native=full.cf96(phase)
    dsgn=32
    stars=np.arange(1,6,dtype='f')
    def compute(us,items,date,rating=None):
        ctx=total[us].copy()
        nr=uc[us].copy()
        if rating is not None:
            ctx-=enc[rating-1,items]
            nr-=1
        ctx/=np.sqrt(np.maximum(nr,1)).astype('f')[:,None]
        rel=(date-mday[us])/sd
        dev=np.sign(rel)*np.abs(rel)**.4
        bias=(s['b'][:,items]+s['user_bias'][:,us]+s['user_drift'][:,us]*dev).T
        h=np.tanh(ctx@s['enc_proj'].T+s['enc_bias'])
        z=h@s['dec_proj'].T
        z[nr==0]=0
        raw=np.einsum('nk,tnk->nt',z,s['p'][:,items])+bias
        scores=np.cumsum(raw,axis=1);scores-=scores.max(1,keepdims=True)
        prob=np.exp(scores);prob/=prob.sum(1,keepdims=True)
        mu=prob@stars
        g=np.einsum('nt,ntk->nk',prob*(stars[None,:]-mu[:,None]),small[items])
        x=np.column_stack((np.ones(len(mu),'f'),mu-3.6,rel.clip(-2,2),prob[:,0]-prob[:,-1],g,
                           g[:,-8:]*rel.clip(-2,2)[:,None])).astype('f')
        return x,mu
    # Verify the unmasked path against existing producer output before using the mask.
    sample=np.linspace(0,len(u)-1,8192,dtype='i')
    _,verify=compute(u[sample],i[sample],d[sample])
    assert np.max(np.abs(verify-trained[sample]))<.0003
    result=np.zeros((len(qi),len(PENS)*3+1),'f')
    start=time.monotonic()
    maxerr=0.
    for first in range(0,len(uc),640):
        last=min(first+640,len(uc));num=last-first
        low,hi=off[first],off[last]
        tr=np.empty((hi-low,dsgn),'f');targets=np.empty((hi-low,3),'f')
        for a in range(low,hi,4096):
            sl=slice(a,min(hi,a+4096));dest=slice(a-low,sl.stop-low)
            x,mu=compute(u[sl],i[sl],d[sl],r[sl])
            tr[dest]=x
            targets[dest,0]=r[sl]-mu
            targets[dest,1]=r[sl]-trained[sl]
            targets[dest,2]=r[sl]-rank[sl]
        gs=np.zeros((num,dsgn,dsgn),'f')
        rhs=np.zeros((num,dsgn,3),'f')
        starts=np.ascontiguousarray(off[first:last+1]-low,'i')
        lib.cf_equations(num,dsgn,3,*[a.ctypes.data for a in (starts,tr,targets,gs,rhs)])
        ix=qo[qoff[first]:qoff[last]]
        qx,qmu=compute(qu[ix],qi[ix],qd[ix])
        if len(ix):maxerr=max(maxerr,float(np.max(np.abs(native[ix]-qmu))))
        assert maxerr<.0003,maxerr
        result[ix,-1]=qmu-base[ix]
        rg=np.arange(dsgn)
        for j,pen in enumerate(PENS):
            sys=gs.copy()
            sys[:,rg,rg]+=pen
            w=np.linalg.solve(sys,rhs)
            result[ix,j*3:j*3+3]=np.einsum('nk,nkh->nh',qx,w[qu[ix]-first])
        if first%(640*100)==0:print(phase,first,'elapsed',round(time.monotonic()-start),'replay',maxerr,flush=True)
    assert np.isfinite(result).all()
    np.save(OUT/(phase+'.npy'),result)
    h=np.load(hist.OUT/(phase+'.npy'),mmap_mode='r')
    return np.column_stack((result,base,h[:,1:9],h[:,13:16])).astype('f')


def model():
    return CatBoostRegressor(iterations=850,depth=5,learning_rate=.04,l2_leaf_reg=45,
                             thread_count=2,allow_writing_files=False,random_seed=20261126,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','-shared','-fPIC',
                    'scripts/own_cf_user_fit.cpp','scripts/own_cf_adapter.cpp','-framework','Accelerate','-o',str(OUT/'equations.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
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
