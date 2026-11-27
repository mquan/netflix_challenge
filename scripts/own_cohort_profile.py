"""Learn per-film effects of the fixed original Probe and qualifying movie addresses."""
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.linalg import blas,solve
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_full_context as full
import own_session_latent as latent
import own_query_sets as sets

OUT=ne.ROOT/'own-cohort-profile-20261126'


def cohort(train):
    u,i,d,p=cov.original()
    present,rows=anchor.training_indices(train,u,i,d)
    e=np.full(len(u),np.nan,'f')
    e[present]=ne.history(train,'ratings')[rows]-p[present]
    flagged=np.zeros(len(ne.history(train,'ratings')),bool);flagged[rows]=True
    au,ai,ad=[np.concatenate((x,ne.history('qual',k))) for x,k in zip((u,i,d),('user_idxs','item_idxs','dates'))]
    e=np.r_[e,np.full(len(au)-len(e),np.nan,'f')]
    idx=np.argsort(au,kind='stable')
    off=np.r_[0,np.bincount(au,minlength=len(ne.history(train,'user_cnts'))).cumsum()].astype('i')
    return off,ai[idx].astype('i2'),ad[idx].astype('i2'),e[idx],flagged


def features(train,phase,base,geom):
    lib=ctypes.CDLL(str((OUT/'cohort.dylib').resolve()));it=ctypes.c_int;pt=ctypes.c_void_p
    lib.group_items.argtypes=[it,pt,pt,it,pt];lib.group_items.restype=None
    lib.cohort_design.argtypes=[it,it]+[pt]*8;lib.cohort_design.restype=None
    u,i,d,r=[ne.history(train,k) for k in ('user_idxs','item_idxs','dates','ratings')]
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off,peers,days,anchors,flagged=cohort(train)
    starts,ix=latent.groups(lib,i,len(ic));qs,qo=latent.groups(lib,qi,len(ic))
    re=(r-ne.verified_rank(train))[ix];ce=(r-full.cf96(train))[ix]
    usr=(np.bincount(u,weights=r,minlength=len(uc))+10*r.mean())/(uc+10)
    film=(np.bincount(i,weights=r,minlength=len(ic))+20*r.mean())/(ic+20)
    users,dates,stars=u[ix],d[ix],r[ix];flag=flagged[ix]
    del ix,flagged
    qu,qd=qu[qo],qd[qo]
    def ptrs(*v):return [a.ctypes.data for a in v]
    args=ptrs(off,peers,days,anchors,geom)
    def design(m,uu,dd):
        x=np.zeros((len(uu),87),'f')
        lib.cohort_design(len(uu),m,*ptrs(uu,dd),*args,x.ctypes.data)
        return x
    out=np.zeros((len(qi),31),'f')
    def one(m):
        a,b=starts[m:m+2];lo,hi=qs[m:m+2]
        if hi==lo:return
        gram=np.zeros((87,87),'f',order='F');rhs=np.zeros((87,3),'f')
        for j in range(a,b,8192):
            sl=slice(j,min(j+8192,b));us=users[sl];x=design(m,us,dates[sl])
            dist=np.expm1(x[:,-3].clip(0,9))
            w=np.sqrt(.18+1/(1+dist/30)+4*flag[sl]).astype('f')
            targ=np.column_stack((re[sl],ce[sl],stars[sl]-usr[us]-film[m]+3.6)).astype('f')
            x=np.asfortranarray(x*w[:,None]);targ*=w[:,None]
            gram=blas.ssyrk(1.,x,trans=1,lower=1,c=gram,beta=1.,overwrite_c=True)
            rhs+=x.T@targ
        ws=[]
        for pen in (60.,550.):
            g=gram.copy(order='F');g.flat[::88]+=pen
            ws.append(solve(g,rhs,assume_a='pos',lower=True,check_finite=False))
        w=np.column_stack(ws)
        for j in range(lo,hi,8192):
            sl=slice(j,min(j+8192,hi));x=design(m,qu[sl],qd[sl])
            out[qo[sl]]=np.column_stack((x@w,x[:,-15:],x[:,:10]))
    with ThreadPoolExecutor(2) as pool:
        for k,_ in enumerate(pool.map(one,range(len(ic)))):
            if k%2000==0:print(phase,k,flush=True)
    assert np.isfinite(out).all()
    return np.column_stack((base,ne.verified_rank(phase)-base,full.cf96(phase)-base,np.log1p(ic[qi]),out)).astype('f')


def model():
    return CatBoostRegressor(iterations=1100,depth=5,learning_rate=.045,l2_leaf_reg=50,thread_count=2,
        random_seed=784,verbose=False,allow_writing_files=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_cohort_profile.cpp',
                    'scripts/own_neighbor_features.cpp','-o',str(OUT/'cohort.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    geo=sets.geometry();geom=np.ascontiguousarray(np.column_stack((sets.projection(geo[:,:50],12),sets.projection(geo[:,-48:],12))))
    x=features('trainx','probex',base,geom)
    key,stats,held=anchor.crossfit(x,y,f,base,model);np.save(OUT/'oof.npy',held)
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features('fulltrain','qual',qb,geom)
        delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
