"""Fold observed competition ratings into two original timeSVD++ bases using local prediction derivatives."""
import ctypes
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import numpy as np
from gbdt_blend import load,PROBE
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_joint_cohort_source as src
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-joint-jacobian-20261126'
MODELS=('J2-joint128-reg02','PQ2-factor200-frequency')
DS=Path('data/processed/netflix.bin')


def coordinates(d,sig):
    app=str(OUT/'export');users=[];films=[]
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','scripts/own_joint_jacobian.cpp',
                    '-framework','Accelerate','-o',app],check=True)
    for name in MODELS:
        rec=sig['models'][name]['model_artifact'];assert ne.sha(rec['path'])==rec['sha256']
        mp,up,axes=[OUT/(name+suffix) for suffix in ('.items.f32','.users.f32','.axis.f32')]
        subprocess.run([app,str(DS),rec['path'],str(mp),'dump','unused'],check=True)
        v=np.fromfile(mp,'<f4').reshape(d['items'],-1);center=v.mean(0);vc=v-center
        _,e=np.linalg.eigh(vc.T@vc);projection=np.ascontiguousarray(e[:,-12:],'f');projection.tofile(axes)
        subprocess.run([app,str(DS),rec['path'],str(mp),str(axes),str(up)],check=True)
        m=vc@projection;u=np.fromfile(up,'<f4').reshape(d['users'],12)
        films.append((m/m.std(0).clip(.0001)).clip(-7,7));users.append(((u-u.mean(0))/u.std(0).clip(.0001)).clip(-6,6))
    return np.ascontiguousarray(np.column_stack([np.ones(d['items'],'f'),*films])),np.column_stack(users)


class Cohort:
    def __init__(self):
        sig=json.loads((src.OUT/'manifest.json').read_text());assert ne.sha(DS)==sig['training_data_sha256']
        d=load(DS);oo=d['probe'];q=np.fromfile(src.OUT/'zero-queries.bin',PROBE);assert not np.any(q['rating'])
        self.n0=len(oo);self.u,self.m,self.day=tuple(np.r_[oo[k],q[k]].astype(ty) for k,ty in [('user','i'),('item','i2'),('day','i2')])
        self.m-=1
        self.star=np.r_[oo['rating'],np.zeros(len(q),'u1')].astype('f')
        self.ix=ret.map_rows()
        old_u,old_m,old_d,old_prior=cov.original()
        seen,_=anchor.training_indices('trainx',old_u,old_m,old_d)
        self.seen=np.r_[seen,np.zeros(len(q),bool)]
        np.testing.assert_array_equal(self.seen[:len(oo)],~np.isin(np.arange(len(oo)),self.ix))
        order=np.argsort(self.u,kind='stable');counts=np.bincount(self.u,minlength=d['users'])
        starts=np.r_[0,counts.cumsum()];self.width=int(counts.max())
        self.grid=np.full((d['users'],self.width),-1,'i')
        self.grid[self.u[order],np.arange(len(order))-starts[self.u[order]]]=order
        self.teacher=np.column_stack([np.fromfile(meta['vector'],'<f4') for meta in sig['models'].values()])
        for meta in sig['models'].values():assert ne.sha(meta['vector'])==meta['vector_sha256']
        self.prior=self.teacher.mean(1)
        self.anchor_error=self.star.copy();self.anchor_error[:len(oo)]-=old_prior
        v,g=coordinates(d,sig)
        self.activity=np.log1p(np.diff(d['train_offsets'])).astype('f')
        self.users=g;self.nm=d['items']
        sub=['c++','-O3','-std=c++17','-Wno-deprecated-declarations','-shared','-fPIC','scripts/own_joint_precision.cpp',
                    '-framework','Accelerate','-o',str(OUT/'precision.dylib')]
        subprocess.run(sub,check=True)
        lib=ctypes.CDLL(str((OUT/'precision.dylib').resolve()));lib.jacobian_kernel.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*8
        lib.jacobian_kernel.restype=None
        self.kernel=np.zeros((d['users'],self.width,self.width,2),'f')
        args=[a.ctypes.data for a in (d['train_offsets'],d['train_items'],d['train_days'],self.grid,self.m,self.day,v,self.kernel)]
        def work(j):lib.jacobian_kernel(j,min(j+256,d['users']),v.shape[1],self.width,*args)
        with ThreadPoolExecutor(2) as pool:
            for j,_ in enumerate(pool.map(work,range(0,d['users'],256))):
                if j%450==0:print('user curvature',j,flush=True)
        self.kernel=self.kernel.astype('f2')
    def design(self,rows):
        u=self.u[rows];pred=self.prior[rows]
        return np.column_stack((np.ones(len(u),'f'),self.users[u],(self.day[rows]-1700)/600,
              pred-3.5,(self.activity[u]-5)/2,self.teacher[rows,0]-pred,self.teacher[rows,-1]-pred)).astype('f')

    def shared(self,tr,label,va,penalties):
        train=np.flatnonzero(tr);order=np.argsort(self.m[train],kind='stable');train=train[order]
        starts=np.r_[0,np.bincount(self.m[train],minlength=self.nm).cumsum()];x=self.design(train);y=label[train]
        dim=x.shape[1];gram=np.zeros((self.nm,dim,dim),'f');rhs=np.zeros((self.nm,dim),'f')
        for m in np.flatnonzero(np.diff(starts)):
            sl=slice(starts[m],starts[m+1]);a=x[sl];gram[m]=a.T@a;rhs[m]=a.T@y[sl]
        ws=[]
        for p in penalties:
            g=gram.copy();idx=np.arange(dim);g[:,idx,idx]+=p
            ws.append(np.linalg.solve(g,rhs[:,:,None])[:,:,0])
        def infer(rows):
            result=np.empty((len(rows),len(ws)),'f')
            for j in range(0,len(rows),16384):
                sl=slice(j,j+16384);ix=rows[sl];a=self.design(ix)
                result[sl]=np.column_stack([np.einsum('ij,ij->i',a,w[self.m[ix]]) for w in ws])
            return result
        return infer(va)

    def personal(self,rows,visible):
        # Solve on all and only the actually observed members of this user's fixed slate.
        index=np.empty(len(self.u),'i');good=self.grid>=0
        index[self.grid[good]]=np.nonzero(good)[1]
        ans=np.empty((len(rows),7),'f')
        for j in range(0,len(rows),8192):
            ix=rows[j:j+8192];u=self.u[ix];grid=self.grid[u];gg=grid.clip(min=0);n=len(u)
            seen=visible[gg]&(grid>=0);assert not np.any(seen[np.arange(n),index[ix]])
            e=self.anchor_error[gg]*seen
            g=self.kernel[u].astype('f');decay=1/(1+abs(self.day[gg][:,:,None]-self.day[gg][:,None,:])/20.)
            result=[]
            for a,b,d in ((1,0,0),(0,1,0),(.35,.5,.12)):
                kernel=a*g[:,:,:,0]+b*g[:,:,:,1]+d*decay
                mat=kernel*(seen[:,:,None]&seen[:,None,:]);ii=np.arange(self.width)
                mat[:,ii,ii]+=.7
                c=np.linalg.solve(mat,e[:,:,None])[:,:,0]
                result.append((kernel[np.arange(n),index[ix]]*c).sum(1))
            k=g[np.arange(n),index[ix]]
            ans[j:j+n]=np.column_stack((*result,(k[:,:,0]*e).sum(1),(k[:,:,1]*e).sum(1),
                                        e.sum(1)/(seen.sum(1)+1),seen.sum(1)))
        return ans

    def features(self,train,va,vis,err,dev,base):
        a=self.shared(train,self.star-self.prior,va,(40,300))
        b=self.shared(dev,err,va,(80,600))
        pers=self.personal(va,vis)
        raw=self.teacher[va].mean(1)-base
        return np.column_stack((a,b,pers[:,:6],raw,raw+a[:,0]+pers[:,0],raw+a[:,1]+pers[:,2]))


def main():
    OUT.mkdir(exist_ok=True);y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    data=Cohort();np.testing.assert_array_equal(data.star[data.ix],y)
    err=np.zeros(len(data.star),'f');err[data.ix]=y-base
    x=np.empty((len(y),13),'f')
    for j in range(5):
        tr=np.zeros(len(data.u),bool);tr[data.ix[f!=j]]=True
        va=f==j
        x[va]=data.features(tr|data.seen,data.ix[va],data.seen,err,tr,base[va])
        print('fold features',j,flush=True)
    w=fit(x.astype(float),y-base);held=np.empty(len(y))
    for j in range(5):
        tr=f!=j;held[~tr]=x[~tr]@fit(x[tr].astype(float),(y-base)[tr])
    ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(parent),known=known,control=ctrl,weights=w.tolist(),selected='candidate' if ok else 'control',qualifying_hashes={},
             qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        tr=np.zeros(len(data.u),bool);tr[data.ix]=True;allseen=tr|data.seen
        qx=data.features(allseen,np.arange(data.n0,len(data.u)),allseen,err,tr,qb);delta=qx@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
