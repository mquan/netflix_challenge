"""Condition ordinal rating distributions on visible ratings with Gaussian quadrature over a shared user effect."""
import ctypes
import json
import subprocess
import numpy as np
from scipy.special import ndtri,roots_hermitenorm
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as cf
import own_peer_replay as replay
import own_retained_history as ret
import own_schema_receipt as receipt

OUT=ne.ROOT/'own-ordinal-copula-20261126'
PARENT=blend.OUT/'paired-03'
STAR=np.arange(1,6,dtype='f')


class Data:
    def __init__(self,lib):
        self.c=replay.Cohort();self.lib=lib;src=cf.OUT/'probabilities.npy';self.digest=receipt.streamed_sha(src)
        assert self.digest==receipt.read(cf.OUT/'manifest.json')['probabilities_sha256']
        self.prob=np.load(src,mmap_mode='r');p=self.prob[:,0].astype('f');p/=p.sum(1,keepdims=True)
        self.thresh=np.ascontiguousarray(ndtri(np.cumsum(p,axis=1)[:,:4].clip(.000001,.999999)),'f')
        self.rating=np.ascontiguousarray(self.c.rows['rating'],'i1');self.day=np.ascontiguousarray(self.c.day,'i');self.movie=np.ascontiguousarray(self.c.m,'i')
        self.geo=np.ascontiguousarray(cov.movie_geometry(),'f');self.geo/=np.linalg.norm(self.geo,axis=1,keepdims=True).clip(.0001)
        n,w=roots_hermitenorm(11);self.nodes=n.astype('f');self.weight=(w/w.sum()).astype('f')
    def generate(self,rows,mode):
        out=[]
        for j in range(0,len(rows),16384):
            idx=np.ascontiguousarray(rows[j:j+16384],'i');ar,masks=self.c.masks(idx);ar=np.ascontiguousarray(ar,'i');mask=np.ascontiguousarray(masks[mode],'u1')
            assert np.all((ar!=idx[:,None])|~mask.astype(bool));assert not np.any(mask&(self.rating[ar]==0))
            raw=np.empty((len(idx),8),'f');args=(idx,ar,mask,self.movie,self.day,self.rating,self.thresh,self.geo,self.nodes,self.weight,raw)
            self.lib.copula(len(idx),self.geo.shape[1],*[q.ctypes.data for q in args])
            n=mask.sum(1);delta=raw[:,::2];mean=self.prob[idx,0].astype('f')@STAR;native=self.prob[idx,mode].astype('f')@STAR
            update=native-mean
            z=np.column_stack((delta,delta-update[:,None],delta*(n[:,None]>=2),delta*np.tanh(mean[:,None]-3.6),
                               .3*raw[:,1::2],update,update*(n>=2))).astype('f')
            z[n==0]=0;out.append(z)
        return np.concatenate(out)


def main():
    OUT.mkdir(exist_ok=True);subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_ordinal_copula.cpp','-o',str(OUT/'copula.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'copula.dylib').resolve()));lib.copula.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*11;lib.copula.restype=None
    y,f,_,_=anchor.history_control();fr=receipt.read(PARENT/'freeze.json');assert receipt.streamed_sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4');d=Data(lib);xs=d.generate(d.c.ix,1);xd=d.generate(d.c.ix,2)
    mu0=d.prob[d.c.ix,1].astype('f')@STAR;new=d.prob[d.c.ix,2].astype('f')@STAR;proxy=new-mu0
    def solve(tr):
        g=.60*(xs[tr].T@xs[tr])+.40*(xd[tr].T@xd[tr]);rhs=.60*(xs[tr].T@(y-b)[tr])+.40*(xd[tr].T@(y-b-proxy)[tr])
        g.flat[::len(g)+1]+=350;return np.linalg.solve(g,rhs)
    held=np.empty(len(y),'f');dense=held.copy()
    for k in range(5):
        tr=f!=k;w=solve(tr);held[~tr]=xs[~tr]@w;dense[~tr]=xd[~tr]@w
    w=solve(np.ones(len(y),bool));ctrl=blend.metrics(b,y,f);dense_ctrl=blend.metrics(ne.clipped(b+proxy),y,f)
    choices=(.25,.5,.75,1);stats={str(a):blend.metrics(ne.clipped(b+a*held),y,f) for a in choices}
    audit={str(a):blend.metrics(ne.clipped(b+proxy+a*dense),y,f) for a in choices}
    good=[k for k in stats if ret.qualifies(stats[k],ctrl) and ret.qualifies(audit[k],dense_ctrl)]
    choice=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=choice,known=stats,control=ctrl,simulated_dense=audit,simulated_control=dense_ctrl,
             weights=w.tolist(),cf_source_sha256=d.digest,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n');print('ordinal copula selection',choice,flush=True)
    if good:
        qz=d.generate(np.arange(d.c.n,len(d.c.rows)),2);delta=qz@w;np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
