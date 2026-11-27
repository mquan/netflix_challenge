"""Cross-fit sparse movie-pair effects on the fixed competition movie addresses for each user."""
import json
import numpy as np
from scipy import sparse
from scipy.cluster.vq import kmeans2
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_query_sets as sets
import own_retained_history as retained

OUT=ne.ROOT/'own-cohort-pairs-20261126'
EXACT=1<<19


def groups():
    geo=sets.geometry()
    return [kmeans2(x.copy(),k,iter=18,minit='++',seed=819+k)[1].astype('i')
            for k,x in ((32,geo[:,0:50]),(96,geo[:,-48:]))]


class Addresses:
    def __init__(self,train):
        old_u,old_i,old_d,prior=cov.original()
        self.u,self.i,self.d=[np.concatenate((a,ne.history('qual',k))).astype(t) for a,k,t in zip(
            (old_u,old_i,old_d),('user_idxs','item_idxs','dates'),('i','i2','i2'))]
        self.nm=len(ne.history(train,'item_cnts'))
        visible,rows=anchor.training_indices(train,old_u,old_i,old_d)
        self.error=np.zeros(len(self.u),'f');self.error[np.flatnonzero(visible)]=ne.history(train,'ratings')[rows]-prior[visible]
        self.observed=np.zeros(len(self.u),bool);self.observed[np.flatnonzero(visible)]=True
        self.target=retained.map_rows() if train=='trainx' else np.arange(len(old_u),len(self.u))
        assert np.all(~self.observed[self.target])
        order=np.argsort(self.u,kind='stable')
        counts=np.bincount(self.u,minlength=len(ne.history(train,'user_cnts')))
        off=np.r_[0,counts.cumsum()];width=int(counts.max())
        self.by_user=np.full((len(counts),width),-1,'i')
        self.by_user[self.u[order],np.arange(len(order))-off[self.u[order]]]=order
        print(train,'maximum cohort size',width,flush=True)

    def chunk(self,sl,base,clusters):
        target=self.target[sl];u=self.u[target];m=self.i[target].astype('i');n=len(target)
        peer=self.by_user[u]
        a,col=np.nonzero((peer>=0)&(peer!=target[:,None]))
        ix=peer[a,col];i=self.i[ix].astype('i')
        dt=self.d[ix].astype('i')-self.d[target[a]]
        count=np.bincount(a,minlength=n)[a]
        norm=1/np.sqrt(np.maximum(count,1))
        near=1/(1+np.abs(dt)/40)
        low,high=clusters;c=low[m[a]];h=high[m[a]]
        nm=self.nm
        exact=((m[a].astype('uint64')*11400714819323198485)^(i.astype('uint64')*14029467366897019727))%EXACT
        cat=(dt==0).astype('i')+2*(dt>0)+3*(dt>40)+4*(dt< -40)
        assert cat.max(initial=0)<8
        ordinal=np.clip(np.floor((base[sl][a]-1)*1.5),0,5).astype('i')
        # Each field has its own vocabulary and scale. Repeated neighbors in a cluster add in CSR.
        fields=[
            (32*nm,m[a]*32+low[i],norm),
            (32*nm,c*nm+i,norm*.8),
            (96*nm,m[a]*96+high[i],norm*.65),
            (EXACT,exact.astype('i'),norm*.65),
            (96*96*8,(h*96+high[i])*8+cat,norm),
            (32*32*6,(c*32+low[i])*6+ordinal,norm),
            (32*32,(c*32+low[i]),norm*(2*near-1)),
            (96*96,h*96+high[i],norm*self.error[ix]*(.3+.7*near)),
            (32*32*6,(c*32+low[i])*6+ordinal,norm*self.observed[ix]),
        ]
        cols=[];values=[];offset=0
        for width,ids,w in fields:
            cols.append(ids+offset);values.append(w.astype('f'));offset+=width
        data=sparse.csr_matrix((np.concatenate(values),(np.tile(a.astype('i'),len(fields)),np.concatenate(cols))),
                               shape=(n,offset),dtype='f')
        data.eliminate_zeros()
        return data


def chunks(table,base,cluster):
    for j in range(0,len(base),32768):
        sl=slice(j,min(j+32768,len(base)));yield sl,table.chunk(sl,base,cluster)


def solve(x,y,fold):
    # Five held-user fits plus an all-development fit, sharing the sparse multiplication.
    xt=x.T.tocsr()
    mask=(fold[:,None]!=np.arange(6)[None,:]).astype('f')
    b=xt@(mask*y[:,None]);reg=75.
    diag=np.asarray(xt.multiply(xt).sum(1)).ravel()+reg
    weights=np.zeros_like(b);res=b.copy();v=res/diag[:,None];g=v.copy()
    inner=np.sum(res.astype('d')*v,axis=0)
    for k in range(18):
        z=xt@((x@g)*mask)+reg*g
        den=np.sum(g.astype('d')*z,axis=0)
        step=(inner/den).astype('f');weights+=step*g;res-=step*z
        v=res/diag[:,None];new=np.sum(res.astype('d')*v,axis=0)
        g=v+(new/inner).astype('f')*g;inner=new
        if k%4==0:print('iteration',k+1,'relative residual',round(float(np.linalg.norm(res)/np.linalg.norm(b)),5),flush=True)
    return weights


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent'];fr=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    geo=groups();table=Addresses('trainx')
    x=sparse.vstack([a for _,a in chunks(table,base,geo)],format='csr')
    w=solve(x,(y-base).astype('f'),f)
    held=(x@w)[np.arange(len(y)),f]
    np.save(OUT/'oof.npy',held)
    stats={str(s):blend.metrics(ne.clipped(base+s*held),y,f) for s in (0,.25,.5,.75,1)}
    ok=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    rep=dict(selected=key,known=stats,parent=str(parent),ridge=75,qualifying_hashes={},
        qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del x,table
        table=Addresses('fulltrain');delta=np.empty(len(qb),'f')
        for sl,xx in chunks(table,qb,geo):delta[sl]=xx@w[:,-1]
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
