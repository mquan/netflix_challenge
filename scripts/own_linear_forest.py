"""Use disjoint sets of users to find rating regimes and estimate a local linear blend within each regime."""
import json
from pathlib import Path
import numpy as np
import own_anchor_context as anchor
import own_cf_context as cf
import own_day_states as ds
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as peer_cf
import own_peer_replay as peer
import own_peer_temporal as temporal
import own_retained_history as ret
OUT=ne.ROOT/'own-linear-forest-20261126'
PARENT=blend.OUT/'schema-08'


class Inputs:
    def __init__(self,phase,p,q,b,report):
        known=phase=='probex';train='trainx' if known else 'fulltrain'
        ref=report['known_sources' if known else 'qual_sources']
        cols=[]
        for k in report['columns']:
            v,mark=blend.direction(k,p,q,known);assert ref[k]==mark
            if v.shape[1]>5:
                cols.extend((v.mean(1).astype('f'),v.std(1).astype('f')))
            else:cols.extend(a.astype('f') for a in v.T)
        rc=ne.ROOT/'own-cf-temporal-attention-20261126'/(phase+'.npy')
        assert ds.sha(rc)==ref['attention:own-cf-temporal-attention-20261126']['sha256']
        att=np.load(rc,mmap_mode='r');native=full.cf96(phase);rank=ne.verified_rank(phase)
        raw=np.load(cf.OUT/(phase+'.npy'),mmap_mode='r')
        law=raw[:,7:12];star=np.arange(1,6,dtype='f');np.testing.assert_allclose(law@star,native,atol=.0004)
        idx=ret.map_rows() if known else slice(1408395,None);mode=1 if known else 2
        for mod,file,key in ((peer_cf,'probabilities.npy','probabilities_sha256'),(peer,'effects.npy','output_sha256'),(temporal,'means.npy','means_sha256')):
            assert ds.sha(mod.OUT/file)==json.loads((mod.OUT/'manifest.json').read_text())[key]
        pr=np.load(peer_cf.OUT/'probabilities.npy',mmap_mode='r');cm=pr[:,mode][idx].astype('f')@star;cz=pr[:,0][idx].astype('f')@star
        hn=np.load(peer.OUT/'effects.npy',mmap_mode='r');hm=hn[:,:,mode][idx];hz=hn[:,:,0][idx]
        tm=np.load(temporal.OUT/'means.npy',mmap_mode='r');tt=tm[:,mode][idx];zt=tm[:,0][idx]
        part=[att-b,native-b,rank-b,cm-b,cm-cz,tt-b,tt-zt,hm.mean(1)-b,(hm-hz).mean(1)]
        self.ex=np.column_stack((*cols,*part)).astype('f')
        u,i,d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        uc=np.log1p(ne.history(train,'user_cnts')[u]);ic=np.log1p(ne.history(train,'item_cnts')[i]);self.user=u
        h=np.load(ne.ROOT/'own-history-context-20261126'/(phase+'.npy'),mmap_mode='r')
        self.ctx=np.column_stack((b,att,native,d/2000.,uc,ic,raw[:,37:42],law,law@(star**2)-native**2,
                                  h[:,::5],self.ex.std(1),*part,hm.std(1))).astype('f')
        assert np.isfinite(self.ex).all() and np.isfinite(self.ctx).all()
    def project(self,transform=None):
        if transform is None:
            sample=self.ex[::9];mu=sample.mean(0);sd=sample.std(0).clip(.012)
            v=((sample-mu)/sd).clip(-6,6);s,a=np.linalg.eigh(v.T@v/len(v))
            transform=(mu,sd,a[:,-22:]/np.sqrt(s[-22:].clip(.035)))
        m,d,proj=transform;ex=np.empty((len(self.ex),proj.shape[1]+1),'f');ex[:,0]=1
        for j in range(0,len(ex),32768):
            sl=slice(j,j+32768);ex[sl,1:]=((((self.ex[sl]-m)/d).clip(-6,6))@proj).clip(-6,6)
        self.ex=ex;return transform


class Tree:
    def __init__(self,x,y,seed):
        self.nodes=[];rng=np.random.default_rng(seed)
        def grow(rows,depth):
            node=len(self.nodes);self.nodes.append(None)
            if depth==4 or len(rows)<3200:return node
            best=0;rule=None;all_y=y[rows].sum(0)
            for feature in rng.choice(x.shape[1],min(9,x.shape[1]),replace=False):
                values=x[rows,feature];cuts=np.unique(np.quantile(values[::7],np.arange(1,12)/12))
                codes=np.searchsorted(cuts,values);num=len(cuts)+1
                n=np.bincount(codes,minlength=num).cumsum()[:-1];valid=(n>=1600)&(len(rows)-n>=1600)
                if not valid.any():continue
                sums=np.column_stack([np.bincount(codes,weights=v,minlength=num).cumsum()[:-1] for v in y[rows].T])
                gain=(sums*sums).sum(1)/n.clip(1)+((all_y-sums)**2).sum(1)/(len(rows)-n).clip(1)
                gain-=all_y@all_y/len(rows);gain[~valid]=-np.inf
                j=gain.argmax()
                if gain[j]>best:best=gain[j];rule=(feature,cuts[j])
            if rule is not None:
                column,threshold=rule;take=x[rows,column]<=threshold
                self.nodes[node]=(column,threshold,grow(rows[take],depth+1),grow(rows[~take],depth+1))
            return node
        grow(np.arange(len(x)),0)
    def apply(self,x):
        leaves=np.zeros(len(x),'i')
        for j,rule in enumerate(self.nodes):
            if rule is None:continue
            k,t,left,right=rule;rows=np.flatnonzero(leaves==j)
            leaves[rows]=np.where(x[rows,k]<=t,left,right)
        return leaves


def fitted(data,err,f,k):
    rng=np.random.default_rng(18251+k);held=np.flatnonzero(f==k);pred=np.zeros(len(held),'f')
    u=data.user;x=data.ex;trees=[]
    for rep in range(36):
        # Split assignment is per user, independent of that user's ratings.
        tagged=rng.integers(0,2,int(u.max())+1,dtype=np.int8)[u].astype(bool)
        a=np.flatnonzero((f!=k)&tagged);b=np.flatnonzero((f!=k)&~tagged)
        grow=rng.choice(a,min(len(a),54000),replace=False);estimate=rng.choice(b,min(len(b),180000),replace=False)
        axes=np.r_[0,rng.choice(np.arange(1,x.shape[1]),5,replace=False)]
        grad=x[grow][:,axes]*err[grow,None];context=rng.choice(data.ctx.shape[1],min(17,data.ctx.shape[1]),replace=False)
        tree=Tree(data.ctx[grow][:,context],grad,rep+33)
        nodes=tree.apply(data.ctx[estimate][:,context]);order=np.argsort(nodes,kind='stable')
        coeff=np.zeros((len(tree.nodes),x.shape[1]),'f');pen=1100 if rep%2 else 2800
        for leaf in np.unique(nodes):
            lo,hi=np.searchsorted(nodes[order],leaf,side='left'),np.searchsorted(nodes[order],leaf,side='right')
            rows=estimate[order[lo:hi]];a=x[rows];gram=a.T@a;gram.flat[::len(gram)+1]+=pen
            coeff[leaf]=np.linalg.solve(gram,a.T@err[rows])
        leaves=tree.apply(data.ctx[held][:,context]);pred+=np.einsum('ij,ij->i',x[held],coeff[leaves])/36
        trees.append((tree,context,coeff))
    return trees,pred


def main():
    OUT.mkdir(exist_ok=True);y,f,p,q=anchor.history_control();r=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==r['qualifying_hashes']['selected'];b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    data=Inputs('probex',p,q,b,r);transform=data.project();held=np.empty(len(y),'f');forest=[];e=(y-b).astype('f')
    for k in range(5):
        trees,pred=fitted(data,e,f,k);forest.extend(trees);held[f==k]=pred
        print('honest forest fold',k,flush=True)
    stats={str(w):blend.metrics(ne.clipped(b+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok=[k for k in stats if ret.qualifies(stats[k],stats['0'])];k=min(ok,key=lambda x:stats[x]['pooled']) if ok else 'control'
    rep=dict(parent=str(PARENT),selected=k,known=stats,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del data
        data=Inputs('qual',p,q,qb,r);data.project(transform);delta=np.zeros(len(qb),'f')
        for j in range(0,len(qb),32768):
            sl=slice(j,j+32768);cx=data.ctx[sl];xx=data.ex[sl]
            for tree,axes,coef in forest:
                delta[sl]+=np.einsum('ij,ij->i',xx,coef[tree.apply(cx[:,axes])])/len(forest)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(k)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
