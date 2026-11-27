"""Fit on the entire original Probe with each target excluded from its observed peer set."""
import json
import numpy as np
import own_anchor_context as anchor
import own_extra_joint as extra
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_replay as replay
import own_retained_history as ret
import own_schema_receipt as receipt

OUT=ne.ROOT/'own-dense-peer-transfer-20261126'
PARENT=blend.OUT/'paired-03'


class Design:
    def __init__(self):
        self.c=replay.Cohort()
        jr=self.c.ref;er=receipt.read(extra.OUT/'manifest.json');hr=receipt.read(replay.OUT/'manifest.json')
        assert not jr['query_outcomes_read'] and not er['qualifying_labels_read'] and not hr['qualifying_ratings_read'] and hr['query_self_excluded']
        self.sources={};pred=[]
        for item in list(jr['models'].values())+list(er['inputs'].values()):
            path=item['vector'];assert receipt.streamed_sha(path)==item['vector_sha256'];self.sources[path]=item['vector_sha256'];pred.append(np.memmap(path,'<f4','r'))
        self.j=np.column_stack(pred);h=replay.OUT/'effects.npy';assert receipt.streamed_sha(h)==hr['output_sha256'];self.sources[str(h)]=hr['output_sha256']
        self.h=np.load(h,mmap_mode='r');self.prior=.77*self.j.mean(1)+.23*self.h[:,:,0].mean(1)
        self.days=self.c.day.astype('i');self.count=np.log1p(np.diff(self.c.d['train_offsets']));self.count=np.asarray(self.count,dtype='f')
        self.pop=np.log1p(np.bincount(self.c.d['train_items'].astype('i'),minlength=int(self.c.m.max()+2)))[1:].astype('f')
        assert len(self.pop)>=self.c.m.max()+1
        self.rng=np.random.default_rng(63701);self.proj=None
    def block(self,rows,mode):
        c=self.c;ix,masks=c.masks(rows);ok=masks[mode];n=ok.sum(1);den=n.clip(1)
        b=self.prior[rows];pre=self.prior[ix];error=c.rows['rating'][ix].astype('f')-pre;gap=self.days[rows,None]-self.days[ix]
        q=self.h[rows,:,mode];intr=q-self.h[rows,:,0];j=self.j[rows]
        def avg(x):return (ok*x).sum(1)/den
        z=[b-3.5,j-b[:,None],self.h[rows,:,0]-b[:,None],intr*2,self.count[c.u[rows]]/5-1,self.pop[c.m[rows]]/8-1,
           self.days[rows]/2000-1,n/3]
        for w in (np.ones_like(gap),1/(1+abs(gap)/25),(gap==0),np.exp(-abs(pre-b[:,None])),1/(1+abs(self.pop[c.m[rows],None]-self.pop[c.m[ix]]))):
            z.extend((avg(w),avg(w*error),avg(w*error*abs(error)),avg(w*(pre-3.5)*error),avg(w*(c.rows['rating'][ix]==5))))
        z.extend((avg(error**2),avg(error)*(n>1),avg(error)*(b-3.5),avg(error*(gap>0)),avg(abs(error)),intr.mean(1)*(n>1)))
        x=np.column_stack(z).astype('f').clip(-5,5)
        if self.proj is None:
            self.proj=self.rng.normal(0,1.3/np.sqrt(x.shape[1]),(x.shape[1],120)).astype('f');self.bias=self.rng.normal(0,.6,120).astype('f')
        return np.column_stack((np.ones(len(x),'f'),x,np.tanh(x@self.proj+self.bias))),b
    def table(self,rows,mode):
        x=np.empty((len(rows),self.block(rows[:1],mode)[0].shape[1]),'f')
        for j in range(0,len(rows),8192):
            sl=slice(j,j+8192);x[sl]=self.block(rows[sl],mode)[0]
        return x


def main():
    OUT.mkdir(exist_ok=True);y,f,_,_=anchor.history_control();fr=receipt.read(PARENT/'freeze.json')
    assert receipt.streamed_sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected'];b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    d=Design();c=d.c;ou,*_=anchor.old_probe();fold=ne.user_folds(ou);np.testing.assert_array_equal(fold[c.ix],f)
    orig=np.arange(c.n);stars=c.rows['rating'][:c.n].astype('f');np.testing.assert_array_equal(stars[c.ix],y)
    dims=d.block(orig[:1],2)[0].shape[1];gram=np.zeros((5,dims,dims));rhs=np.zeros((5,dims));e=stars-d.prior[:c.n]
    actual=d.table(c.ix,1);dense=d.table(orig,2)
    # Dense examples use all other original Probe stars, including those of the extra 280,231 training targets.
    # The user fold removes every rating of validation users from the fit.
    for k in range(5):
        for x,target,mask,weight in ((actual,e[c.ix],f==k,.30),(dense,e,fold==k,.70)):
            v=x[mask];gram[k]+=weight*(v.T@v);rhs[k]+=weight*(v.T@target[mask])
    def solve(g,r):
        a=g.copy();a.flat[::len(a)+1]+=85;return np.linalg.solve(a,r)
    weights=solve(gram.sum(0),rhs.sum(0));raw=np.empty(len(y));hyp=np.empty(c.n)
    for k in range(5):
        w=solve(gram.sum(0)-gram[k],rhs.sum(0)-rhs[k]);raw[f==k]=d.prior[c.ix[f==k]]+actual[f==k]@w
        tr=fold==k;hyp[tr]=d.prior[:c.n][tr]+dense[tr]@w
    delta=raw-b
    def shrink(mask):return float(np.clip(delta[mask]@(y-b)[mask]/(50+delta[mask]@delta[mask]),0,1))
    held=np.empty(len(y),'f')
    for k in range(5):held[f==k]=delta[f==k]*shrink(f!=k)
    a=shrink(np.ones(len(y),bool));stats=blend.metrics(ne.clipped(b+held),y,f);ctrl=blend.metrics(b,y,f);ok=ret.qualifies(stats,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=stats,control=ctrl,raw_sources=d.sources,
             original_dense_peer_crossfit=blend.metrics(hyp,stars,fold),original_prior=blend.metrics(d.prior[:c.n],stars,fold),
             original_targets=c.n,target_self_excluded=True,blend_weight=a,qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n');print('dense transfer',rep['selected'],flush=True)
    if ok:
        del actual,dense
        out=np.empty(len(qb),'f')
        for j in range(0,len(qb),8192):
            sl=slice(j,min(len(qb),j+8192));xx,pp=d.block(c.n+np.arange(sl.start,sl.stop),2);out[sl]=a*(xx@weights+pp-qb[sl])
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',out);ne.clipped(qb+out).astype('<f4').tofile(OUT/'selected.f32')
        rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
