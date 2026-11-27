"""Cross-fit the incremental value of revealed peer errors against measured changes in frozen ratings models."""
import json
from time import sleep
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_day_states as days
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as cf
import own_peer_replay as replay
import own_peer_temporal as temporal
import own_rbm_joint as rbm
import own_retained_history as ret
OUT=ne.ROOT/'own-peer-innovation-20261126'
PARENT=blend.OUT/'schema-08'
STAR=np.arange(1,6,dtype='f')


class Inputs:
    def __init__(self,base,qb,y,fold):
        c=self.c=replay.Cohort();self.g=cov.movie_geometry();self.old=cov.original()[-1]
        for mod,file,key in ((cf,'probabilities.npy','probabilities_sha256'),(replay,'effects.npy','output_sha256'),(temporal,'means.npy','means_sha256')):
            assert days.sha(mod.OUT/file)==json.loads((mod.OUT/'manifest.json').read_text())[key]
        p=np.load(cf.OUT/'probabilities.npy',mmap_mode='r')
        self.cfm=np.column_stack([p[:,k].astype('f')@STAR for k in (0,1,2)])
        self.hn=np.load(replay.OUT/'effects.npy',mmap_mode='r');self.temp=np.load(temporal.OUT/'means.npy',mmap_mode='r')
        self.star=c.rows['rating'].astype('f');np.testing.assert_array_equal(self.star[c.ix],y)
        self.pred=np.empty(len(self.star),'f');self.pred[c.ix]=base;self.pred[c.n:]=qb
        sj=np.column_stack([np.memmap(m['vector'],'<f4','r')[:c.n] for m in c.ref['models'].values()])
        pre=self.old;original=[]
        for name in ret.MODELS:
            r=json.loads((ret.QUAL/name/'report.json').read_text());path=r['column'];assert days.sha(path)==r['column_sha256']
            original.append(np.fromfile(path,'<f4'))
        r=json.loads((ret.QUAL/'B2-fixed128/report.json').read_text());assert days.sha(r['column'])==r['column_sha256']
        original.append(np.fromfile(r['column'],'<f4'))
        info=json.loads((rbm.OUT/'freeze.json').read_text())['signals']
        for name in rbm.MODELS:
            path=rbm.OUT/(name+'.f32');assert days.sha(path)==info[str(path)]
            original.append(np.memmap(path,'<f4','r').reshape(-1,7)[:c.n,0])
        supplied=np.column_stack((sj,self.hn[:c.n,:,0],self.cfm[:c.n,0],self.temp[:c.n,0],*original))-pre[:,None]
        features=np.column_stack((supplied,pre,self.g[c.m[:c.n],::5],np.log1p(np.diff(c.d['train_offsets']))[c.u[:c.n]])).astype('f')
        seen=np.flatnonzero(c.seen);imputed=np.zeros(len(seen),'f');oof=np.empty(len(base),'f')
        for k in range(5):
            tr=fold!=k;va=~tr;fit=CatBoostRegressor(iterations=410,depth=6,learning_rate=.065,l2_leaf_reg=45,random_seed=9245+k,
                thread_count=2,verbose=False,allow_writing_files=False)
            fit.fit(features[c.ix[tr]],base[tr]-pre[c.ix[tr]])
            imputed+=fit.predict(features[seen],thread_count=2)/5;oof[va]=pre[c.ix[va]]+fit.predict(features[c.ix[va]],thread_count=2)
        self.pred[seen]=pre[seen]+imputed;self.distill_rmse=ne.rmse(oof,base)
        self.activity=np.log1p(np.diff(c.d['train_offsets'])).astype('f')
    def features(self,rows,secondary):
        c=self.c;ar=c.anchors[c.u[rows]];idx=ar.clip(min=0)
        one=1 if secondary else 0;two=2 if secondary else 1
        mask=(ar>=0)&(ar!=rows[:,None])&(c.seen[idx]!=secondary)
        e=self.star[idx]-self.pred[idx];old=self.star[idx]-self.old[idx];n=mask.sum(1);den=1+.22*np.maximum(0,n-1)
        d=c.day[rows,None].astype('i')-c.day[idx];near=1/(1+abs(d)/35)
        m=c.m[rows];am=c.m[idx];sim=np.einsum('nd,nad->na',self.g[m],self.g[am])
        b=self.pred[rows];p=self.pred[idx];same=d==0
        cp=self.cfm[rows,two]-self.cfm[rows,one];tp=self.temp[rows,two]-self.temp[rows,one]
        h=self.hn[rows,:,two]-self.hn[rows,:,one];proxy=.55*cp+.35*tp+.10*h.mean(1)
        def add(val):return (val*mask).sum(1)/den
        row=lambda x:x[:,None]
        mean=np.where(n,add(e)*den/n.clip(1),0)
        features=[cp,tp,*h.T,proxy,proxy*(n>1),proxy*np.tanh(b-3.5)]
        for err in (e,old,p-self.old[idx],e*abs(e),e*(p-3.5)):
            for gate in (1,near,same,sim,sim**2,sim*near,np.exp(-abs(row(b)-p))):
                features.append(add(err*gate))
        features.extend((mean,mean*abs(mean),mean*(n>1),mean*np.tanh(b-3.5),mean*np.tanh(self.activity[c.u[rows]]-5),
                    add(e*near)*np.tanh(b-3.5),add(e*sim)*np.tanh(b-3.5),add(e*(d>0)),add(e*(d<0)),
                    add((e-row(mean))**2)*(n>1),add(np.sign(e)),add((e>0)*(self.star[idx]==5)),add((e<0)*(self.star[idx]==1))))
        x=np.column_stack(features).astype('f');x[n==0]=0;proxy[n==0]=0
        assert np.isfinite(x).all()
        return x,proxy,n
    def table(self,rows,mode):
        features=[];proxy=[];counts=[]
        for j in range(0,len(rows),16384):
            x,p,n=self.features(rows[j:j+16384],mode);features.append(x);proxy.append(p);counts.append(n)
        return np.concatenate(features),np.concatenate(proxy),np.concatenate(counts)


def solve(xn,xs,e,proxy):
    # Both losses rely exclusively on target ratings of this fold's training users.
    gram=.28*(xn.T@xn)+.72*(xs.T@xs);rhs=.28*(xn.T@e)+.72*(xs.T@(e-proxy))
    gram.flat[::len(gram)+1]+=1100
    return np.linalg.solve(gram,rhs)


def main():
    OUT.mkdir(exist_ok=True)
    protocol=dict(parent=str(PARENT),standard_weights=dict(cf96=.55,temporal=.35,history=.10),
        standard='Measured same-model change after other Probe stars become observable. Prediction of this row excludes its own rating.',
        secondary='All other withheld Probe stars of this user may be revealed. Model parameters fitted only on training-fold users.',
        canonical='Only observed trainx stars are available. Their missing full-ensemble forecasts are imputed from local models.',
        deployment='Apply the additional learned correction for newly observed ProbeX peers on top of the full-data parent.',
        gate='Require lower pooled and at least four better user folds separately for both protocols at the same shrinkage.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    while not (temporal.OUT/'manifest.json').is_file():sleep(12)
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    d=Inputs(b,qb,y,f);xn,_,n=d.table(d.c.ix,False);xs,p,ns=d.table(d.c.ix,True)
    en=np.empty(len(y));es=en.copy()
    for k in range(5):
        tr=f!=k;w=solve(xn[tr],xs[tr],(y-b)[tr],p[tr]);en[~tr]=xn[~tr]@w;es[~tr]=xs[~tr]@w
    w=solve(xn,xs,y-b,p)
    def scores(base,d):return {str(t):blend.metrics(ne.clipped(base+t*d),y,f) for t in (0,.125,.25,.5,1)}
    native,secondary=scores(b,en),scores(b+p,es)
    ok=[k for k in native if ret.qualifies(native[k],native['0']) and ret.qualifies(secondary[k],secondary['0'])]
    choice=min(ok,key=lambda k:native[k]['pooled']) if ok else 'control'
    rep=dict(parent=str(PARENT),selected=choice,known=native,extra_peer_validation=secondary,protocol_sha256=ne.sha(OUT/'protocol.json'),
        imputed_parent_oof_rmse=d.distill_rmse,native_anchor_counts=np.bincount(n).tolist(),extra_peer_counts=np.bincount(ns).tolist(),
        qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        delta=np.empty(len(qb),'f')
        for j in range(0,len(qb),16384):
            rows=np.arange(j,min(j+16384,len(qb)));delta[rows]=d.features(rows+d.c.n,True)[0]@w
        np.save(OUT/'oof.npy',en);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
