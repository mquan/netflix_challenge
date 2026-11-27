"""Learn bounded ensemble corrections in forecast coordinates stable across both prediction roles."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
import own_aligned_forecasts as aligned
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_peer_cf as pc
import own_peer_replay as ph
import own_joint_cohort_source as joint
import own_retained_history as ret
OUT=ne.ROOT/'own-invariant-experts-20261126'
PARENT=blend.OUT/'schema-07'
STAR=np.arange(1,6,dtype='f')


class Role:
    def __init__(self,known,base,cols):
        self.base=base;self.known=known;phase='probex' if known else 'qual'
        ix=ret.map_rows() if known else slice(1408395,None)
        self.cols=[]
        for item in cols.values():
            idx=int(not known);p=Path(item['paths'][idx]);assert ne.sha(p)==item['sha256'][idx]
            self.cols.append(np.load(p,mmap_mode='r'))
        mc=json.loads((pc.OUT/'manifest.json').read_text());mh=json.loads((ph.OUT/'manifest.json').read_text())
        assert ne.sha(pc.OUT/'probabilities.npy')==mc['probabilities_sha256'] and ne.sha(ph.OUT/'effects.npy')==mh['output_sha256']
        cp=np.load(pc.OUT/'probabilities.npy',mmap_mode='r');hn=np.load(ph.OUT/'effects.npy',mmap_mode='r')
        mode=1 if known else 2
        if known:
            h0=hn[ix,:,0];hp=hn[ix,:,mode];p0=cp[ix,0].astype('f');pp=cp[ix,mode].astype('f')
        else:
            h0=hn[ix][:,:,0];hp=hn[ix][:,:,mode];p0=cp[ix][:,0].astype('f');pp=cp[ix][:,mode].astype('f')
        jr=json.loads((joint.OUT/'manifest.json').read_text());pred=[]
        for info in jr['models'].values():
            assert ne.sha(info['vector'])==info['vector_sha256'];pred.append(np.memmap(info['vector'],'<f4','r')[ix])
        j=np.column_stack(pred);jm=j.mean(1);cm=p0@STAR;actual=pp@STAR;self.ref=(jm*.4+actual*.6).astype('f')
        u,m,d=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
        uc=np.log1p(ne.history('trainx','user_cnts')[u]);ic=np.log1p(ne.history('trainx','item_cnts')[m])
        self.cat=(np.clip(((jm-1)*4).astype('i'),0,15)*12+np.clip(((uc-1)*1.3).astype('i'),0,11))*6+np.clip((ic-5).astype('i'),0,5)
        self.c=np.column_stack((jm-3.5,j-jm[:,None],cm-jm,actual-cm,h0-jm[:,None],hp-h0,
                               (uc-5)/2,(ic-8)/2,(d.astype('f')-1700)/500,p0[:,0],p0[:,4],
                               (p0@STAR**2-cm**2)-.8)).astype('f')
    def refdesign(self,rows):
        c=self.c[rows]
        return np.column_stack((np.ones(len(c),'f'),c,c[:,0]**2,c[:,0]**3*.3,c[:,0]*c[:,-5],c[:,0]*c[:,-6],
                  c[:,0]*c[:,6],c[:,0]*c[:,7],c[:,7]**2,c[:,6]**2))
    def raw(self,rows):
        return np.column_stack([v[rows] for v in self.cols])-self.ref[rows,None]
    def calibrate(self):
        rows=np.arange(0,len(self.base),max(1,len(self.base)//42000))
        a=self.refdesign(rows);r=self.raw(rows)
        self.beta=np.linalg.solve(a.T@a+np.eye(a.shape[1])*1.5,a.T@r).astype('f')
        return r-a@self.beta,r
    def block(self,rows,proj):
        r=self.raw(rows)-self.refdesign(rows)@self.beta;v=r@proj
        return np.column_stack((v.clip(-6,6),self.c[rows],self.ref[rows]-self.base[rows],self.base[rows]-3.5)).astype('f')
    def table(self,rows,proj):
        ans=np.empty((len(rows),proj.shape[1]+self.c.shape[1]+2),'f')
        for i in range(0,len(rows),8192):
            sl=slice(i,i+8192);ans[sl]=self.block(rows[sl],proj)
        return ans


def discrepancy(a,b):
    n=min(len(a),len(b),16000);a=a[:n];b=b[:n]
    x=np.r_[a,b];y=np.r_[np.zeros(n),np.ones(n)];tr=np.arange(len(x))%3!=0
    xf=x[tr].astype(float);yf=y[tr]
    def objective(w):
        z=xf@w
        return np.logaddexp(0,z).sum()-yf@z+w@w,xf.T@(expit(z)-yf)+2*w
    fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':160})
    labels=y[~tr]>0;ranks=rankdata(x[~tr]@fit.x);n1=labels.sum();n0=len(labels)-n1
    return float((ranks[labels].sum()-n1*(n1+1)/2)/(n1*n0))


def projection(a,b):
    sd=np.sqrt((a.var(0)+b.var(0))/2).clip(.045)
    x=a/sd;y=b/sd
    c1=x.T@x/len(x);c2=y.T@y/len(y)
    vals,vec=np.linalg.eigh((c1+c2)/2)
    shared=vec[:,-88:]/np.sqrt(vals[-88:].clip(.15))
    drift,axes=np.linalg.eigh(shared.T@(c2-c1)@shared)
    atten=1/np.sqrt(1+(drift/.32)**2)
    proj=(shared@axes)*atten
    return (proj/sd[:,None]).astype('f'),drift


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.view_as(x)
    @staticmethod
    def backward(ctx,grad):return -.075*grad


class Net(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(n,72),nn.LayerNorm(72),nn.SiLU(),nn.Dropout(.20),
                    nn.Linear(72,42),nn.SiLU())
        self.mean=nn.Linear(42,1);self.domain=nn.Sequential(nn.Linear(42,24),nn.SiLU(),nn.Linear(24,1))
        nn.init.zeros_(self.mean.weight);nn.init.zeros_(self.mean.bias)
    def forward(self,x):
        h=self.trunk(x)
        return .16*self.mean(h)[:,0].tanh(),self.domain(Reverse.apply(h))[:,0]


def infer(net,x):
    with torch.inference_mode():return net(x)[0].numpy()


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    receipt=json.loads((aligned.OUT/'freeze.json').read_text());schema=aligned.OUT/'schema.json'
    assert ne.sha(schema)==receipt['schema_sha256'];cols=json.loads(schema.read_text())['forecasts']
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    k=Role(True,base,cols);q=Role(False,qb,cols)
    a,ar=k.calibrate();b,br=q.calibrate();pr,shift=projection(a,b)
    report=dict(raw_domain_auc=discrepancy(ar,br),aligned_domain_auc=discrepancy(a@pr,b@pr),
                covariance_shift_eigenvalues=shift.tolist(),forecasts=len(cols),qualifying_labels_read=False)
    (OUT/'domain.json').write_text(json.dumps(report,indent=2)+'\n');print('domain AUC',report['raw_domain_auc'],report['aligned_domain_auc'],flush=True)
    x=k.table(np.arange(len(y)),pr);rng=np.random.default_rng(9963)
    qr=np.sort(rng.choice(len(qb),300000,replace=False));qx=q.table(qr,pr);qc=q.cat[qr]
    center=x[::17].mean(0);scale=x[::17].std(0).clip(.07)
    def norm(x):return torch.from_numpy(((x-center)/scale).clip(-5,5))
    x=norm(x);qx=norm(qx);err=torch.from_numpy((y-base).astype('f'));held=np.empty(len(y),'f')
    order=np.argsort(qc,kind='stable');count=np.bincount(qc,minlength=16*12*6);off=np.r_[0,count.cumsum()]
    models=[]
    for fold in range(5):
        torch.manual_seed(9857+fold);net=Net(x.shape[1])
        opt=torch.optim.AdamW(net.parameters(),lr=.00075,weight_decay=.045);states=[]
        tr=np.flatnonzero(f!=fold);va=np.flatnonzero(f==fold)
        for ep in range(14):
            net.train();rng.shuffle(tr)
            if ep==9:opt.param_groups[0]['lr']=.00030
            for j in range(0,len(tr),4096):
                rows=tr[j:j+4096];c=k.cat[rows];qi=order[np.minimum(len(order)-1,off[c]+(rng.random(len(c))*count[c]).astype('i'))]
                qi=np.where(count[c]>0,qi,rng.integers(len(qx),size=len(c)))
                pred,dk=net(x[rows]);predq,dq=net(qx[qi])
                dom=.5*(F.softplus(dk).mean()+F.softplus(-dq).mean())
                loss=(pred-err[rows]).square().mean()+.12*pred.square().mean()+.14*dom+.35*(pred.mean()-predq.mean()).square()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.5);opt.step()
            if ep in (6,13):states.append({k:v.detach().clone() for k,v in net.state_dict().items()})
        foldmodels=[]
        for state in states:
            m=Net(x.shape[1]);m.load_state_dict(state);foldmodels.append(m.eval())
        models.extend(foldmodels)
        for j in range(0,len(va),8192):
            rr=va[j:j+8192];held[rr]=sum(infer(m,x[rr]) for m in foldmodels)/len(foldmodels)
        print('fold',fold,ne.rmse(base[va]+held[va],y[va]),flush=True)
    scores={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok=[w for w in scores if ret.qualifies(scores[w],scores['0'])];choice=min(ok,key=lambda w:scores[w]['pooled']) if ok else 'control'
    rep=dict(parent=str(PARENT),selected=choice,known=scores,forecast_schema_sha256=ne.sha(schema),domain_checks=report,qualifying_hashes={},
            qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        del x,qx,k
        delta=np.empty(len(qb),'f')
        for j in range(0,len(qb),8192):
            rows=np.arange(j,min(j+8192,len(qb)));batch=norm(q.block(rows,pr))
            delta[rows]=sum(infer(m,batch) for m in models)/len(models)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
