"""Fit an ordinal CFNADE posterior update for the unknown ratings in historical last-nine cohorts."""
import ctypes
import json
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_suffix_cf as suffix
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-suffix-impute-20261126'


class Net(nn.Module):
    def __init__(self,d):
        super().__init__()
        for key,val in [('enc',d.enc.transpose(1,0,2)),('op',d.op),('geo',d.geo),('eproj',d.eproj),('eb',d.eb)]:
            self.register_buffer(key,torch.from_numpy(np.ascontiguousarray(val)),persistent=False)
        self.movie=nn.Embedding(len(d.geo),16);nn.init.normal_(self.movie.weight,std=.08)
        self.user=nn.Sequential(nn.Linear(96*3,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,32))
        self.token=nn.Sequential(nn.Linear(24+16+8+20+32+2,72),nn.LayerNorm(72),nn.SiLU(),nn.Dropout(.12),nn.Linear(72,48))
        self.q=nn.Linear(48,24);self.k=nn.Linear(48,24);self.v=nn.Linear(48,24)
        self.gate=nn.Sequential(nn.Linear(6,20),nn.SiLU(),nn.Linear(20,1))
        self.message=nn.Sequential(nn.Linear(48+24+5,40),nn.SiLU(),nn.Linear(40,5))
        self.hidden=nn.Sequential(nn.Linear(48+24+5,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,24))
        self.basis=nn.Parameter(torch.randn(24,320)*.01)
        self.visible=nn.Sequential(nn.Linear(48+24+5,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.15),nn.Linear(48,7))
        for layer in (self.message[-1],self.visible[-1],self.gate[-1]):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.register_buffer('stars',torch.arange(1,6,dtype=torch.float32))
    def forward(self,m,days,valid,anc,features,logits,ctx,norms,bias):
        prob=logits.softmax(3);p0=prob[:,:,0];native=logits[:,:,0];b,n=m.shape
        cfuser=self.user(torch.cat((ctx[:,0],ctx[:,2]-ctx[:,0],ctx[:,3]-ctx[:,0]),1))
        inp=torch.cat((self.geo[m],self.movie(m),features,prob.flatten(2),cfuser[:,None].expand(-1,n,-1),
                  (anc>0).float()[:,:,None],((anc-p0@self.stars)*(anc>0))[:,:,None]),2)
        z=self.token(inp)
        gap=(days[:,:,None]-days[:,None,:]).abs()
        affinity=self.q(z)@self.k(z).transpose(1,2)/math.sqrt(24)
        sim=F.normalize(self.geo[m],dim=2)@F.normalize(self.geo[m],dim=2).transpose(1,2)
        allow=valid[:,None,:]&(m[:,:,None]!=m[:,None,:])
        unknown=allow&(anc==0)[:,None,:]
        pair=torch.stack((affinity,sim,1/(1+gap/21),(gap==0).float(),
              features[:,:,7,None].expand_as(gap),(p0@self.stars)[:,None,:].expand_as(gap)-3.5),3)
        gate=2*torch.sigmoid(self.gate(pair).squeeze(3))*unknown
        attention=affinity.masked_fill(~allow,-10000).softmax(2)*allow
        hidden_m=self.v(z);context=attention@hidden_m
        N=norms[:,0,None];tot=ctx[:,0]*N.clamp(min=1).sqrt()
        ep=self.enc[m];op=self.op[m]
        prior=p0;state=z;raw=native
        for _ in range(2):
            token=torch.cat((z,context,prior),2)
            message=(prior.clamp(min=1e-8).log()+.75*self.message(token).tanh()).softmax(2)
            expected=(ep*message[:,:,:,None]).sum(2)
            total=(tot[:,None,:]+gate@expected)/(N+gate.square().sum(2)).clamp(min=1).sqrt()[:,:,None]
            latent=torch.tanh(total@self.eproj.T+self.eb)
            shift=.20*(self.hidden(token).tanh()@self.basis)
            raw=(op*(latent+shift).unsqueeze(2)).sum(-1)+bias
            prior=raw.softmax(2)
        o=self.visible(torch.cat((state,context,prior),2)).tanh()
        adjusted=native+(1+.7*o[:,:,5,None])*(raw-native)+.65*o[:,:,:5]
        adjusted=adjusted+.3*o[:,:,6,None]*(logits[:,:,2]-native).clamp(-5,5)
        return adjusted,p0,prior


def run(train,phase,lib):
    d=suffix.Data(train,phase,lib);torch.manual_seed(19187);net=Net(d);opt=torch.optim.AdamW(net.parameters(),lr=.00064,weight_decay=.022)
    start=time.monotonic()
    for ep in range(20):
        rng=np.random.default_rng(8893+ep);net.train()
        opt.param_groups[0]['lr']=.00010+.00064*(1+math.cos(ep*math.pi/20))/2
        for j in range(475):
            u=rng.choice(d.ready,144,replace=False);args,y=d.batch(u,int(rng.integers(1,2**30)),True,return_state=True)
            v,raw,mid=net(*args);ok=y>=0
            mu=v[ok].softmax(1)@net.stars
            loss=.58*(mu-y[ok]-1).square().mean()+.31*F.cross_entropy(v[ok],y[ok])+.13*((mid[ok]@net.stars)-y[ok]-1).square().mean()+.09*(mu-raw[ok]@net.stars).square().mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
        print(train,'epoch',ep+1,round(time.monotonic()-start),flush=True)
    torch.save(net.state_dict(),OUT/(train+'.pt'));net.eval();result=np.empty((len(d.native),8),'f')
    with torch.inference_mode():
        for j in range(0,len(d.eval_users),256):
            u=d.eval_users[j:j+256];args,_=d.batch(u,1,False,return_state=True);v,raw,mid=net(*args)
            grid=d.grid[u];rows=d.lookup[grid.clip(min=0)];ok=(grid>=0)&(rows>=0)
            native=raw@net.stars
            assert np.max(abs(native.numpy()[ok]-d.native[rows[ok]]),initial=0)<.0004
            prob=v.softmax(2)
            result[rows[ok]]=torch.cat((prob,(prob@net.stars)[:,:,None],native[:,:,None],(mid@net.stars)[:,:,None]),2).numpy()[ok]
    np.save(OUT/(phase+'.npy'),result);return result


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    lib=ctypes.CDLL(str((suffix.OUT/'suffix.dylib').resolve()));lib.suffix_cf.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*18;lib.suffix_cf.restype=None
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    def feats(v,b,train,phase):
        u=ne.history(phase,'user_idxs');return np.column_stack((b,v,v[:,5]-b,v[:,5]-v[:,6],ne.verified_rank(phase)-b,np.log1p(ne.history(train,'user_cnts')[u]))).astype('f')
    def model():return CatBoostRegressor(iterations=600,depth=5,learning_rate=.045,l2_leaf_reg=55,thread_count=2,random_seed=1417,verbose=False,allow_writing_files=False)
    v=run('trainx','probex',lib);x=feats(v,base,'trainx','probex');_,_,td=anchor.crossfit(x,y,f,base,model)
    def design(v,b,t):return np.column_stack((v[:,5]-b,v[:,5]-v[:,6],v[:,7]-v[:,6],t))
    z=design(v,base,td);held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=z[~tr]@fit(z[tr],(y-base)[tr])
    w=fit(z,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(parent),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),qualifying_hashes={},
            qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        v=run('fulltrain','qual',lib);t=model().fit(x,y-base).predict(feats(v,qb,'fulltrain','qual'),thread_count=2)
        delta=design(v,qb,t)@w;np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
