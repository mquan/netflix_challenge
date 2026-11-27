"""Fine-tune our trained history model to pass predicted star distributions between missing titles."""
import argparse
import json
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import own_anchor_context as anchor
import own_cf_context as cf
import own_masked_attention as first
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_temporal_attention as temporal

OUT=ne.ROOT/'own-slate-attention-20261126'
PARENT=ne.ROOT/'own-temporal-attention-20261126'


class Net(nn.Module):
    def __init__(self,data,train):
        super().__init__()
        self.base=temporal.Net(data)
        state=torch.load(PARENT/(train+'.pt'),map_location='cpu',weights_only=False)
        assert state['epoch']==18 and state['config']['name']==PARENT.name
        self.base.load_state_dict(state['net'])
        mat=cf.checkpoint(train)['a'].mean(0);_,v=np.linalg.eigh(mat.T@mat)
        side=mat@v[:,-16:];side/=np.maximum(side.std(0),1e-6)
        self.register_buffer('side',torch.from_numpy(side.clip(-5,5)))
        dim=self.base.film.embedding_dim
        self.token=nn.Sequential(nn.Linear(dim+16+7,48),nn.LayerNorm(48),nn.SiLU())
        self.q=nn.Linear(48,36);self.k=nn.Linear(48,36)
        self.value=nn.Linear(48+5,36)
        self.out=nn.Sequential(nn.Linear(48+36+5,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.12),nn.Linear(64,5))
        self.time=nn.Parameter(torch.tensor([.15,1.,2.]))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
    def forward(self,hi,hr,hd,ti,td,counts,user):
        n=ti.shape[1];b=ti.shape[0]
        present=hr==6
        # The data sampler places unknown tokens after the observed history. Gather a bounded slate.
        pos=torch.topk(present.to(torch.float32),min(20,hr.shape[1]),dim=1,sorted=False).indices
        peers=hi.gather(1,pos);when=hd.gather(1,pos);valid=present.gather(1,pos)
        movies=torch.cat((ti,peers),1);days=torch.cat((td,when),1)
        raw=self.base(hi,hr,hd,movies,days,counts,user)
        pi=raw.softmax(-1)
        geo=torch.cat((self.base.film(movies),self.side[movies.long()]),-1)
        feat=torch.cat((geo,pi,days[:,:,None]/2000.,self.base.pop[movies.long()][:,:,None]),-1)
        state=self.token(feat)
        dt=(days[:,:,None]-when[:,None,:]).abs().float()
        allow=valid[:,None,:]&(movies[:,:,None]!=peers[:,None,:])
        logits=raw
        for _ in range(2):
            prior=logits.softmax(-1)
            qq=self.q(state).reshape(b,-1,3,12).transpose(1,2)
            kk=self.k(state[:,n:]).reshape(b,-1,3,12).transpose(1,2)
            vv=self.value(torch.cat((state[:,n:],prior[:,n:]),2)).reshape(b,-1,3,12).transpose(1,2)
            att=qq@kk.transpose(-1,-2)/math.sqrt(12)
            att-=self.time.clamp(0,5)[None,:,None,None]*torch.log1p(dt[:,None]/12)
            weights=att.masked_fill(~allow[:,None],-10000).softmax(-1)*allow[:,None]
            msg=(weights@vv).transpose(1,2).reshape(b,-1,36)
            logits=raw+self.out(torch.cat((state,msg,prior),2))
        return logits[:,:n]


def train(train,phase,args):
    data=temporal.Data(train,phase,matched_days=True)
    torch.manual_seed(75511);torch.mps.manual_seed(75511)
    model=Net(data,train).to('mps')
    params=[p for name,p in model.named_parameters() if not name.startswith('base.')]
    opt=torch.optim.AdamW([dict(params=model.base.parameters()),dict(params=params)],lr=.0002,weight_decay=.004)
    path=OUT/(train+'.pt');start=time.monotonic();done=0
    if path.exists():
        saved=torch.load(path,map_location='mps',weights_only=False);assert saved['config']==vars(args)
        model.load_state_dict(saved['net']);opt.load_state_dict(saved['opt']);done=saved['epoch']
    stars=torch.arange(1,6,dtype=torch.float32,device='mps')
    for ep in range(done,args.epochs):
        rng=np.random.default_rng(75511+ep);users=rng.permutation(data.ready)
        model.train()
        factor=(1+math.cos(ep*math.pi/args.epochs))/2
        opt.param_groups[0]['lr']=.000045+.00019*factor
        opt.param_groups[1]['lr']=.00010+.00060*factor
        for k in range(args.steps):
            us=users[np.arange(k*args.batch,(k+1)*args.batch)%len(users)]
            values,y=data.batch(us,rng,True)
            logits=model(*values);valid=y>=0;mu=logits.softmax(-1)@stars
            ce=F.cross_entropy(logits.reshape(-1,5),y.reshape(-1),ignore_index=-1)
            loss=.44*ce+.56*(((mu-y-1).square()*valid).sum()/valid.sum().clamp(min=1))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            opt.step()
            if k%256==0:print(train,ep+1,k,round(float(loss.detach().cpu()),4),round(time.monotonic()-start),flush=True)
        tmp=path.with_suffix('.tmp')
        torch.save(dict(net=model.state_dict(),opt=opt.state_dict(),epoch=ep+1,config=vars(args)),tmp);tmp.replace(path)
    preds=first.predict(model,data,args.batch,passes=3);np.save(OUT/(phase+'.npy'),preds)
    return preds


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=9);ap.add_argument('--steps',type=int,default=768)
    ap.add_argument('--batch',type=int,default=224);args=ap.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    fr=json.loads((PARENT/'freeze.json').read_text());assert fr['qualifying_hashes']['selected']==ne.sha(PARENT/'selected.f32')
    y,f,_,_=anchor.history_control();root=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((root/'freeze.json').read_text());assert fr['qualifying_hashes']['selected']==ne.sha(root/'selected.f32')
    base=np.load(root/'known-selected.npy');qb=np.fromfile(root/'selected.f32','<f4')
    v=train('trainx','probex',args)
    initial=np.load(PARENT/'probex.npy')
    key,stats=ne.select(np.column_stack((v-base,v-initial)),y,f,base)
    rep=dict(selected=key,known=stats,config=vars(args),standalone=ne.rmse(v,y),parent=str(root),
             starter_checkpoint={k:ne.sha(PARENT/(k+'.pt')) for k in ('trainx','fulltrain')},qualifying_hashes={},
             qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        v=train('fulltrain','qual',args);path=OUT/'selected.f32'
        reference=qb if int(key)==0 else np.load(PARENT/'qual.npy')
        ne.clipped(qb+stats[key]['weight']*(v-reference)).astype('<f4').tofile(path)
        rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
