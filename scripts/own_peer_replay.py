"""Replay original-history neural predictors on all subsets of observable competition peers."""
from pathlib import Path
import json
import numpy as np
import torch
from gbdt_blend import load,PROBE
import history_model as hn
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_retained_history as ret

OUT=ne.ROOT/'own-peer-replay-20261126'
DATA=Path('data/processed/netflix.bin')
MODELS=('HN1-history','HN2-sparse','HN3-session')


class Cohort:
    def __init__(self):
        receipt=json.loads((joint.OUT/'manifest.json').read_text());assert ne.sha(DATA)==receipt['training_data_sha256']
        self.d=load(DATA);probe=self.d['probe'];self.n=len(probe)
        q=np.fromfile(joint.OUT/'zero-queries.bin',PROBE);assert len(q)==2817131 and not q['rating'].any()
        self.rows=np.concatenate((probe,q));self.u=self.rows['user'];self.m=self.rows['item'].astype('i')-1;self.day=self.rows['day']
        self.ix=ret.map_rows();self.seen=np.ones(self.n,bool);self.seen[self.ix]=False
        self.ref=json.loads((joint.OUT/'manifest.json').read_text())
        def grid(users):
            order=np.argsort(users,kind='stable');count=np.bincount(users,minlength=self.d['users']);off=np.r_[0,count.cumsum()]
            slot=np.arange(len(users))-off[users[order]]
            tab=np.full((len(count),count.max()),-1,'i');tab[users[order],slot]=order
            return tab
        self.anchors=grid(self.u[:self.n]);self.grid=grid(self.u)
        assert self.anchors.shape[1]==9 and self.grid.shape[1]==9

    def masks(self,rows):
        peers=self.anchors[self.u[rows]];idx=peers.clip(min=0)
        valid=(peers>=0)&(peers!=rows[...,None])
        dist=abs(self.day[idx].astype('i')-self.day[rows,None].astype('i'))
        near=np.argsort(np.argsort(np.where(valid,dist,30000),axis=-1,kind='stable'),axis=-1,kind='stable')
        far=np.argsort(np.argsort(np.where(valid,-dist,30000),axis=-1,kind='stable'),axis=-1,kind='stable')
        parity=(self.m[idx]*131+self.day[idx])%2==0
        return idx,[np.zeros_like(valid),valid&self.seen[idx],valid,valid&(near<1),
            valid&(near<2),valid&(far<1),valid&parity,valid&~parity]

    def make(self):
        models=[];refs=[];lineage={};cap=None
        for n in MODELS:
            p=Path('experiments',n+'.json');r=json.loads(p.read_text());cmd=r['command']
            assert cmd[1]=='scripts/history_model.py' and cmd[cmd.index('--data')+1]==str(DATA) and '--init-model' not in cmd
            md,pr=[r[k] for k in ('model_artifact','predictions_artifact')]
            assert ne.sha(md['path'])==md['sha256'] and ne.sha(pr['path'])==pr['sha256']
            state=torch.load(md['path'],weights_only=False,map_location='cpu');cfg=state['config']
            if cap is None:cap=cfg['context']
            assert cap==cfg['context']
            model=hn.HistoryNet(self.d['items'],cfg['rank'],cfg['hidden'],cfg['arm']=='session')
            model.load_state_dict(state['state']);models.append(model.eval());refs.append(np.fromfile(pr['path'],'<f4'))
            lineage[n]=dict(receipt=str(p),receipt_sha256=ne.sha(p),model=md,original_predictions=pr)
        out=np.lib.format.open_memmap(OUT/'effects.npy',mode='w+',dtype='f',shape=(len(self.rows),3,8))
        user=np.flatnonzero(self.grid[:,0]>=0)
        stars=torch.arange(1,6,dtype=torch.float32)
        cv=lambda a:torch.from_numpy(np.ascontiguousarray(a))
        for start in range(0,len(user),384):
            us=user[start:start+384];g=self.grid[us];good=g>=0;rows=g.clip(min=0)
            ar=self.anchors[us];active=ar>=0;ar=ar.clip(min=0)
            _,visibility=self.masks(rows)
            batch=hn.batch(self.d,us,None,cap,1,training=False)
            tokens,day=cv(batch['token']),cv(batch['days'])
            m=cv(self.m[rows].astype('int64'));dt=cv(self.day[rows].astype('f'))
            a=cv(np.where(active,1+self.m[ar]*5+self.rows['rating'][ar]-1,0).astype('int64'))
            ad=cv(self.day[ar].astype('f'));cnt=(tokens>0).sum(1).float()
            for j,model in enumerate(models):
                em=model.input(tokens);am=model.input(a);acc=em.sum(1)
                decoder=model.output(m).reshape(*m.shape,5,-1);bias=model.bias(m)
                if model.session:
                    near=(-(dt[:,:,None]-day[:,None,:]).abs()/30).exp()*(tokens>0)[:,None]
                    local=near@em;mass=near.sum(2)
                    coef=(-(dt[:,:,None]-ad[:,None,:]).abs()/30).exp()
                vals=np.empty((*m.shape,8),'f')
                for mask,visible in enumerate(visibility):
                    take=cv(visible).float()
                    n=(cnt[:,None]+take.sum(2)).clamp(min=1)
                    pooled=(acc[:,None]+take@am)/n.sqrt()[:,:,None]
                    state=model.encoder(torch.cat((pooled,n.log1p()[:,:,None]/5),2))
                    if model.session:
                        weights=coef*take
                        state=state+model.local((local+weights@am)/(mass+weights.sum(2)).clamp(min=1).sqrt()[:,:,None])
                    logits=(decoder*state[:,:,None,:]).sum(-1)+bias
                    vals[:,:,mask]=(logits.softmax(-1)@stars).numpy()
                out[rows[good],j]=vals[good]
            if start%(384*200)==0:print('replayed users',start,flush=True)
        out.flush()
        for k,n in enumerate(MODELS):
            diff=float(np.max(abs(out[:self.n,k,0]-refs[k])));assert diff<.0004,(n,diff)
            lineage[n]['maximum_original_replay_difference']=diff
        assert np.isfinite(out).all()
        info=dict(models=lineage,output_sha256=ne.sha(OUT/'effects.npy'),query_self_excluded=True,qualifying_ratings_read=False)
        (OUT/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    with torch.inference_mode():Cohort().make()
