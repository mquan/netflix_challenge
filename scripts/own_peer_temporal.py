"""Replay one trainx attention network for each possible target with zero, native and all other observed Probe peers."""
import ctypes
import json
import math
import subprocess
import numpy as np
import torch
import own_hybrid_attention as hybrid
import own_neighbor_experiment as ne
import own_peer_replay as peer
import own_retained_history as ret
import own_temporal_attention as tm
OUT=ne.ROOT/'own-peer-temporal-20261126'
STAR=torch.arange(1,6,dtype=torch.float32)


def forward(net,hm,hr,hd,cm,cd,cr,actual,user,count,valid,vis):
    b,h=hm.shape;n=cm.shape[1];t=net.film(cm);he=net.film(hm);dim=tm.WIDTH;heads_count=tm.HEADS
    def encode(emb,m,r,d):
        known=(r>0)&(r<6);center=(r.float()-3.5)*known/1.4
        side=torch.stack((net.pop[m],net.year[m],d/2000.,(r==6).float()),-1)
        return net.encoder(torch.cat((emb,emb*center[:,:,None],net.star(r),side),-1)).chunk(2,-1)
    hk,hv=encode(he,hm,hr,hd);ak,av=encode(t,cm,cr,cd);uk,uv=encode(t,cm,torch.full_like(cr,6),cd)
    key=torch.cat((hk,ak,uk),1);value=torch.cat((hv,av,uv),1)
    mov=torch.cat((hm,cm,cm),1);date=torch.cat((hd,cd,cd),1);rating=torch.cat((hr,cr,torch.full_like(cr,6)),1)
    def heads(v):return v.reshape(b,-1,heads_count,dim//heads_count).transpose(1,2)
    raw=heads(net.query(t))@heads(key).transpose(-1,-2)/math.sqrt(dim//heads_count)
    distance=(cd[:,:,None]-date[:,None,:]).abs().float()
    raw=raw-net.time_gate.clamp(0,5)[None,:,None,None]*torch.log1p(distance[:,None]/5)
    raw=raw+net.hidden_gate[None,:,None,None]*(rating==6)[:,None,None,:]
    indicator=(rating>0)&(rating<6);center=(rating.float()-3.5)*indicator/1.4
    hist=torch.stack([(rating==k).float() for k in range(1,6)],2)
    out=[]
    for mask in vis:
        choose=mask&actual[:,None,:]&valid[:,None,:]
        available=torch.cat(((hr>0)[:,None,:].expand(-1,n,-1),choose,(~choose)&valid[:,None,:]),2)
        usable=available&(cm[:,:,None]!=mov[:,None,:])
        w=raw.masked_fill(~usable[:,None],-10000).softmax(-1)*usable[:,None]
        attend=(w@heads(value)).transpose(1,2).reshape(b,n,dim)
        known=available&indicator[:,None,:];weights=known.float()/known.sum(-1,keepdim=True).clamp(min=1)
        size=count[:,None]+choose.sum(-1)
        stat=torch.cat((weights@value,net.user(user)[:,None].expand(-1,n,-1),weights@hist,size.log1p()[:,:,None]/7),2)
        same=(distance==0)&usable;obs=same&known;hits=obs.sum(-1)
        dyn=torch.stack((cd/2000.,net.pop[cm],net.year[cm],same.sum(-1).log1p()/4,hits.log1p()/4,
                   (obs*center[:,None,:]).sum(-1)/hits.clamp(min=1)),2)
        logits=net.bias(cm)+net.head(torch.cat((t,attend,stat,dyn),2));out.append(logits.softmax(-1)@STAR.to('mps'))
    return torch.stack(out,-1)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);c=peer.Cohort();d=c.d
    raw=np.load(ret.QUAL/'dataset/dense-to-raw-user.npy');uid=ne.history('trainx','uidx_to_uid')
    ids=np.searchsorted(uid,raw).astype('i');np.testing.assert_array_equal(uid[ids],raw)
    data=tm.Data('trainx','probex',True,True);net=tm.Net(data).to('mps');cp=hybrid.WARM/'trainx.pt'
    state=torch.load(cp,map_location='mps',weights_only=False);assert state['epoch']==16 and state['config']['cf_factors']
    net.load_state_dict(state['net']);net.eval();del state,data
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_peer_temporal.cpp','-o',str(OUT/'rows.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'rows.dylib').resolve()));lib.background_rows.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*9;lib.background_rows.restype=None
    result=np.zeros((len(c.rows),3),'f');users=np.flatnonzero(c.grid[:,0]>=0).astype('i');cv=lambda x:torch.as_tensor(np.ascontiguousarray(x),device='mps')
    check=0.;cap=tm.H-9
    for r in range(2):
        for start in range(0,len(users),256):
            us=users[start:start+256];g=c.grid[us];valid=g>=0;rows=g.clip(min=0)
            m=c.m[rows];day=c.day[rows].astype('i');star=c.rows['rating'][rows].astype('i')
            mask=(rows[:,:,None]!=rows[:,None,:])
            actual=(rows<c.n)&valid;native=actual&c.seen[rows.clip(max=c.n-1)]
            visibility=[np.zeros_like(mask),mask&native[:,None,:],mask&actual[:,None,:]]
            mid=np.nanmedian(np.where(valid,day,np.nan),axis=1).astype('i')
            hm,hd,hr=[np.zeros((len(us),cap),'i') for _ in range(3)]
            args=(us,d['train_offsets'],d['train_items'],d['train_days'],d['train_ratings'],mid,hm,hd,hr)
            lib.background_rows(len(us),cap,r,*[a.ctypes.data for a in args])
            count=d['train_offsets'][us+1]-d['train_offsets'][us]
            vals=list(map(cv,(hm,hr,hd,m,day-314,star,actual,ids[us],count.astype('f'),valid)))
            value=forward(net,*vals,list(map(cv,visibility)))
            if start==0 and r==0:
                # Zero observed peers is an ordinary forward call on the same background and unknown movie tokens.
                im,ir,it=vals[:3];cm,cd=vals[3:5]
                q=net(torch.cat((im,cm),1),torch.cat((ir,6*vals[-1].int()),1),torch.cat((it,cd),1),cm,cd,vals[-2],vals[-3])
                check=float((value[:,:,0]-(q.softmax(-1)@STAR.to('mps'))).abs().max().cpu());assert check<.0001,check
            result[rows[valid]]+=value.cpu().numpy()[valid]/2
            if start%(256*550)==0:print('history replay',r,start,flush=True)
    assert np.isfinite(result).all();np.save(OUT/'means.npy',result)
    (OUT/'manifest.json').write_text(json.dumps(dict(checkpoint_sha256=ne.sha(cp),means_sha256=ne.sha(OUT/'means.npy'),
           maximum_zero_mask_native_difference=check,query_self_excluded=True,qualifying_ratings_read=False),indent=2)+'\n')


if __name__=='__main__':
    with torch.inference_mode():main()
