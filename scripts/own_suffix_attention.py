"""Train an ordinal attention model on earlier replicas of the Netflix last-nine-rating split."""
import ctypes
import json
import math
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from gbdt_blend import load
import own_anchor_context as anchor
import own_masked_attention as initial
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_slate_bayes as slate
import own_temporal_attention as temporal
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-suffix-attention-20261126'
PARENT=ne.ROOT/'own-temporal-attention-20261126'


def audit(lib):
    d=load(slate.KNOWN);bag=slate.OUT/'cohort.bin'
    assert ne.sha(bag)==json.loads((slate.OUT/'setup.json').read_text())['cohort_sha256']
    with bag.open('rb') as file:
        o=np.fromfile(file,'<u8',d['users']+1);i=np.fromfile(file,'<u2',int(o[-1]));day=np.fromfile(file,'<u2',int(o[-1]))
    result=np.zeros(5,'u8')
    lib.verify_suffix.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*7;lib.verify_suffix.restype=None
    lib.verify_suffix(d['users'],d['items'],*[x.ctypes.data for x in (d['implicit_offsets'],d['implicit_items'],d['implicit_days'],o,i,day,result)])
    assert result[2]==4225526 and result[3]==0
    info=dict(users=d['users'],exposures=len(d['implicit_items']),cohort_rows=int(result[2]),users_with_noncohort_after_boundary=int(result[0]),
        noncohort_addresses_after_boundary=int(result[1]),addresses_strictly_after_boundary=int(result[4]),ratings_used=False)
    (OUT/'chronology.json').write_text(json.dumps(info,indent=2)+'\n')
    print(info,flush=True)
    assert result[0]<d['users']*.001


class Data(temporal.Data):
    def __init__(self,train,phase,lib):
        super().__init__(train,phase,matched_days=True);self.lib=lib
        a,_,d,_=anchor.old_probe();u=np.r_[a,ne.history('qual','user_idxs')];day=np.r_[d,ne.history('qual','dates')]
        self.cohort=np.bincount(u,minlength=len(self.counts)).astype('i')
        self.hidden=np.diff(self.ioff).astype('i')
        self.boundary=np.full(len(self.counts),30000,'i2');np.minimum.at(self.boundary,u,day)
        self.off=np.ascontiguousarray(self.off,'i8')
        assert self.i.dtype==self.d.dtype==np.int16 and self.r.dtype==np.int8
    def masking(self,us,seed):
        b=len(us);h=temporal.H;k=9
        hi,hr,hd=[np.zeros((b,h),'i') for _ in range(3)]
        ti,td=[np.zeros((b,k),'i') for _ in range(2)];label=np.full((b,k),-1,'i');counts=np.zeros(b,'f')
        us=np.ascontiguousarray(us,'i')
        arr=(us,self.off,self.i,self.d,self.r,self.boundary,self.cohort,self.hidden,hi,hr,hd,ti,td,label,counts)
        self.lib.suffix_masks(b,h,k,seed,*[x.ctypes.data for x in arr])
        assert (label>=0).sum(1).min()>0
        cv=lambda a:torch.as_tensor(a,device='mps')
        return tuple(map(cv,(hi,hr,hd,ti,td,counts,us))),cv(label).long()


def run(train,phase,lib):
    data=Data(train,phase,lib);net=temporal.Net(data).to('mps')
    warm=PARENT/(train+'.pt');source=torch.load(warm,map_location='mps',weights_only=False)
    assert source['config']['name']==PARENT.name and source['epoch']==18
    net.load_state_dict(source['net']);del source
    torch.manual_seed(21591);torch.mps.manual_seed(21591)
    opt=torch.optim.AdamW(net.parameters(),lr=.00028,weight_decay=.008)
    model=OUT/(train+'.pt');done=0;epochs=12
    if model.exists():
        state=torch.load(model,map_location='mps',weights_only=False);net.load_state_dict(state['net']);opt.load_state_dict(state['opt']);done=state['epoch']
    started=time.monotonic();star=torch.arange(1,6,device='mps',dtype=torch.float32)
    for ep in range(done,epochs):
        rng=np.random.default_rng(81952+ep);ids=rng.permutation(data.ready);net.train()
        opt.param_groups[0]['lr']=.000055+.00030*(1+math.cos(ep*math.pi/epochs))/2
        for step in range(832):
            us=ids[np.arange(step*224,(step+1)*224)%len(ids)]
            args,y=data.masking(us,int(rng.integers(1,2**30)));logits=net(*args);active=y>=0;mu=logits.softmax(2)@star
            loss=.58*((mu-y-1).square()*active).sum()/active.sum()+.42*F.cross_entropy(logits.reshape(-1,5),y.reshape(-1),ignore_index=-1)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.5);opt.step()
        tmp=model.with_suffix('.tmp')
        torch.save(dict(net=net.state_dict(),opt=opt.state_dict(),epoch=ep+1,parent_hash=ne.sha(warm),
                   config=dict(epochs=epochs,name=OUT.name,chronological=True)),tmp);tmp.replace(model)
        print(train,'epoch',ep+1,round(time.monotonic()-started),flush=True)
    pred=initial.predict(net,data,256,passes=3);np.save(OUT/(phase+'.npy'),pred);return pred


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    subprocess.run(['c++','-O3','-std=c++17','-fPIC','-shared','scripts/own_suffix_attention.cpp','-o',str(OUT/'suffix.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'suffix.dylib').resolve()));lib.suffix_masks.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*15
    lib.suffix_masks.restype=None;audit(lib)
    y,f,_,_=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    base=np.load(parent/'known-selected.npy');fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    v=run('trainx','probex',lib);w0=np.load(PARENT/'probex.npy')
    x=np.column_stack((v-base,v-w0));held=np.empty(len(y))
    for j in range(5):
        tr=f!=j;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(parent),selected='candidate' if ok else 'control',control=ctrl,known=known,weights=w.tolist(),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        qb=np.fromfile(parent/'selected.f32','<f4').astype(float);v=run('fulltrain','qual',lib)
        delta=np.column_stack((v-qb,v-np.load(PARENT/'qual.npy')))@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
