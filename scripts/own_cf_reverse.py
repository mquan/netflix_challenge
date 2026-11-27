"""Score each possible hidden star by how it predicts a user's actual nearby ratings with CFNADE."""
import ctypes
import json
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
from own_query_sets import projection

OUT=ne.ROOT/'own-cf-reverse-20261126'
STARS=np.arange(1,6,dtype='f')


def softmax(x):
    z=np.exp(x-x.max(-1,keepdims=True));return z/z.sum(-1,keepdims=True)


def features(train,phase,base):
    lib=ctypes.CDLL(str((OUT/'reverse.dylib').resolve()));ni=ctypes.c_int;pt=ctypes.c_void_p
    lib.reverse_neighbors.argtypes=[ni]+[pt]*8;lib.cf_totals.argtypes=[ni,ni]+[pt]*7
    lib.reverse_neighbors.restype=lib.cf_totals.restype=None
    m,d,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    u,i,day=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    off=np.r_[0,uc.cumsum()].astype('i')
    s=cf.checkpoint(train)
    enc=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
    tot=np.empty((len(uc),96),'f');md=np.empty(len(uc),'f')
    def ptrs(*a):return [v.ctypes.data for v in a]
    lib.cf_totals(len(uc),len(ic),*ptrs(off,m,d,r,enc,tot,md))
    avg=d.mean(dtype=float);md[uc==0]=avg
    sd=(sum(np.sum((d[j:j+1000000].astype(float)-avg)**2) for j in range(0,len(d),1000000))/len(d))**.5
    coord=projection(s['p'][2:].mean(0),16)
    coord=np.ascontiguousarray(coord/np.maximum(np.linalg.norm(coord,axis=1,keepdims=True),1e-7))
    ep=np.ascontiguousarray((enc@s['enc_proj'].T).transpose(1,0,2))
    cp=s['p'].cumsum(0,dtype=float).astype('f')
    op=np.ascontiguousarray((cp@s['dec_proj']).transpose(1,0,2))
    del enc,cp
    tot=tot@s['enc_proj'].T
    cb,ub,ud=[np.ascontiguousarray(s[k].cumsum(0,dtype=float).T,'f') for k in ('b','user_bias','user_drift')]
    def bias(items,us,dates):
        diff=dates-md[us]
        dev=np.sign(diff)*np.power(np.abs(diff)/sd,.4)
        return cb[items]+ub[us]+ud[us]*dev[...,None]
    original=full.cf96(phase);result=np.zeros((len(i),59),'f')
    largest=0.
    for start in range(0,len(i),768):
        sl=slice(start,min(start+768,len(i)));qu,qi,qd=u[sl],i[sl],day[sl];n=len(qi)
        ids=np.full((n,2,8),-1,'i')
        lib.reverse_neighbors(n,*ptrs(qu,qi,qd,off,m,d,coord,ids))
        count=uc[qu];total=tot[qu];direct=bias(qi,qu,qd)
        dec=op[qi];enc=ep[qi]
        def hidden(v,count):
            return np.tanh(v/np.sqrt(np.maximum(1,count)).astype('f')[...,None]+s['enc_bias'])
        def decode(h):return direct+np.einsum('bsh,bh->bs',dec,h)
        norm=hidden(total,count);forward=softmax(decode(norm))
        largest=max(largest,float(np.max(abs(forward@STARS-original[sl]))))
        assert largest<.0003,largest
        blocks=[forward]
        for j in range(2):
            rr=ids[:,j];mask=rr>=0;rr=rr.clip(min=0);cnt=mask.sum(1)
            peers=m[rr];stars=r[rr]-1
            vis=(ep[peers,stars]*mask[:,:,None]).sum(1)
            remaining=total-vis
            prior=decode(hidden(remaining,count-cnt))
            conditioned=hidden(remaining[:,None]+enc,(count-cnt+1)[:,None])
            observed=op[peers]
            logits=(conditioned@observed.reshape(n,8*5,-1).transpose(0,2,1)).reshape(n,5,8,5)
            logits+=bias(peers,qu[:,None],d[rr])[:,None,:,:]
            logprob=logits-np.logaddexp.reduce(logits,axis=-1,keepdims=True)
            actual=np.take_along_axis(logprob,stars[:,None,:,None],axis=3)[:,:,:,0]
            score=(actual*mask[:,None,:]).sum(2)
            slope=score-score.mean(1,keepdims=True)
            near=(np.abs(d[rr].astype('i')-qd[:,None])<=7)&mask
            quick=(actual*near[:,None,:]).sum(2)
            probs=[softmax(prior),softmax(prior+.5*slope),softmax(prior+slope),softmax(prior+.8*quick),
                   softmax(np.log(np.maximum(forward,1e-16))+.5*slope)]
            blocks.extend([(pr-forward)[:,:4] for pr in probs])
            blocks.extend([slope[:,:4],cnt,near.sum(1),probs[2]@STARS])
        result[sl]=np.column_stack(blocks)
        if start%(768*400)==0:print(phase,start,'native error',largest,flush=True)
    assert np.isfinite(result).all()
    return np.column_stack((base,original-base,ne.verified_rank(phase)-base,np.log1p(uc[u]),np.log1p(ic[i]),result)).astype('f')


def model():
    return CatBoostRegressor(iterations=1000,depth=5,learning_rate=.045,l2_leaf_reg=55,thread_count=2,
        random_seed=842,verbose=False,allow_writing_files=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_cf_reverse.cpp','scripts/own_cf_adapter.cpp',
                    '-o',str(OUT/'reverse.dylib')],check=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('trainx','probex',base)
    key,stats,held=anchor.crossfit(x,y,f,base,model);np.save(OUT/'oof.npy',held)
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features('fulltrain','qual',qb)
        delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta);path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
