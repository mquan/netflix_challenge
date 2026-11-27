"""Vary checkpoint and expert coefficients with regularized user and movie context interactions."""
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_history_context as hist
import own_retained_history as retained
import own_producer_bank as producer
import own_trajectory_bank as trajectory

OUT=ne.ROOT/'own-tensor-ridge-20261126'
PENS=(5000.,35000.,180000.)


def components(phase,keys,p,q,base,params=None):
    known=phase=='probex'
    chunks=[blend.direction(k,p,q,known)[0].astype('f') for k in keys]
    core=np.column_stack(chunks)
    dep=[np.load(d/(phase+'.npy'),mmap_mode='r') for d in (producer.OUT,trajectory.OUT)]
    # Include every second principal component and the eight leading directions from each bank.
    pc=sorted(set(range(0,64,2))|set(range(56,64)))
    expert=np.column_stack([np.ones(len(base)),core]+[a[:,pc] for a in dep]).astype('f')
    ctx=np.load(hist.OUT/(phase+'.npy'),mmap_mode='r')
    raw=np.column_stack((base,ctx[:,1:9],ctx[:,10:16],np.std(core,axis=1),
                         np.abs(core[:,0]),core[:,0]-core[:,1])).astype('f')
    support=json.loads((trajectory.OUT/'sources.json').read_text())['support']
    # A common, locally trained factor system provides both sets of coordinates.
    z=np.column_stack([np.load(info['paths'][int(not known)],mmap_mode='r') for info in support.values()])
    if params is None:
        zs=z[::16]
        mu=zs.mean(0)
        sd=np.maximum(zs.std(0),1e-5)
        zz=(zs-mu)/sd
        eig,vec=np.linalg.eigh(zz.T@zz/len(zz))
        rot=vec[:,-8:]/np.sqrt(np.maximum(eig[-8:],1e-5))
        sample=expert[::14]
        es=np.maximum(sample.std(0),.02)
        em=sample.mean(0)
        es[0]=1;em[0]=0
        hm=np.median(raw[::14],axis=0)
        hs=np.maximum(raw[::14].std(0),.01)
        params=(mu,sd,rot,em,es,hm,hs)
    mu,sd,rot,em,es,hm,hs=params
    expert-=em
    expert/=es
    np.clip(expert,-4,4,out=expert)
    raw=np.tanh((raw-hm)/hs)
    zg=np.empty((len(base),8),'f')
    for j in range(0,len(base),32768):
        sl=slice(j,j+32768)
        zg[sl]=np.tanh(((z[sl]-mu)/sd)@rot)
    gate=np.column_stack((np.ones(len(base)),raw,zg)).astype('f')
    return expert,gate,params


def design(ex,gate,idx):
    return (ex[idx,:,None]*gate[idx,None,:]).reshape(len(idx),-1)


def main():
    OUT.mkdir(exist_ok=True)
    y,f,p,q=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    ex,gate,params=components('probex',rec['columns'],p,q,base)
    n=ex.shape[1]*gate.shape[1]
    gs=[];bs=[]
    err=(y-base).astype('f')
    for j in range(5):
        rows=np.flatnonzero(f==j)
        g=np.zeros((n,n),'d');b=np.zeros(n)
        for start in range(0,len(rows),2048):
            ix=rows[start:start+2048]
            x=design(ex,gate,ix)
            g+=x.T@x
            b+=x.T@err[ix]
        gs.append(g);bs.append(b)
        print('fold statistics',j,n,flush=True)
    G=sum(gs);B=sum(bs)
    def solve(g,b,pen):
        g=g.copy();g.flat[::len(g)+1]+=pen
        return cho_solve(cho_factor(g,check_finite=False),b,check_finite=False)
    held=np.zeros((len(y),len(PENS)),'f')
    for j in range(5):
        w=np.column_stack([solve(G-gs[j],B-bs[j],t) for t in PENS]).astype('f')
        rows=np.flatnonzero(f==j)
        for start in range(0,len(rows),4096):
            ix=rows[start:start+4096]
            held[ix]=design(ex,gate,ix)@w
    orig=blend.metrics(base,y,f)
    stats={str(t):blend.metrics(ne.clipped(base+held[:,j]),y,f) for j,t in enumerate(PENS)}
    ok=[j for j,t in enumerate(PENS) if retained.qualifies(stats[str(t)],orig)]
    best=min(ok,key=lambda j:stats[str(PENS[j])]['pooled']) if ok else None
    report=dict(selected=str(PENS[best]) if ok else 'control',known=stats,control=orig,parent=str(parent),nfeatures=n,
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        np.save(OUT/'oof.npy',held[:,best])
        del ex,gate,gs,bs
        ex,gate,_=components('qual',rec['columns'],p,q,qb,params)
        w=solve(G,B,PENS[best]).astype('f')
        delta=np.empty(len(qb),'f')
        for a in range(0,len(qb),4096):
            ix=np.arange(a,min(a+4096,len(qb)))
            delta[ix]=design(ex,gate,ix)@w
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(qb+delta.astype(float)).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
