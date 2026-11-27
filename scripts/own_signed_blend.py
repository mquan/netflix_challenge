"""Fit regularized signed expert corrections and smooth context-varying coefficients."""
import argparse
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import own_anchor_context as anchor
import own_history_context as hist
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained


def build(phase,columns,p,q,base):
    known=phase=='probex'
    experts=np.column_stack([blend.direction(k,p,q,known)[0].astype('f') for k in columns])
    ctx=np.load(hist.OUT/(phase+'.npy'),mmap_mode='r')
    train='trainx' if known else 'fulltrain'
    u,i,day=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    qcount=np.bincount(u,minlength=len(uc))
    # Prespecified smooth gates only use features present in both deployment and validation.
    vals=np.column_stack((base,np.log1p(uc[u]),np.log1p(ic[i]),day,np.log1p(qcount[u]),ctx[:,13],
                          ctx[:,7],np.std(experts,axis=1),ne.verified_rank(phase)-base)).astype('f')
    return experts,vals


def make(rows,experts,ctx,transform,mode):
    em,es,cm,cs=transform
    d=np.clip((experts[rows]-em)/es,-12,12)
    cv=((ctx[rows]-cm)/cs).clip(-5,5)
    gate=np.tanh(cv)
    if mode=='linear':
        return np.column_stack((np.ones(len(d),'f'),d,gate))
    hinges=np.column_stack((gate,np.tanh(1.8*(cv[:,0]-.8)),np.tanh(1.8*(cv[:,0]+.8)),
         np.tanh(cv[:,1]+cv[:,2]),gate[:,0]*gate[:,1],gate[:,0]*gate[:,7])).astype('f')
    return np.column_stack((np.ones(len(d),'f'),d,hinges,.45*(d[:,:,None]*hinges[:,None,:]).reshape(len(d),-1)))


def solve(g,b,pen):
    z=g.copy();z.flat[::len(z)+1]+=pen
    return cho_solve(cho_factor(z,check_finite=False),b,check_finite=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='own-signed-blend-20261126');args=ap.parse_args()
    out=ne.ROOT/args.name;out.mkdir(exist_ok=True)
    y,f,p,q=anchor.history_control();parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text());assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    cols=fr['columns'];ex,ctx=build('probex',cols,p,q,base)
    transform=(np.median(ex[::12],0),np.maximum(ex[::12].std(0),.015),np.median(ctx[::12],0),np.maximum(ctx[::12].std(0),.04))
    control=blend.metrics(base,y,f);stats={};est={};full={};err=y-base
    for mode in ('linear','varying'):
        n=make(slice(0,1),ex,ctx,transform,mode).shape[1]
        grams=[];rhs=[]
        for fold in range(5):
            ix=np.flatnonzero(f==fold);g=np.zeros((n,n));b=np.zeros(n)
            for j in range(0,len(ix),4096):
                rows=ix[j:j+4096];x=make(rows,ex,ctx,transform,mode)
                g+=x.T@x;b+=x.T@err[rows]
            grams.append(g);rhs.append(b)
        total=sum(grams);right=sum(rhs)
        for pen in (900.,9000.):
            name=mode+'-'+str(int(pen));held=np.zeros(len(y));w=solve(total,right,pen)
            for fold in range(5):
                ww=solve(total-grams[fold],right-rhs[fold],pen)
                ix=np.flatnonzero(f==fold)
                for j in range(0,len(ix),8192):
                    rows=ix[j:j+8192];held[rows]=make(rows,ex,ctx,transform,mode)@ww
            stats[name]=blend.metrics(ne.clipped(base+held),y,f);est[name]=held;full[name]=(mode,w)
            print(name,stats[name]['pooled'],flush=True)
    ok=[key for key in stats if retained.qualifies(stats[key],control)];best=min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    rep=dict(selected=best,known=stats,control=control,parent=str(parent),columns=cols,qualifying_hashes={},
             qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (out/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        np.save(out/'oof.npy',est[best]);del ex,ctx,est
        ex,ctx=build('qual',cols,p,q,qb);mode,w=full[best];delta=np.empty(len(qb))
        for j in range(0,len(qb),8192):
            sl=slice(j,j+8192);delta[sl]=make(sl,ex,ctx,transform,mode)@w
        np.save(out/'qual-correction.npy',delta);path=out/'selected.f32'
        ne.clipped(qb+delta).astype('<f4').tofile(path);rep['qualifying_hashes']['selected']=ne.sha(path)
    (out/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(out,ne.LABELS)


if __name__=='__main__':main()
