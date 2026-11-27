"""Estimate conditional blend weights with training cases reweighted to match observable query exposure."""
import json
import numpy as np
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_schema_receipt as receipt

OUT=ne.ROOT/'own-exposure-balance-20261126'
PARENT=blend.OUT/'paired-03'


def exposure(phase,old_u,old_i,old_d,b):
    train='trainx' if phase=='probex' else 'fulltrain'
    present,_=anchor.training_indices(train,old_u,old_i,old_d)
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')];nu=len(uc)
    anchor_count=np.bincount(old_u[present],minlength=nu);n=anchor_count[qu]
    seen=old_u[present]*4096+old_d[present];addresses=qu.astype('int64')*4096+qd
    same_day=np.isin(addresses,seen);old_count=np.bincount(old_u,minlength=nu)[qu]
    cohort=old_count+np.bincount(ne.history('qual','user_idxs'),minlength=nu)[qu]
    history=np.log1p(uc[qu]);pop=np.log1p(ic[qi]);frac=n/np.maximum(cohort-1,1)
    age=qd.astype('f')/2000;cf=full.cf96(phase)
    gates=np.column_stack((np.ones(len(b)),np.minimum(n,3)/3,same_day,np.tanh(3*(frac-.3)),np.tanh(history-4.7),
                           np.tanh((pop-8)/2),np.tanh((b-3.5)*1.2),np.tanh((cf-b)*4),age-.8,(cohort-5)/4)).astype('f')
    segments=(((np.minimum(n,3)*2+same_day)*3+np.digitize(cohort,[5,8]))*3+np.digitize(history,[4.3,5.8]))*3+np.digitize(pop,[7,10])
    return gates,segments.astype('i')


def experts(rec,known,p,q):
    src=rec['known_sources' if known else 'qual_sources'];chunks=[]
    for name in rec['columns']:
        v,meta=blend.direction(name,p,q,known);assert meta==src[name]
        chunks.append(v.astype('f'))
    return np.column_stack(chunks)


def block(x,gate,sl):
    v=x[sl].clip(-.75,.75);g=gate[sl]
    return (v[:,:,None]*g[:,None,:]).reshape(len(v),-1)


def main():
    OUT.mkdir(exist_ok=True);y,f,p,q=anchor.history_control();fr=receipt.read(PARENT/'freeze.json')
    assert receipt.streamed_sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected'];b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    ou,oi,od,_=anchor.old_probe();g,seg=exposure('probex',ou,oi,od,b);qg,qseg=exposure('qual',ou,oi,od,qb)
    size=max(seg.max(),qseg.max())+1;dev=np.bincount(seg,minlength=size);query=np.bincount(qseg,minlength=size)
    ratio=(((query+30)/(len(qb)+30*size))/((dev+30)/(len(b)+30*size))).clip(.25,4)
    wrow=(.3+.7*ratio[seg]).astype('f');x=experts(fr,True,p,q);dim=x.shape[1]*g.shape[1]
    gram=np.zeros((5,dim,dim));rhs=np.zeros((5,dim))
    for k in range(5):
        ids=np.flatnonzero(f==k)
        for j in range(0,len(ids),8192):
            sl=ids[j:j+8192];mat=block(x,g,sl)*np.sqrt(wrow[sl,None]);gram[k]+=mat.T@mat;rhs[k]+=mat.T@((y-b)[sl]*np.sqrt(wrow[sl]))
        print('exposure fold moments',k,flush=True)
    pen=np.tile(np.array([360,900,900,900,900,900,900,900,1600,1200]),x.shape[1])
    def solve(G,r):
        a=G.copy();a.flat[::len(a)+1]+=pen;return np.linalg.solve(a,r)
    w=solve(gram.sum(0),rhs.sum(0));held=np.empty(len(y),'f')
    for k in range(5):
        wk=solve(gram.sum(0)-gram[k],rhs.sum(0)-rhs[k]);ids=np.flatnonzero(f==k)
        for j in range(0,len(ids),8192):
            sl=ids[j:j+8192];held[sl]=block(x,g,sl)@wk
    stats={str(a):blend.metrics(ne.clipped(b+a*held),y,f) for a in (0,.25,.5,.75,1)}
    good=[k for k in stats if ret.qualifies(stats[k],stats['0'])];choice=min(good,key=lambda k:stats[k]['pooled']) if good else 'control'
    rep=dict(parent=str(PARENT),selected=choice,known=stats,ratio_bounds=[.25,4],weights=w.tolist(),total_bins=int(size),
             query_mass_without_development_support=float(query[dev==0].sum()/len(qb)),mean_row_weight=float(wrow.mean()),
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if good:
        del x
        qx=experts(fr,False,p,q);delta=np.zeros(len(qb),'f')
        for j in range(0,len(qb),8192):
            sl=slice(j,j+8192);delta[sl]=block(qx,qg,sl)@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
