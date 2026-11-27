"""Recompute conditional free energies and peer messages from our trained 400-hidden RBM."""
import json
import sys
import numpy as np
import torch
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_cf_messages as msg
import own_native_holdout as native
import own_neighbor_experiment as ne
import own_overnight_blend as blend

UP=native.UP
sys.path.append(str((UP/'scripts').resolve()))
from upstream_factored_rbm_rows import UserRows
from upstream_factored_rbm_admission import phase_paths

OUT=ne.ROOT/'own-rbm-conditional-20261126'
STUDY='upstream-factored-rbm-gpu400-deterministic-metal'
EXP=UP/'experiments'/STUDY


def inputs(phase):
    paired=EXP/'paired-verification.json'
    assert ne.sha(paired)=='3e8d2ecdbb40d0c953f98b24601c2e84a9353bb45ca23d74fea26797a814199a'
    rec=json.loads(paired.read_text())
    assert not rec['qualifying_labels_read'] and not rec['recovered_outcome_paths_read']
    admission=EXP/'data-admission.json'
    assert ne.sha(admission)==rec['data_admission_sha256']
    data=json.loads(admission.read_text())
    plan=UP/'experiments/upstream-factored-rbm-inputs/plan.json'
    assert ne.sha(plan)==data['input_declaration_sha256']
    decl=json.loads(plan.read_text());arrays=[]
    for paths in phase_paths(decl,phase):
        group={}
        for k,p in paths.items():
            assert ne.sha(UP/p)==data['bound_file_records'][p]['sha256']
            group[k]=np.load(UP/p,mmap_mode='r')
        arrays.append(group)
    reader=UserRows(*arrays,decl['items'])
    paths={}
    for k in (phase+'.pt',phase+'-native.npy'):
        p=UP/'artifacts'/STUDY/k
        assert ne.sha(p)==rec['actual_model_payloads'][phase][str(p.relative_to(UP))]['sha256']
        paths[k]=p
    model=torch.load(paths[phase+'.pt'],map_location='cpu',weights_only=True)['model']
    return reader,{k:v.numpy() for k,v in model.items()},np.load(paths[phase+'-native.npy'],mmap_mode='r')


def grouped(value,ids,n):
    out=np.zeros((n,value.shape[1]),'f')
    if len(ids):
        first=np.r_[0,np.flatnonzero(ids[:-1]!=ids[1:])+1]
        out[ids[first]]=np.add.reduceat(value,first,axis=0)
    return out


def probabilities(logits):
    ex=np.exp(logits-logits.max(1,keepdims=True))
    return ex/ex.sum(1,keepdims=True)


def features(phase,base):
    reader,s,original=inputs(phase)
    p,q=s['p'],s['q']
    w=(p.reshape(-1,q.shape[0])@q).reshape(p.shape[0],5,-1)
    # log cosh((a+w)/2) - log cosh(a/2), computed without subtracting nearly equal large values.
    coshm1=2*np.sinh(w*.25)**2
    sinh=np.sinh(w*.5)
    prior=(reader.visible_counts+1)/(reader.visible_counts.sum(1,keepdims=True)+5)
    priorp=np.einsum('mr,mrf->mf',prior.astype('f'),p)
    ix=native.identities()[-1] if phase=='known' else np.arange(len(base))
    native_base=original.copy();native_base[ix]=base
    result=np.empty((len(original),46),'f')
    ids=np.flatnonzero(reader.qr['user_cnts'])
    largest=0.
    for start in range(0,len(ids),192):
        batch=reader.batch(ids[start:start+192])
        g=batch['global_users'];sup=batch['support'];qr=batch['query']
        i,u,r=(sup[k] for k in ('items','users','ratings'))
        z=grouped(p[i,r],u,len(g))
        cond=grouped(s['d'][sup['conditional_items']],sup['conditional_users'],len(g))
        a=s['bh']+z@q+cond
        qi,qu,days=(qr[k] for k in ('items','users','days'));rows=batch['query_rows']
        bias=s['bv'][qi]+s['bu'][g[qu]]
        present=days>=0
        bias[present]+=s['but'][batch['global_days'][days[present]]]
        h=np.tanh(.5*a)
        def decode(latent):
            v=latent@q.T
            return probabilities(bias+np.einsum('nrf,nf->nr',p[qi],v))
        raw=decode(h[qu]);stars=msg.STARS
        largest=max(largest,float(np.max(abs(raw@stars-original[rows]))))
        assert largest<.0005,(start,largest)
        free=probabilities(bias+2*np.log1p(coshm1[qi]+h[qu,None,:]*sinh[qi]).sum(2))
        informed=msg.tilt(free,native_base[rows])
        outputs=[free]
        for probs in (free,informed):
            e=np.einsum('nr,nrf->nf',probs,p[qi])
            for value,group,scale in ((e,qu,.5),(e,qu,1.),(e-priorp[qi],qu,1.),
                    (e,qu.astype('int64')*65536+reader.qr['dates'][rows],1.)):
                delta,n=msg.peer_sum(value,group)
                outputs.append(decode(np.tanh(.5*(a[qu]+scale*(delta@q)))))
        result[rows]=np.column_stack([raw]+[(v-raw)[:,:4] for v in outputs]+[np.log1p(n),np.log1p(reader.positive_counts[g[qu]]),
            raw@stars,free@stars,informed@stars])
        if start%(192*300)==0:print(phase,start,'native error',largest,flush=True)
    assert np.isfinite(result).all()
    return np.column_stack((base,result[ix]))


def model():
    return CatBoostRegressor(iterations=1050,depth=5,learning_rate=.045,l2_leaf_reg=45,thread_count=2,
        allow_writing_files=False,random_seed=1402,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    x=features('known',base)
    key,stats,held=anchor.crossfit(x,y,f,base,model)
    np.save(OUT/'oof.npy',held)
    rep=dict(selected=key,known=stats,parent=str(parent),qualifying_hashes={},
        checkpoint_receipt=str(EXP/'paired-verification.json'),qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':
        xx=features('qualifying',qb);delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32';ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        rep['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if key!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
