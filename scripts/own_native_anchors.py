"""Transfer independently held-out native movie errors visible in each actual training split."""
import json
import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_native_holdout as native
import own_anchor_covariance as covariance
import own_full_context as full
import own_history_context as hist

OUT=ne.ROOT/'own-native-anchors-20261126'


def features(train,phase,base,u,i,d,y,ix,raw):
    expect=len(u)-len(ix) if phase=='probex' else len(u)
    present,rows=anchor.training_indices(train,u,i,d,expected=expect)
    if phase=='probex':assert not present[ix].any()
    stars=ne.history(train,'ratings')[rows]
    np.testing.assert_array_equal(stars,y[present])
    tu,ti,td,pr=u[present],i[present],d[present],raw[present]
    delta=np.column_stack((stars-pr[:,0],stars-pr[:,1],stars-full.cf96(train)[rows],
                           stars-ne.verified_rank(train)[rows],stars-3.6)).astype('f')
    qu,qi,qd=[ne.history(phase,k) for k in ('user_idxs','item_idxs','dates')]
    g=covariance.movie_geometry()
    order=np.argsort(tu,kind='stable')
    off=np.r_[0,np.bincount(tu,minlength=480189).cumsum()]
    cnt=np.diff(off)
    result=np.zeros((len(base),7,8),'f')
    for pos in range(int(cnt.max())):
        qk=np.flatnonzero(cnt[qu]>pos)
        for a in range(0,len(qk),32768):
            j=qk[a:a+32768]
            other=order[off[qu[j]]+pos]
            assert np.all(ti[other]!=qi[j])
            gap=np.abs(td[other].astype('i')-qd[j])
            sim=np.einsum('ij,ij->i',g[ti[other]],g[qi[j]])
            near=1/(1+gap/15.)
            same=(gap==0)
            w=np.column_stack((np.ones(len(j)),near,same,np.exp(2*(sim-1)),np.exp(2*(sim-1))*near,
                               np.exp(-np.abs(base[j]-pr[other,0])),(np.maximum(sim,0)**2)*same))
            val=np.column_stack((np.ones(len(j)),delta[other],delta[other,0]**2,pr[other,0]-base[j]))
            result[j]+=w[:,:,None]*val[:,None,:]
    z=result[:,:,0].copy()
    result[:,:,1:]/=(z[:,:,None]+.5)
    result[:,:,0]=np.log1p(z)
    h=np.load(hist.OUT/(phase+'.npy'),mmap_mode='r')
    x=np.column_stack((base,h[:,1:10],h[:,13:17],result.reshape(len(base),-1))).astype('f')
    assert np.isfinite(x).all()
    return x


def model():
    return CatBoostRegressor(iterations=950,depth=5,learning_rate=.035,l2_leaf_reg=45,
                             thread_count=2,allow_writing_files=False,random_seed=20261126,verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    u,i,d,stars,ix=native.identities()
    raw=np.load(native.OUT/'native-oof.npy')
    # A native anchor's refit excludes its user's entire validation group.
    assert raw.shape==(len(stars),2) and np.isfinite(raw).all()
    x=features('trainx','probex',base,u,i,d,stars,ix,raw)
    selected,stats,held=anchor.crossfit(x,y,f,base,model)
    np.save(OUT/'oof.npy',held)
    report=dict(selected=selected,known=stats,parent=str(parent),anchor_predictions_sha256=ne.sha(native.OUT/'native-oof.npy'),
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        xx=features('fulltrain','qual',qb,u,i,d,stars,ix,raw)
        delta=model().fit(x,y-base).predict(xx,thread_count=2)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(qb+float(selected)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
