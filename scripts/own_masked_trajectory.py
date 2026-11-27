"""Calibrate early-to-late masked sequence checkpoints on held-out users."""
import json
import numpy as np
import own_neighbor_experiment as ne
import own_retained_history as ret
import own_overnight_blend as blend
import own_history_context as hist
import own_anchor_context as anchor

OUT=ne.ROOT/'own-masked-trajectory-20261126'
PARENT=blend.OUT/'stage-03'
EPOCHS=(3,4,5,6,8)


def features(query,base,ix):
    xs=[]
    verified={}
    for epoch in EPOCHS:
        if query=='probex':
            path=ne.ROOT/'retained-neural'/f'A2h-epoch{epoch}.f32'
            rec=json.loads(path.with_suffix('.f32.json').read_text())
            assert rec['epochs']==[epoch] and rec['config']['objective']=='masked'
            assert rec['data_sha256']=='08b417f7601339c95727cd02af4f839b6043c326a5fb0350245a1f6f2054c673'
            assert ne.sha(path)==rec['column_sha256']
            x=np.fromfile(path,'<f4')[ix]
        else:
            folder=ret.QUAL/'A2h-fixed12-pad16/inference-cache'/f'e{epoch}'
            path=folder/'model.f32'
            rec=json.loads((folder/'report.json').read_text())
            assert rec['state']=='complete' and rec['snapshot_epochs']==[epoch] and not rec['qualifying_labels_read']
            assert rec['config']['objective']=='masked' and rec['dataset_manifest_sha256']==ne.sha(ret.QUAL/'dataset/manifest.json')
            assert rec['predictions_sha256']==ne.sha(path)
            x=np.fromfile(path,'<f4')
        verified[str(path)]=ne.sha(path)
        xs.append(x)
    n='known' if query=='probex' else 'qual'
    avg=np.load(ret.OUT/(n+'.npy'),mmap_mode='r')[:,0]
    h=np.load(hist.OUT/(query+'.npy'),mmap_mode='r')
    diff=np.column_stack([x-avg for x in xs]+[avg-base]).astype('f')
    return diff,np.column_stack([base,h[:,1],h[:,2],h[:,4],h[:,13],np.std(diff[:,:-1],axis=1)]),verified


def design(d,h,center,scale):
    return np.column_stack((d,(d[:,:,None]*np.clip((h-center)/scale,-2,2)[:,None,:]).reshape(len(h),-1)))


def fit(d,h,e,mask):
    center=np.median(h[mask][::10],axis=0)
    scale=np.maximum(np.std(h[mask][::10],axis=0),.001)
    x=design(d[mask],h[mask],center,scale).astype(float)
    G=x.T@x
    G.flat[::len(G)+1]+=np.r_[[75]*d.shape[1],[200]*(len(G)-d.shape[1])]
    return center,scale,np.linalg.solve(G,x.T@e[mask])


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control()
    p=np.load(PARENT/'known-selected.npy')
    fr=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    ix=ret.map_rows()
    d,h,sources=features('probex',p,ix)
    delta=np.empty(len(y))
    for j in range(5):
        val=f==j
        c,s,w=fit(d,h,y-p,~val)
        delta[val]=design(d[val],h[val],c,s)@w
    stats=blend.metrics(ne.clipped(p+delta),y,f)
    base=blend.metrics(p,y,f)
    passes=ret.qualifies(stats,base)
    np.save(OUT/'oof.npy',delta)
    report=dict(selected='candidate' if passes else 'control',known=stats,control=base,known_sources=sources,
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if passes:
        q=np.fromfile(PARENT/'selected.f32','<f4')
        qd,qh,report['qual_sources']=features('qual',q,ix)
        c,s,w=fit(d,h,y-p,np.ones(len(y),bool))
        v=np.empty(len(q))
        for j in range(0,len(q),65536):
            sl=slice(j,j+65536)
            v[sl]=design(qd[sl],qh[sl],c,s)@w
        np.save(OUT/'qual-correction.npy',v)
        path=OUT/'selected.f32'
        ne.clipped(q+v).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if passes:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
