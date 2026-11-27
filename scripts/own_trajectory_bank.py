"""Cross-fit native CF, RBM, attention and factor trajectories with latent feature interactions."""
import json
import numpy as np
import torch
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_overnight_blend as blend
import own_retained_history as retained
import own_producer_bank as bank
import own_full_context as full

OUT=ne.ROOT/'own-trajectory-bank-20261126'
FAMILIES={
    'upstream-own391-paired-data':('b62ab10101b5e785f7d854dbddb379879283c101c87c4c3c3ac3846d6c13637c',
                                  ('original_rows','temporal_rows','new_rows')),
    'upstream-complete-trajectory-data':('53d159bf68a5f829e1e8e05dcd8a1a2e49236299f7a1dce6b4e916bfb5199e9b',('retained',)),
}


def sources():
    pairs={}
    seen=set()
    for n,(digest,keys) in FAMILIES.items():
        folder=bank.UP/'experiments'/n
        path=folder/'manifest.json'
        assert ne.sha(path)==digest
        rec=json.loads(path.read_text())
        proof=json.loads((folder/'independent-verification.json').read_text())
        assert proof['manifest_sha256']==digest and rec['qualifying_labels_read'] is False
        assert rec['matched_B_known_and_qualifying'] and not rec['nonlinear_component_OOF_predictions_used_as_features']
        for field in keys:
            for row in rec[field]:
                p=[row['paired_files'][phase] for phase in ('probex','qual')]
                if 'artifacts/upstream-repro/' in p[0]:continue
                assert all('public' not in name.lower() for name in p)
                ids=[rec['source_sha256'][name] for name in p]
                if tuple(ids) in seen:continue
                paths=[bank.UP/name for name in p]
                if not all(path.is_file() and ne.sha(path)==sig for path,sig in zip(paths,ids)):continue
                name=row.get('name',row.get('id'))
                # Explicit allowlist of the locally executed native model families in these receipts.
                if not any(a in name for a in ('cfnade','s3-attn','s3-dnn','s3-rbm','s3-tx','tsvdx','s3-mf','s3-ae','s3-cf','s3-tsvd')):continue
                pairs[n+'/'+name]=dict(receipt=str(path),paths=list(map(str,paths)),sha256=ids)
                seen.add(tuple(ids))
    assert len(pairs)>=125
    support={}
    for model in ('rbm50lat','tx20lat'):
        path=bank.UP/'experiments/upstream-latent90-local'/(model+'-completion-verification.json')
        rec=json.loads(path.read_text())
        assert rec['source_pin']==bank.PIN and rec['both_full_epoch_traces'] and rec['all_support_rows_independently_replayed_bitwise']
        assert rec['qualifying_targets_zero'] and not rec['public_prediction_values_read'] and not rec['qualifying_labels_read']
        bindings=rec['bindings']
        for k,v in bindings.items():
            if '/lat/'+model+'.' not in k or not k.endswith('.probex.npy'):continue
            q=k.removesuffix('probex.npy')+'qual.npy'
            paths=[bank.UP/k,bank.UP/q]
            if not all(pp.is_file() and ne.sha(pp)==bindings[str(pp.relative_to(bank.UP))] for pp in paths):continue
            support[k]=dict(paths=list(map(str,paths)),sha256=[v,bindings[q]],receipt=str(path))
    assert len(support)==90
    return pairs,support


def inputs(phase,support):
    j=int(phase=='qual')
    pc=np.load(OUT/(phase+'.npy'),mmap_mode='r')
    fresh=np.load(bank.OUT/(phase+'.npy'),mmap_mode='r')
    ctx=np.load(full.OUT/(phase+'.npy'),mmap_mode='r')
    width=pc.shape[1]+fresh.shape[1]+len(support)+len(range(0,ctx.shape[1],6))
    result=np.empty((len(pc),width),'f')
    lo=0
    for a in (pc,fresh,ctx[:,::6]):
        result[:,lo:lo+a.shape[1]]=a
        lo+=a.shape[1]
    for s in support.values():
        result[:,lo]=np.load(s['paths'][j],mmap_mode='r')
        lo+=1
    return torch.from_numpy(result)


def main():
    OUT.mkdir(exist_ok=True)
    y,f,p,q=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    rec=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==rec['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy')
    qb=np.fromfile(parent/'selected.f32','<f4')
    pairs,supp=sources()
    print('verified',len(pairs),'prediction pairs and',len(supp),'factor pairs',flush=True)
    (OUT/'sources.json').write_text(json.dumps(dict(predictions=pairs,support=supp),indent=2)+'\n')
    bank.project(pairs,p,q,OUT)
    torch.set_num_threads(2)
    x=inputs('probex',supp)
    e=torch.from_numpy((y-base).astype('f'))
    held=np.empty(len(y))
    for j in range(5):
        tr,va=[torch.from_numpy(np.flatnonzero(v)) for v in (f!=j,f==j)]
        held[va.numpy()]=bank.fit(x,x,tr,va,e)
        print('fold',j,ne.rmse(base[va]+held[va],y[va]),flush=True)
    np.save(OUT/'oof.npy',held)
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    eligible=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(eligible,key=lambda k:stats[k]['pooled']) if eligible else 'control'
    report=dict(selected=key,known=stats,parent=str(parent),sources_sha256=ne.sha(OUT/'sources.json'),
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:
        delta=bank.fit(x,inputs('qual',supp),torch.arange(len(y)),torch.arange(len(q)),e)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if eligible:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
