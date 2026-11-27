"""Fit two bagged leaf-wise residual models to the frozen own-model expert predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained
import own_trajectory_bank as trajectory


def sources():
    paths={}
    for family,stride in [('kernel-family',2),('cf-context',2),('cf-user-fit',1),
                          ('user-gp',2),('session-pairs',3),('anchor-context',4)]:
        root=ne.ROOT/('own-'+family+'-20261126')
        if all((root/(phase+'.npy')).is_file() for phase in ('probex','qual')):
            paths[str(root)]=stride
    return paths


def data(phase,p,q,base,columns,paths):
    known=phase=='probex'
    ctx=np.load(full.OUT/(phase+'.npy'),mmap_mode='r')
    sig=json.loads((full.OUT/(phase+'-sources.json')).read_text())
    assert ne.sha(full.OUT/(phase+'.npy'))==sig['feature_sha256']
    blocks=[ctx[:,::5]]
    for root,stride in paths.items():
        raw=np.load(Path(root)/(phase+'.npy'),mmap_mode='r')
        blocks.append(raw[:,::stride])
    support=json.loads((trajectory.OUT/'sources.json').read_text())['support']
    for info in list(support.values())[::4]:
        z=np.load(info['paths'][int(not known)],mmap_mode='r')
        blocks.append(z.reshape(-1,1))
    d=np.column_stack([blend.direction(k,p,q,known)[0].astype('f') for k in columns])
    x=np.empty((len(base),sum(a.shape[1] for a in blocks)+d.shape[1]+6),'f')
    for j in range(0,len(base),16384):
        sl=slice(j,j+16384)
        x[sl]=np.column_stack((base[sl],d[sl],np.std(d[sl],1),np.min(d[sl],1),np.max(d[sl],1),
            np.quantile(d[sl],.25,axis=1),np.quantile(d[sl],.75,axis=1),*[a[sl] for a in blocks]))
    return x


def worker(out,phase,x):
    path=out/'features.tmp.npy'
    np.save(path,x)
    del x
    try:
        subprocess.run([sys.executable,'-u','scripts/own_late_boost_worker.py',str(out),phase],check=True)
    finally:
        path.unlink(missing_ok=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='own-late-boost-20261126')
    args=ap.parse_args();out=ne.ROOT/args.name;out.mkdir(exist_ok=True)
    y,f,p,q=anchor.history_control()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    fr=json.loads((parent/'freeze.json').read_text())
    assert ne.sha(parent/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(parent/'known-selected.npy');qb=np.fromfile(parent/'selected.f32','<f4')
    cols=fr['columns'];paths=sources()
    np.save(out/'folds.npy',f);np.save(out/'residuals.npy',y-base)
    worker(out,'known',data('probex',p,q,base,cols,paths))
    held=np.load(out/'oof.npy')
    stats={str(w):blend.metrics(ne.clipped(base+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok=[k for k in stats if retained.qualifies(stats[k],stats['0'])]
    key=min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    rep=dict(selected=key,known=stats,parent=str(parent),columns=cols,features=paths,qualifying_hashes={},
             qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (out/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        worker(out,'qual',data('qual',p,q,qb,cols,paths))
        delta=np.load(out/'qual-correction.npy')
        path=out/'selected.f32';ne.clipped(qb+float(key)*delta).astype('<f4').tofile(path)
        rep['qualifying_hashes']['selected']=ne.sha(path)
    (out/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:ne.audit(out,ne.LABELS)


if __name__=='__main__':main()
