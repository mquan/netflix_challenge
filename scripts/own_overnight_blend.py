"""Select cumulative own-model packages on ProbeX before freezing one qualifying vector."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_history_context as hist
import own_full_context as full
import own_retained_history as retained

OUT=ne.ROOT/'own-overnight-blend-20261126'


def direction(key,p,q,known):
    phase='probex' if known else 'qual'
    base=p if known else q
    prefix='known' if known else 'qual'
    if key=='retained':
        path=retained.OUT/(prefix+'.npy')
        z=np.load(path).astype(float)-base[:,None]
    elif key=='bayes':
        if known:
            ix=retained.map_rows()
            rec=json.loads(Path('experiments/QUAL-B2-fixed128-recipe-20261116.json').read_text())
            path=Path(rec['column'])
            assert ne.sha(path)==rec['verified_files'][str(path.relative_to(Path.cwd()))]
            raw=np.fromfile(path,dtype='<f4')[ix]
        else:
            path=retained.OUT/'bayes/model.f32'
            raw=np.fromfile(path,dtype='<f4')
        z=raw.astype(float)[:,None]-base[:,None]
    elif key.startswith('attention'):
        root=ne.ROOT/('own-masked-attention-20261126' if key=='attention' else key.removeprefix('attention:'))
        path=root/(phase+'.npy')
        z=np.load(path).astype(float)[:,None]-base[:,None]
    elif key.startswith('signal:'):
        _,family,step=key.split(':')
        path=ne.ROOT/family/(phase+'.npy')
        v=np.load(path,mmap_mode='r')
        z=np.asarray(v[:,::int(step)],dtype=float)
    elif key.startswith('raw:'):
        root=ne.ROOT/key.removeprefix('raw:')
        path=root/(phase+'.npy')
        z=np.load(path).astype(float).reshape(len(base),-1)-base[:,None]
    else:
        special={'anchor':anchor.OUT,'pairs':ne.ROOT/'own-session-pairs-20261126','latent':ne.ROOT/'own-session-latent-20261126',
                 'shallow':full.OUT/'shallow','deep':full.OUT/'deep'}
        root=special[key] if key in special else ne.ROOT/key
        path=root/('oof.npy' if known else 'qual-correction.npy')
        z=np.load(path).astype(float).reshape(len(base),-1)
    assert len(z)==len(base) and np.isfinite(z).all()
    return z,{'path':str(path),'sha256':ne.sha(path),'width':z.shape[1]}


def solve(x,e):
    g=x.T@x
    g.flat[::len(g)+1]+=25.
    l=np.linalg.cholesky(g)
    ans=lsq_linear(l.T,np.linalg.solve(l,x.T@e),bounds=(0.,2.),tol=1e-12)
    assert ans.success
    return ans.x


def metrics(v,y,f):
    return dict(pooled=ne.rmse(v,y),folds=[ne.rmse(v[f==j],y[f==j]) for j in range(5)])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('keys',nargs='+')
    ap.add_argument('--parent',default='retained')
    args=ap.parse_args()
    out=OUT/args.name
    out.mkdir(parents=True,exist_ok=False)
    y,f,p,q=anchor.history_control()
    if args.parent=='retained':
        control=ne.clipped(p+np.load(retained.OUT/'oof.npy'))
        control_qp=retained.OUT/'selected.f32'
        expected=json.loads((retained.OUT/'freeze.json').read_text())['qualifying_hashes']['selected']
    else:
        prior=OUT/args.parent
        pr=json.loads((prior/'freeze.json').read_text())
        control=np.load(prior/'known-selected.npy')
        control_qp=prior/'selected.f32'
        expected=pr['qualifying_hashes']['selected']
    assert ne.sha(control_qp)==expected
    chunks=[direction(key,p,q,True) for key in args.keys]
    x=np.column_stack([z for z,_ in chunks])
    sources={k:r for k,(_,r) in zip(args.keys,chunks)}
    del chunks
    held=p.copy()
    ws=[]
    for j in range(5):
        tr=f!=j
        w=solve(x[tr],(y-p)[tr])
        held[~tr]=ne.clipped(p[~tr]+x[~tr]@w)
        ws.append(w.tolist())
    old,new=metrics(control,y,f),metrics(held,y,f)
    wins=sum(a<b for a,b in zip(new['folds'],old['folds']))
    ok=old['pooled']>new['pooled'] and wins>=4
    w=solve(x,y-p)
    report=dict(parent=args.parent,columns=args.keys,known_sources=sources,control=old,candidate=new,
                fold_weights=ws,weights=w.tolist(),improved_folds=wins,selected='candidate' if ok else 'parent',
                qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False,
                control_qual_sha256=expected)
    (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        value=q.copy()
        j=0
        report['qual_sources']={}
        for key in args.keys:
            v,receipt=direction(key,p,q,False)
            report['qual_sources'][key]=receipt
            value+=v@w[j:j+v.shape[1]]
            j+=v.shape[1]
        np.save(out/'known-selected.npy',held)
        ne.clipped(value).astype('<f4').tofile(out/'selected.f32')
    else:
        import os
        np.save(out/'known-selected.npy',control)
        os.link(control_qp,out/'selected.f32')
    report['qualifying_hashes']['selected']=ne.sha(out/'selected.f32')
    (out/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    print(args.name,report['selected'],old['pooled'],new['pooled'],wins,flush=True)
    if ok:ne.audit(out,ne.LABELS)


if __name__=='__main__': main()
