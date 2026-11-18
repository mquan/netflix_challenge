#!/usr/bin/env python3
"""Assemble paired, freshly trained ProbeX/qual inputs without qualifying labels."""
import argparse
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import numpy as np
import upstream_repro as u
import upstream_snapshot_blend as b
import upstream_missing_screen as s

RUN=u.ROOT/'artifacts/upstream-production-bank'
REPORT=u.ROOT/'experiments/upstream-production-bank'
LOCATIONS=[(u.FRESH,u.BASE/'fresh-receipts','native'),
 (u.ROOT/'artifacts/upstream-qualifying-backfill/fresh',u.ROOT/'artifacts/upstream-qualifying-backfill/fresh-receipts','native-backfill'),
 (u.ROOT/'artifacts/full-upstream-mapped/source',u.ROOT/'artifacts/full-upstream-mapped/receipts','mapped')]


def recipe():
    frozen=u.read(b.REPORT/'manifest.json')
    rows=[dict(name=r['name'],clip=r['clip'],producer=r.get('parent')) for r in frozen['entries'] if r['duplicate_of'] is None]
    rows.append(dict(name='rbmx2-150__knnf',clip=True,producer='rbmx2-150__knnf'))
    if len(rows)!=62:raise ValueError('Original bank width changed')
    for path in ['experiments/upstream-remote-integration/manifest.json','experiments/upstream-remote-batch/manifest.json']:
        rows.extend(dict(name=name,clip=True,producer=None) for name in u.read(u.ROOT/path)['remote_columns'])
    rows.extend(dict(name=name,clip=True,producer=name) for name in ['mf-61','mf-61__knn3'])
    if len(rows)!=85 or len({r['name'] for r in rows})!=85:raise ValueError('Production bank differs from fixed recipe')
    jobs=u.read(u.BASE/'registry.json')['jobs']
    for row in rows:
        row['producer']=row['producer'] or s.producer_for(row['name'],jobs)[0]
        if row['producer'] is None:raise ValueError('Unresolved production owner '+row['name'])
    return rows,jobs


def choose_pair(row,jobs,verify=False):
    for tree,receipts,kind in LOCATIONS:
        rp=receipts/(row['producer']+'.json')
        if not rp.exists():continue
        record=u.read(rp)
        if record['source_commit']!=u.PIN or record['job']!=row['producer'] or record['command']!=jobs[row['producer']]['command']:
            raise ValueError('Invalid production owner receipt '+str(rp))
        paths={split:'preds_new/'+row['name']+'.'+split+'.npy' for split in ['probex','qual']}
        if not all(path in record['outputs'] and (tree/path).is_file() for path in paths.values()):continue
        if kind=='mapped':
            proof=u.ROOT/'experiments/full-upstream-reproduction/mapped-parity/verification.json'
            if record['parity_sha256']!=u.sha(proof) or not u.read(proof)['pass_all'] or record['storage']!=u.read(proof)['preparation']:
                raise ValueError('Mapped model lacks bound parity proof')
        if verify:
            for path in paths.values():
                if u.sha(tree/path)!=record['outputs'][path]:raise ValueError('Changed production prediction '+path)
        return dict(name=row['name'],clip=row['clip'],producer=row['producer'],kind=kind,
            receipt=str(rp.relative_to(u.ROOT)),receipt_sha256=u.sha(rp),
            files={split:str((tree/path).relative_to(u.ROOT)) for split,path in paths.items()},
            sha256={split:record['outputs'][path] for split,path in paths.items()})
    return None


def inventory(verify=False):
    rows,jobs=recipe();chosen=[choose_pair(row,jobs,verify) for row in rows]
    result=dict(columns=chosen,total=len(rows),complete=sum(v is not None for v in chosen),
                missing=[r['name'] for r,v in zip(rows,chosen) if v is None],qualifying_labels_read=False)
    u.write(REPORT/'inventory.json',result);return result


def blocks(raw,clip,context,voting):
    """Match the frozen 62-column centering, 14 interactions, then 23 additions."""
    raw=np.asarray(raw,dtype=np.float32).copy();raw[:,clip]=np.clip(raw[:,clip],0,6)
    original=raw[:,:62];first_clip=clip[:62]
    # The frozen full-bank fancy-indexed slice is column-major: NumPy sums its
    # slow axis in column order. A one-row chunk would otherwise switch to a
    # different pairwise reduction. Keep the original float32 addition order.
    mean=np.zeros((len(original),1),dtype=np.float32)
    for column in np.flatnonzero(first_clip):mean[:,0]+=original[:,column]
    mean/=np.float32(first_clip.sum())
    nonlin=original.copy();nonlin[:,first_clip]-=mean
    design=b.design(original,first_clip,context,tuple(range(14)))
    design=np.column_stack([design[:,:-1],raw[:,62:],design[:,-1:]])
    features=np.column_stack([mean-3.6,nonlin,context,raw[:,62:]-3.6,voting]).astype(np.float32)
    if design.shape[1]!=161 or features.shape[1]!=221:raise ValueError('Production feature dimensions changed')
    return design,features


def build():
    if (REPORT/'manifest.json').exists():
        frozen=u.read(REPORT/'manifest.json')
        if frozen['code_sha256']!=u.sha(Path(__file__)):raise ValueError('Frozen production assembly code changed')
        for name,digest in frozen['files'].items():
            if u.sha(RUN/name)!=digest:raise ValueError('Production bank changed')
        return frozen
    if RUN.exists():raise ValueError('Partial production assembly preserved')
    u.validate_fresh();inv=inventory(verify=True)
    if inv['missing']:raise ValueError('Production bank still incomplete')
    from upstream_voting_blend import feature_inventory
    jobs=u.read(u.BASE/'registry.json')['jobs']
    from upstream_full_reproduce import receipt_valid
    if not receipt_valid('vfeat1',jobs):raise ValueError('Voting features are not verified fresh outputs')
    votes=feature_inventory(u.read(u.BASE/'fresh-receipts/vfeat1.json'))
    sources={str((u.BASE/'fresh-receipts/vfeat1.json').relative_to(u.ROOT)):u.sha(u.BASE/'fresh-receipts/vfeat1.json')}
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    clip=np.array([r['clip'] for r in inv['columns']],dtype=bool)
    for split,train,rows in [('probex','trainx',1128164),('qual','fulltrain',2817131)]:
        metadata=[u.FRESH/'data'/stage/(key+'.npy') for stage in [train,split] for key in ['user_idxs','item_idxs','dates']]
        for p in metadata:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
        context=b.context_features(*[np.load(p,mmap_mode='r') for p in metadata])
        arrays=[np.load(u.ROOT/r['files'][split],mmap_mode='r') for r in inv['columns']]
        voting=[np.load(u.FRESH/(path if split=='probex' else path.replace('.probex.npy','.qual.npy')),mmap_mode='r') for path in votes]
        if any(a.shape!=(rows,) or a.dtype!=np.dtype('float32') for a in arrays+voting):raise ValueError('Bad production shape/type')
        z=np.lib.format.open_memmap(RUN/(split+'-design.npy'),mode='w+',dtype=np.float64,shape=(rows,161))
        f=np.lib.format.open_memmap(RUN/(split+'-features.npy'),mode='w+',dtype=np.float32,shape=(rows,221))
        for lo in range(0,rows,8192):
            hi=min(lo+8192,rows);raw=np.column_stack([a[lo:hi] for a in arrays]);v=np.column_stack([a[lo:hi] for a in voting])
            if not np.isfinite(raw).all() or not np.isfinite(v).all():raise ValueError('Nonfinite production input')
            z[lo:hi],f[lo:hi]=blocks(raw,clip,context[lo:hi],v)
        z.flush();f.flush()
    for key in ['ratings','user_idxs']:
        source=u.FRESH/'data/probex'/(key+'.npy');sources[str(source.relative_to(u.ROOT))]=u.sha(source)
        shutil.copyfile(source,RUN/('labels.npy' if key=='ratings' else 'users.npy'))
    for record in inv['columns']:
        sources[record['receipt']]=record['receipt_sha256']
        sources.update({record['files'][key]:record['sha256'][key] for key in ['probex','qual']})
    frozen=dict(state='complete',source_pin=u.PIN,columns=inv['columns'],code_sha256=u.sha(Path(__file__)),
        source_sha256=sources,files={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        qualifying_labels_read=False,qualifying_rows=2817131,rows=1128164,
        feature_recipe='Same 85-column/221-feature schema as voting experiment, rebuilt with paired fresh local ProbeX and qual outputs. Retrained Hogwild predictors may differ from remote development vectors; all final weights must be fitted to this rebuilt ProbeX bank.')
    u.write(REPORT/'manifest.json',frozen);return frozen


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--build',action='store_true');ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    if args.watch:
        while inventory()['missing']:time.sleep(60)
        build()
    elif args.build:build()
    else:print({k:v for k,v in inventory().items() if k!='columns'})
