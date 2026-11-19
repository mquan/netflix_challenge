#!/usr/bin/env python3
"""Independently compare all91 production columns with their declared inputs."""
import datetime as dt
from pathlib import Path
import numpy as np
import upstream_extension_bank as bank

u=bank.u


def compare_rows(left,right):
    if left.shape!=right.shape:raise ValueError('Array shape differs')
    changed=0;maximum=0.;columns=set()
    for lo in range(0,len(left),8192):
        a,b=left[lo:lo+8192],right[lo:lo+8192];bad=a!=b
        changed+=int(np.count_nonzero(bad));maximum=max(maximum,float(np.max(np.abs(a-b))))
        if bad.ndim==2:columns.update(np.flatnonzero(bad.any(axis=0)).tolist())
    return dict(all_values_equal=changed==0,unequal_values=changed,max_abs_difference=maximum,
                changed_columns=sorted(columns),production_dtype=str(left.dtype),development_dtype=str(right.dtype))


def check_block(raw,clip,z,f,voting):
    if raw.shape[1]!=91 or z.shape!=(len(raw),167) or f.shape!=(len(raw),227):
        raise ValueError('Declared all91 block shape differs')
    a=np.asarray(raw,dtype=np.float32).copy();a[:,clip]=np.clip(a[:,clip],0,6)
    mean=np.add.accumulate(a[:,:62][:,clip[:62]],axis=1,dtype=np.float32)[:,-1]/np.float32(clip[:62].sum())
    centered=a[:,:62].copy();centered[:,clip[:62]]-=mean[:,None]
    if (not np.isfinite(a).all() or not np.isfinite(z).all() or not np.isfinite(f).all() or
            not np.array_equal(z[:,:62],a[:,:62]) or not np.all(z[:,166]==1) or
            not np.array_equal(z[:,137:166],a[:,62:91]) or
            not np.array_equal(f[:,0],mean-np.float32(3.6)) or
            not np.array_equal(f[:,1:63],centered) or
            not np.array_equal(f[:,63:68],z[:,62:67].astype(np.float32)) or
            not np.array_equal(f[:,68:91],a[:,62:85]-np.float32(3.6)) or
            not np.array_equal(f[:,221:227],a[:,85:91]-np.float32(3.6)) or
            not np.array_equal(f[:,91:221],voting)):
        raise ValueError('Declared paired native features/centering/intercept differ')
    for j in range(5):
        expected=(a[:,:14].astype(np.float64)-3.6)*z[:,62+j:63+j]
        if not np.array_equal(z[:,67+14*j:81+14*j],expected):
            raise ValueError('Declared context interactions differ')


def verify():
    manifest=bank.build();development=bank.development
    dm=u.read(development.REPORT/'manifest.json');records=[];bindings={}
    for prod,known in [('probex-design.npy','design.npy'),('probex-features.npy','features.npy'),
                       ('labels.npy','labels.npy'),('users.npy','users.npy')]:
        path=development.RUN/known
        if u.sha(path)!=dm['prepared_sha256'][known]:raise ValueError('Original development array changed')
        compared=compare_rows(np.load(bank.RUN/prod,mmap_mode='r'),np.load(path,mmap_mode='r'))
        if known in ['labels.npy','users.npy'] and not compared['all_values_equal']:
            raise ValueError('Known row identity differs')
        bindings[str(path.relative_to(u.ROOT))]=dm['prepared_sha256'][known]
        records.append(dict(production=prod,development=known,**compared))
    base=u.read(bank.parent.REPORT/'manifest.json');clip=np.array([r['clip'] for r in base['columns']]+[True]*6)
    from upstream_voting_blend import feature_inventory
    vote_receipt=u.BASE/'fresh-receipts/vfeat1.json';vr=u.read(vote_receipt);votes=feature_inventory(vr)
    bindings[str(vote_receipt.relative_to(u.ROOT))]=u.sha(vote_receipt);blocks={}
    for split,rows in [('probex',1128164),('qual',2817131)]:
        z=np.load(bank.RUN/(split+'-design.npy'),mmap_mode='r');f=np.load(bank.RUN/(split+'-features.npy'),mmap_mode='r')
        native=[];vote_arrays=[]
        for entry in base['columns']:
            path=u.ROOT/entry['files'][split];native.append(np.load(path,mmap_mode='r'))
            bindings[str(path.relative_to(u.ROOT))]=entry['sha256'][split]
        for name in development.NEW:
            path=u.FRESH/'preds_new'/(name+'.'+split+'.npy');native.append(np.load(path,mmap_mode='r'))
            bindings[str(path.relative_to(u.ROOT))]=manifest['source_sha256'][str(path.relative_to(u.ROOT))]
        for filename in votes:
            relative=filename if split=='probex' else filename.replace('.probex.npy','.qual.npy')
            path=u.FRESH/relative;vote_arrays.append(np.load(path,mmap_mode='r'))
            bindings[str(path.relative_to(u.ROOT))]=vr['outputs'][relative]
        blocks[split]=0
        for lo in range(0,rows,8192):
            hi=min(lo+8192,rows)
            raw=np.column_stack([a[lo:hi] for a in native]);voting=np.column_stack([a[lo:hi] for a in vote_arrays])
            check_block(raw,clip,z[lo:hi],f[lo:hi],voting);blocks[split]+=1
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):
        raise ValueError('Native qualifying targets must remain zero')
    for name,digest in manifest['files'].items():bindings[str((bank.RUN/name).relative_to(u.ROOT))]=digest
    for path in [Path(__file__),bank.REPORT/'manifest.json',development.REPORT/'manifest.json',bank.parent.REPORT/'manifest.json']:
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name in ['qual-design.npy','qual-features.npy']:
        path=bank.parent.RUN/name;bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    for relative,digest in bindings.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Bank evidence changed during verification')
    report=dict(state='complete_independent_column_verification',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        known_rows=manifest['rows'],qualifying_rows=2817131,development_comparison=records,
        verified_blocks=blocks,paired_native_design_columns_equal=True,intercept_preserved=True,
        centered_features_equal=True,context_interactions_equal=True,all130_native_voting_columns_equal=True,
        native_order=development.NEW,bindings=bindings,qualifying_labels_read=False,qualifying_targets_zero=True,
        development_predictor_values_identical=all(r['all_values_equal'] for r in records),
        scope='Production and development share row identities and feature recipes, but some independently retrained own source vectors differ as already disclosed by the parent bank. Differences are quantified, not treated as exact development reproduction. Every production row matches the actual declared paired native predictions, fixed centering, interactions, intercept and all130 native voting columns. Metadata context is preserved from the hashed parent bank, not rederived from training dates in this check. No recovered outcomes or fitting.')
    u.write(bank.REPORT/'column-verification.json',report)
    print(dict(state=report['state'],known_rows=report['known_rows'],qualifying_rows=report['qualifying_rows']))


if __name__=='__main__':verify()
