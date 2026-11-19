#!/usr/bin/env python3
"""Exact own109/item154 tensors assembled in RAM from immutable source columns."""
from pathlib import Path
import hashlib
import shutil
import numpy as np
import upstream_item154_features as parent
import upstream_itemae_pair2_screen as screen

u,n=parent.u,parent.n
RUN=u.ROOT/'artifacts/upstream-itemae-virtual-features'
REPORT=u.ROOT/'experiments/upstream-itemae-virtual-features'
NEW=screen.NEW

def code_hashes():
    return {**parent.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(screen.__file__),u.ROOT/'scripts/verify_upstream_itemae_pair2_screen.py']}}

def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,u.read(parent.REPORT/'manifest.json')['config'],code_hashes

def inputs():
    rows,_=screen.verified_inputs()
    return (np.load(parent.RUN/'design.npy',mmap_mode='r'),np.load(parent.RUN/'features.npy',mmap_mode='r'),
        [np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in rows])

def block(arrays,lo,hi):
    old,features,raw=arrays;extra=np.column_stack([np.clip(a[lo:hi],0,6) for a in raw])
    return (np.column_stack([old[lo:hi,:182],extra,old[lo:hi,-1]]),
            np.column_stack([features[lo:hi,:243],extra-np.float32(3.6),features[lo:hi,243:]]).astype(np.float32))

def fingerprints(z,f):
    digests=[hashlib.sha256(),hashlib.sha256()]
    for lo in range(0,len(z),8192):
        for h,a in zip(digests,[z[lo:lo+8192],f[lo:lo+8192]]):h.update(np.ascontiguousarray(a).tobytes())
    return [h.hexdigest() for h in digests]

def matrices():
    # No source/config rebinding here: the active model's n.CONFIG stays fixed.
    arrays=inputs();z=np.empty((1128164,185),dtype=np.float64);f=np.empty((1128164,399),dtype=np.float32)
    for lo in range(0,len(z),8192):z[lo:lo+8192],f[lo:lo+8192]=block(arrays,lo,min(lo+8192,len(z)))
    proof=u.read(REPORT/'cache-column-verification.json')
    if fingerprints(z,f)!=proof['virtual_c_order_sha256']:raise ValueError('Exact virtual source tensors differ')
    return z,f

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing virtual tensor evidence preserved')
    parent.configure();pm=n.verify_prepared();rows,sources=screen.verified_inputs()
    rp=screen.REPORT/'ridge-verification.json';r=u.read(rp);evaluation=u.read(screen.REPORT/'ridge-screen.json')
    if r['state']!='complete' or r['screen_sha256']!=u.sha(screen.REPORT/'ridge-screen.json') or not r['users_disjoint']:
        raise ValueError('Independent full five-fold native batch ridge replay required')
    sources.update(evaluation['source_sha256'])
    for name,digest in pm['prepared_sha256'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    for p in [parent.REPORT/'manifest.json',rp,screen.REPORT/'ridge-screen.json']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    RUN.mkdir();REPORT.mkdir();arrays=inputs();hashes=[hashlib.sha256(),hashlib.sha256()]
    if arrays[0].shape!=(1128164,183) or arrays[1].shape!=(1128164,397):raise ValueError('Own107/item154 parent shape differs')
    for lo in range(0,1128164,8192):
        hi=min(lo+8192,1128164);z,f=block(arrays,lo,hi);old,of,raw=arrays
        if z.shape!=(hi-lo,185) or f.shape!=(hi-lo,399):raise ValueError('Virtual tensor dimensions differ')
        if not np.array_equal(z[:,:182],old[lo:hi,:182]) or not np.array_equal(z[:,-1],old[lo:hi,-1]):raise ValueError('Retained ridge differs')
        if not np.array_equal(f[:,:243].view(np.uint32),of[lo:hi,:243].view(np.uint32)) or not np.array_equal(f[:,245:].view(np.uint32),of[lo:hi,243:].view(np.uint32)):
            raise ValueError('Retained raw/context values differ')
        for j,a in enumerate(raw):
            expected=np.clip(a[lo:hi],0,6)
            if not np.array_equal(z[:,182+j],expected) or not np.array_equal(f[:,243+j],expected-np.float32(3.6)):raise ValueError('Original native transform differs')
        hashes[0].update(z.tobytes());hashes[1].update(f.tobytes())
    for name in ['labels','users','groups']:u.link_or_copy(parent.RUN/(name+'.npy'),RUN/(name+'.npy'))
    shutil.copyfile(screen.RUN/'main-ridge.npy',RUN/'baseline.npy');shutil.copyfile(screen.RUN/'stats.npz',RUN/'stats.npz')
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(screen.REPORT/name,REPORT/name)
    manifest=dict(pm,code_sha256=code_hashes(),source_manifests_sha256=sources,
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},prediction_columns=109,design_columns=185,nonlinear_features=399,
        new_native_predictions=NEW,virtual_tensor_inputs=True,
        baseline_rmse=next(r['rmse'] for r in evaluation['records'] if r['arm']=='main'),
        scope='Exact own107/item154 retained, insert both independently completed native itemae pairs. Arrays are constructed in RAM with every column and complete C-order fingerprints checked. No large redundant input files, public numerical features, qualifying outcomes or new accuracy claimed.')
    u.write(REPORT/'manifest.json',manifest)
    u.write(REPORT/'cache-column-verification.json',dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),
        screen_sha256=u.sha(REPORT/'ridge-screen.json'),source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),
        all_retained_and_new_columns_checked=True,original_support_floats_preserved_bitwise=True,outer_users_disjoint=True,
        rows=1128164,prediction_columns=109,support_columns=154,virtual_c_order_sha256=[h.hexdigest() for h in hashes],
        exact_virtual_shapes=[[1128164,185],[1128164,399]],qualifying_labels_read=False,verifier_sha256=u.sha(Path(__file__))))

if __name__=='__main__':prepare()
