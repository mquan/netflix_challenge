#!/usr/bin/env python3
"""Exact own107/item154 plus32 native rank1000 item contexts, reconstructed in RAM."""
from pathlib import Path
import hashlib
import shutil
import numpy as np
import upstream_item154_features as parent
import upstream_rank1000_item32 as support

u,n=parent.u,parent.n
RUN=u.ROOT/'artifacts/upstream-item186-virtual-features'
REPORT=u.ROOT/'experiments/upstream-item186-virtual-features'
NEW=parent.NEW

def code_hashes():
    return {**parent.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(support.__file__),u.ROOT/'scripts/verify_upstream_rank1000_item32.py']}}

def inputs():
    path=support.REPORT/'verification.json';proof=u.read(path)
    if (proof['state']!='complete_independent_rank1000_item32_projection' or
        proof['receipt_sha256']!=u.sha(support.REPORT/'receipt.json') or not proof['independent_top_spectrum_checked'] or
        not proof['all_projected_items_replayed_bitwise'] or proof['qualifying_labels_read'] is not False):
        raise ValueError('Independent full item projection replay required')
    sources=dict(proof['bindings'])
    for p in [path,support.REPORT/'receipt.json']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Item projection source changed')
    query=u.FRESH/'data/probex/item_idxs.npy';sources[str(query.relative_to(u.ROOT))]=u.sha(query)
    ids=np.load(query);table=np.load(support.RUN/'item32.npy')
    if ids.shape!=(1128164,) or table.shape!=(17770,32) or ids.min()<0 or ids.max()>=17770:raise ValueError('Canonical item schema differs')
    return np.load(parent.RUN/'design.npy',mmap_mode='r'),np.load(parent.RUN/'features.npy',mmap_mode='r'),ids,table,sources

def block(arrays,lo,hi):
    z,f,ids,table,_=arrays
    return np.array(z[lo:hi]),np.column_stack([f[lo:hi],table[ids[lo:hi]]]).astype(np.float32)

def fingerprints(z,f):
    hashes=[hashlib.sha256(),hashlib.sha256()]
    for lo in range(0,len(z),8192):
        for h,a in zip(hashes,[z[lo:lo+8192],f[lo:lo+8192]]):h.update(np.ascontiguousarray(a).tobytes())
    return [h.hexdigest() for h in hashes]

def matrices():
    arrays=inputs();z=np.empty((1128164,183),dtype=np.float64);f=np.empty((1128164,429),dtype=np.float32)
    for lo in range(0,len(z),8192):z[lo:lo+8192],f[lo:lo+8192]=block(arrays,lo,min(lo+8192,len(z)))
    if fingerprints(z,f)!=u.read(REPORT/'cache-column-verification.json')['virtual_c_order_sha256']:raise ValueError('Complete virtual tensor fingerprints differ')
    return z,f

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing item186 evidence preserved')
    parent.configure();pm=n.verify_prepared();arrays=inputs();sources=arrays[-1]
    pp=parent.REPORT/'cache-column-verification.json';proof=u.read(pp)
    if proof['manifest_sha256']!=u.sha(parent.REPORT/'manifest.json') or not proof['all_retained_and_new_columns_checked']:raise ValueError('Complete verified item154 parent required')
    for name,digest in pm['prepared_sha256'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    for p in [parent.REPORT/'manifest.json',pp]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    if arrays[0].shape!=(1128164,183) or arrays[1].shape!=(1128164,397):raise ValueError('Parent schema differs')
    hashes=[hashlib.sha256(),hashlib.sha256()]
    for lo in range(0,1128164,8192):
        hi=min(lo+8192,1128164);z,f=block(arrays,lo,hi)
        if not np.array_equal(z,arrays[0][lo:hi]) or not np.array_equal(f[:,:397].view(np.uint32),arrays[1][lo:hi].view(np.uint32)):raise ValueError('Retained values differ')
        for j in range(32):
            if not np.array_equal(f[:,397+j].view(np.uint32),arrays[3][arrays[2][lo:hi],j].view(np.uint32)):raise ValueError('Every original item gather must match')
        for h,a in zip(hashes,[z,f]):h.update(a.tobytes())
    RUN.mkdir();REPORT.mkdir()
    for name in ['labels.npy','users.npy','groups.npy','baseline.npy','stats.npz']:u.link_or_copy(parent.RUN/name,RUN/name)
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(parent.REPORT/name,REPORT/name)
    u.write(REPORT/'manifest.json',dict(pm,code_sha256=code_hashes(),source_manifests_sha256=sources,
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},prediction_columns=107,design_columns=183,nonlinear_features=429,
        support_columns=186,own_item_pca_support_columns=96,new_native_predictions=NEW,virtual_tensor_inputs=True,
        new_support_features=pm['new_support_features']+[f'own-rank1000.item-pca{j:02}' for j in range(32)],
        scope='Current own107/item154 preserved exactly, append32 fixed unsupervised own rank1000 item components. No public numerical features or qualifying outcomes. Full virtual matrices are reconstructed in RAM.'))
    u.write(REPORT/'cache-column-verification.json',dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),
        screen_sha256=u.sha(REPORT/'ridge-screen.json'),source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),
        rows=1128164,prediction_columns=107,support_columns=186,outer_users_disjoint=True,
        all_retained_and_new_columns_checked=True,original_support_floats_preserved_bitwise=True,
        virtual_c_order_sha256=[h.hexdigest() for h in hashes],exact_virtual_shapes=[[1128164,183],[1128164,429]],
        qualifying_labels_read=False,verifier_sha256=u.sha(Path(__file__))))
if __name__=='__main__':prepare()
