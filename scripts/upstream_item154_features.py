#!/usr/bin/env python3
"""Own107/all90 plus fixed64 own PCA item context, with all columns checked."""
from pathlib import Path
import shutil
import numpy as np
import upstream_latent90_features as parent
import upstream_item64_support as support

u,n=parent.u,parent.n
RUN=u.ROOT/'artifacts/upstream-item154-features'
REPORT=u.ROOT/'experiments/upstream-item154-features'
NEW=parent.NEW

def code_hashes():
    return {**parent.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(support.__file__),u.ROOT/'scripts/verify_upstream_item64_support.py']}}

def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,u.read(parent.REPORT/'manifest.json')['config'],code_hashes

def checked_support():
    path=support.REPORT/'verification.json';proof=u.read(path)
    if (proof['state']!='complete_independent_item_projection_and_all_query_rows' or
            not proof['all_projected_items_and_query_rows_replayed_bitwise'] or
            proof['receipt_sha256']!=u.sha(support.REPORT/'receipt.json') or proof['qualifying_labels_read'] is not False):
        raise ValueError('Independent complete own item support proof required')
    sources=dict(proof['bindings'])
    for p in [path,support.REPORT/'receipt.json']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Item context lineage differs')
    return np.load(support.RUN/'probex.npy',mmap_mode='r'),sources

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing item154 evidence preserved')
    parent.configure();pm=n.verify_prepared();extra,sources=checked_support()
    old=np.load(parent.RUN/'features.npy',mmap_mode='r')
    if old.shape!=(1128164,333) or extra.shape!=(1128164,64):raise ValueError('Feature schema differs')
    for name,digest in pm['prepared_sha256'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    sources[str((parent.REPORT/'manifest.json').relative_to(u.ROOT))]=u.sha(parent.REPORT/'manifest.json')
    RUN.mkdir();REPORT.mkdir()
    for name,digest in pm['prepared_sha256'].items():
        if name=='features.npy':continue
        u.link_or_copy(parent.RUN/name,RUN/name)
        if u.sha(RUN/name)!=digest:raise ValueError('Retained ridge/identity cache changed')
    out=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(1128164,397))
    for lo in range(0,len(out),8192):out[lo:lo+8192,:333]=old[lo:lo+8192];out[lo:lo+8192,333:]=extra[lo:lo+8192]
    out.flush();del out
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(parent.REPORT/name,REPORT/name)
    u.write(REPORT/'manifest.json',dict(pm,code_sha256=code_hashes(),source_manifests_sha256=sources,
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},nonlinear_features=397,support_columns=154,
        new_support_features=pm['new_support_features']+[f'{name}.item-pca{j:02}' for name in support.MODELS for j in range(32)],
        own_item_pca_support_columns=64,scope='Retain own107/all90, unchanged ridge/identities/folds; append fixed64 unsupervised item context from own fresh native factors. No public numerical features or qualifying outcomes. Distinct additional own schema.'))
    return verify_columns()

def verify_columns():
    configure();m=n.verify_prepared();extra,_=checked_support()
    out=np.load(RUN/'features.npy',mmap_mode='r');old=np.load(parent.RUN/'features.npy',mmap_mode='r')
    for lo in range(0,len(out),16384):
        if not np.array_equal(out[lo:lo+16384,:333].view(np.uint32),old[lo:lo+16384].view(np.uint32)):raise ValueError('Retained context differs')
        if not np.array_equal(out[lo:lo+16384,333:].view(np.uint32),extra[lo:lo+16384].view(np.uint32)):raise ValueError('Added item context differs')
    for name in ['design.npy','labels.npy','users.npy','groups.npy','baseline.npy','stats.npz']:
        if u.sha(RUN/name)!=u.sha(parent.RUN/name):raise ValueError('Retained ridge/identities/folds differ')
    u.write(REPORT/'cache-column-verification.json',dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),
        screen_sha256=u.sha(REPORT/'ridge-screen.json'),source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),
        rows=len(out),prediction_columns=107,support_columns=154,outer_users_disjoint=True,
        all_retained_and_new_columns_checked=True,original_support_floats_preserved_bitwise=True,qualifying_labels_read=False,
        verifier_sha256=u.sha(Path(__file__)),scope='All397 feature columns checked bitwise. Ridge and row identities unchanged. Separate PCA and full source item gathers verified. No new accuracy claimed.'))

if __name__=='__main__':prepare()
