#!/usr/bin/env python3
"""Append all40 original Tx user/item components to the verified RBM50 cache."""
from pathlib import Path
import shutil
import time
import numpy as np
import upstream_rbm50_features as parent
import upstream_latent90_early_support as support

u,n=parent.u,parent.n
RUN=u.ROOT/'artifacts/upstream-latent90-features'
REPORT=u.ROOT/'experiments/upstream-latent90-features'
NEW=parent.NEW
TX_NAMES=[f'tx20lat.{kind}{j:02}' for kind in ['u','i'] for j in range(20)]

def code_hashes():
    return {**parent.code_hashes(),str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__))}

def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,u.read(parent.REPORT/'manifest.json')['config'],code_hashes

def tx_inputs():
    path=support.REPORT/'tx20lat-receipt.json';r=u.read(path)
    if (r['state']!='complete_original_first_phase_paired_supports' or r['model']!='tx20lat' or
        r['first_phase_epochs_complete']!=20 or r['support_columns']!=40 or
        not r['all_support_rows_independently_replayed_bitwise'] or r['plan_sha256']!=u.sha(support.REPORT/'plan.json') or
        r['first_phase_log_sha256']!=u.sha(support.REPORT/'tx20lat-first-phase.log')):
        raise ValueError('Complete original40 Tx supports and all-row gather replay required')
    sources={**r['factor_sha256'],**r['support_sha256']}
    for p in [path,support.REPORT/'plan.json',support.REPORT/'tx20lat-first-phase.log']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original Tx output changed')
    paths={s:[support.OUT/f'{name}.{s}.npy' for name in TX_NAMES] for s in ['probex','qual']}
    if any(str(p.relative_to(u.ROOT)) not in r['support_sha256'] for ps in paths.values() for p in ps):
        raise ValueError('Complete fixed Tx component order required')
    return paths,sources

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing full90 cache preserved')
    parent.configure();pm=n.verify_prepared();paths,sources=tx_inputs()
    raw=[np.load(p,mmap_mode='r') for p in paths['probex']]
    old=np.load(parent.RUN/'features.npy',mmap_mode='r')
    if old.shape!=(1128164,293) or pm['support_columns']!=50:raise ValueError('Verified original50 parent schema required')
    for name,digest in pm['prepared_sha256'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    sources[str((parent.REPORT/'manifest.json').relative_to(u.ROOT))]=u.sha(parent.REPORT/'manifest.json')
    rbm=u.ROOT/'experiments/upstream-latent90-local/rbm50lat-completion-verification.json'
    if u.read(rbm)['state']!='complete_original_two_phase_latent_model':raise ValueError('Completed original RBM proof required')
    sources[str(rbm.relative_to(u.ROOT))]=u.sha(rbm)
    RUN.mkdir();REPORT.mkdir()
    for name,digest in pm['prepared_sha256'].items():
        if name=='features.npy':continue
        u.link_or_copy(parent.RUN/name,RUN/name)
        if u.sha(RUN/name)!=digest:raise ValueError('Retained cache changed')
    f=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(1128164,333))
    for lo in range(0,len(f),8192):
        hi=min(lo+8192,len(f));f[lo:hi,:293]=old[lo:hi];f[lo:hi,293:]=np.column_stack([a[lo:hi] for a in raw])
    f.flush()
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(parent.REPORT/name,REPORT/name)
    manifest=dict(pm,code_sha256=code_hashes(),source_manifests_sha256=sources,
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},nonlinear_features=333,support_columns=90,
        new_support_features=pm['new_support_features']+TX_NAMES,tx_first_phase_epochs_complete=20,
        rbm_full_two_phase_completion_claimed=True,tx_full_two_phase_completion_claimed=False,
        scope='Own107 and all50 RBM supports retained exactly; append all20 original Tx user and20 item components, unchanged. Original predfeat uses trainx factors for both query sets, with prior every-row gather replay. Raw predictors/ridge/labels/users/folds unchanged. Tx full second phase continues separately; no full90 two-model completion or qualifying score claimed.')
    u.write(REPORT/'manifest.json',manifest);return verify_columns()

def verify_columns():
    configure();m=n.verify_prepared();paths,_=tx_inputs()
    f=np.load(RUN/'features.npy',mmap_mode='r');old=np.load(parent.RUN/'features.npy',mmap_mode='r')
    raw=[np.load(p,mmap_mode='r') for p in paths['probex']]
    for lo in range(0,len(f),16384):
        hi=min(lo+16384,len(f))
        if not np.array_equal(f[lo:hi,:293].view(np.uint32),old[lo:hi].view(np.uint32)):raise ValueError('Retained features changed')
        for j,a in enumerate(raw):
            if a.shape!=(len(f),) or a.dtype!=np.float32 or not np.isfinite(a[lo:hi]).all() or not np.array_equal(f[lo:hi,293+j].view(np.uint32),a[lo:hi].view(np.uint32)):
                raise ValueError('Original Tx support values changed')
    for name in ['design.npy','labels.npy','users.npy','groups.npy','baseline.npy','stats.npz']:
        if u.sha(RUN/name)!=u.sha(parent.RUN/name):raise ValueError('Retained ridge/row/fold identity changed')
    proof=u.read(REPORT/'ridge-verification.json')
    if proof['screen_sha256']!=u.sha(REPORT/'ridge-screen.json') or not proof['users_disjoint']:raise ValueError('Retained independent ridge proof differs')
    result=dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),
        screen_sha256=u.sha(REPORT/'ridge-screen.json'),source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),
        verifier_sha256=u.sha(Path(__file__)),rows=len(f),prediction_columns=107,support_columns=90,outer_users_disjoint=True,
        all_retained_and_new_columns_checked=True,independent_ridge_proof_reused=True,
        original_support_floats_preserved_bitwise=True,qualifying_labels_read=False,
        scope='All retained293 and appended40 nonlinear columns checked bitwise; unchanged ridge/labels/users/folds. Prior original native-factor gather replay and ridge solve proofs bound. No extra solve or qualifying result.')
    u.write(REPORT/'cache-column-verification.json',result);print(result,flush=True);return result

if __name__=='__main__':
    while not (support.REPORT/'tx20lat-receipt.json').exists():time.sleep(30)
    prepare()
