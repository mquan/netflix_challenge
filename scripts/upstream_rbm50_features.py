#!/usr/bin/env python3
"""Append the exact original50 RBM user supports to own107 nonlinear context."""
from pathlib import Path
import shutil
import numpy as np
import upstream_ready107 as parent
import upstream_latent90_early_support as support

u,n=parent.u,parent.n
RUN=u.ROOT/'artifacts/upstream-rbm50-features'
REPORT=u.ROOT/'experiments/upstream-rbm50-features'
NEW=parent.NEW  # Raw-predictor schema retained; these50 additions are context.
SUPPORT_NAMES=[f'rbm50lat.u{j:02}' for j in range(50)]

def code_hashes():
    return {**parent.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(support.__file__)]}}

def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,u.read(parent.REPORT/'manifest.json')['config'],code_hashes

def inputs():
    path=support.REPORT/'rbm50lat-receipt.json';receipt=u.read(path)
    if (receipt['state']!='complete_original_first_phase_paired_supports' or receipt['model']!='rbm50lat' or
            receipt['first_phase_epochs_complete']!=20 or receipt['support_columns']!=50 or
            not receipt['all_support_rows_independently_replayed_bitwise'] or
            receipt['plan_sha256']!=u.sha(support.REPORT/'plan.json') or
            receipt['first_phase_log_sha256']!=u.sha(support.REPORT/'rbm50lat-first-phase.log')):
        raise ValueError('Complete original50-user-support family and gather replay required')
    sources={**receipt['factor_sha256'],**receipt['support_sha256']}
    for p in [path,support.REPORT/'plan.json',support.REPORT/'rbm50lat-first-phase.log']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original support source/output changed: '+name)
    paths={s:[support.OUT/f'{name}.{s}.npy' for name in SUPPORT_NAMES] for s in ['probex','qual']}
    if any(str(p.relative_to(u.ROOT)) not in receipt['support_sha256'] for ps in paths.values() for p in ps):
        raise ValueError('Ordered complete support columns missing')
    return paths,sources

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing support cache preserved')
    parent.configure();pm=n.verify_prepared();paths,sources=inputs()
    old=np.load(parent.RUN/'features.npy',mmap_mode='r')
    if old.shape!=(pm['rows'],243) or pm['prediction_columns']!=107:raise ValueError('Own107 parent schema mismatch')
    raw=[np.load(p,mmap_mode='r') for p in paths['probex']]
    if any(a.shape!=(pm['rows'],) or a.dtype!=np.float32 for a in raw):raise ValueError('Support query shape/dtype mismatch')
    for name,digest in pm['prepared_sha256'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    sources[str((parent.REPORT/'manifest.json').relative_to(u.ROOT))]=u.sha(parent.REPORT/'manifest.json')
    RUN.mkdir();REPORT.mkdir()
    for name,digest in pm['prepared_sha256'].items():
        if name=='features.npy':continue
        u.link_or_copy(parent.RUN/name,RUN/name)
        if u.sha(RUN/name)!=digest:raise ValueError('Retained ridge/row/fold cache changed')
    f=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(pm['rows'],293))
    for lo in range(0,pm['rows'],8192):
        hi=min(lo+8192,pm['rows']);f[lo:hi,:243]=old[lo:hi]
        # Learned context: preserve every original float without rating clipping
        # or rating centering. Scalers fit only eligible known-user folds later.
        f[lo:hi,243:]=np.column_stack([a[lo:hi] for a in raw])
    f.flush()
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(parent.REPORT/name,REPORT/name)
    manifest=dict(config=pm['config'],versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=pm['rows'],prediction_columns=107,design_columns=183,nonlinear_features=293,
        baseline_rmse=pm['baseline_rmse'],parent_baseline_rmse=pm['baseline_rmse'],new_native_predictions=NEW,
        support_columns=50,new_support_features=SUPPORT_NAMES,qualifying_labels_read=False,isolation=pm['isolation'],arms=pm['arms'],
        rbm_first_phase_epochs_complete=20,rbm_full_two_phase_completion_claimed=False,
        scope='Own107 retained exactly, plus all50 original learned RBM user components from the completed20-epoch trainx model. Original predfeat maps that model onto BOTH query sets; every support row previously replayed bitwise. Raw-predictor/ridge schema and baseline unchanged. Original two-phase training continues separately.')
    u.write(REPORT/'manifest.json',manifest);return verify_columns()

def verify_columns():
    configure();m=n.verify_prepared();paths,sources=inputs()
    f=np.load(RUN/'features.npy',mmap_mode='r');old=np.load(parent.RUN/'features.npy',mmap_mode='r')
    raw=[np.load(p,mmap_mode='r') for p in paths['probex']]
    for lo in range(0,len(f),16384):
        hi=min(lo+16384,len(f))
        if not np.array_equal(f[lo:hi,:243].view(np.uint32),old[lo:hi].view(np.uint32)):raise ValueError('Retained features differ')
        for j,a in enumerate(raw):
            if not np.array_equal(f[lo:hi,243+j].view(np.uint32),a[lo:hi].view(np.uint32)):
                raise ValueError('Original support values changed')
    for name in ['design.npy','labels.npy','users.npy','groups.npy','baseline.npy','stats.npz']:
        if u.sha(RUN/name)!=u.sha(parent.RUN/name):raise ValueError('Ridge/row/fold identity differs')
    proof=u.read(REPORT/'ridge-verification.json')
    if proof['screen_sha256']!=u.sha(REPORT/'ridge-screen.json') or not proof['users_disjoint']:raise ValueError('Retained independent ridge proof differs')
    result=dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),
        screen_sha256=u.sha(REPORT/'ridge-screen.json'),source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),
        verifier_sha256=u.sha(Path(__file__)),rows=len(f),prediction_columns=107,support_columns=50,outer_users_disjoint=True,
        all_retained_and_new_columns_checked=True,independent_ridge_proof_reused=True,
        original_support_floats_preserved_bitwise=True,qualifying_labels_read=False,
        scope='All nonlinear context columns checked against exact retained/learned source values. Ridge, labels, users and folds remain byte-identical to own107. Earlier all-row native-factor gather proof and independent ridge replay retained; no new ridge solve or qualifying audit claimed.')
    u.write(REPORT/'cache-column-verification.json',result);print(result,flush=True);return result

if __name__=='__main__':prepare()
