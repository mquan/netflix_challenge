#!/usr/bin/env python3
"""Own109 raw predictors/all90 context cache, with every column checked."""
from pathlib import Path
import shutil
import numpy as np
import upstream_latent90_features as parent
import upstream_autorec_pair2_screen as screen

u,n=parent.u,parent.n
RUN=u.ROOT/'artifacts/upstream-autorec109'
REPORT=u.ROOT/'experiments/upstream-autorec109'
NEW=screen.NEW

def code_hashes():
    return {**parent.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(screen.__file__),u.ROOT/'scripts/verify_upstream_autorec_pair2_screen.py',u.ROOT/'scripts/verify_upstream_ready_public_schema.py']}}

def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,u.read(parent.REPORT/'manifest.json')['config'],code_hashes

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing own109 cache preserved')
    parent.configure();pm=n.verify_prepared();proof=u.read(screen.REPORT/'ridge-verification.json');evaluation=u.read(screen.REPORT/'ridge-screen.json')
    if proof['state']!='complete' or not proof['users_disjoint'] or proof['screen_sha256']!=u.sha(screen.REPORT/'ridge-screen.json'):
        raise ValueError('Independent all-two ridge replay required')
    rows,sources=screen.verified_inputs();sources.update(evaluation['source_sha256'])
    for p in [screen.REPORT/'ridge-verification.json',u.ROOT/'scripts/verify_upstream_autorec_pair2_screen.py',parent.REPORT/'manifest.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in pm['prepared_sha256'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Completed source changed')
    raw=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in rows]
    old=np.load(parent.RUN/'design.npy',mmap_mode='r');of=np.load(parent.RUN/'features.npy',mmap_mode='r')
    if old.shape!=(1128164,183) or of.shape!=(1128164,333):raise ValueError('Own107/all90 schema differs')
    RUN.mkdir();REPORT.mkdir()
    z=np.lib.format.open_memmap(RUN/'design.npy',mode='w+',dtype=np.float64,shape=(1128164,185))
    f=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(1128164,335))
    for lo in range(0,len(z),8192):
        hi=min(lo+8192,len(z));extra=np.column_stack([np.clip(a[lo:hi],0,6) for a in raw])
        z[lo:hi,:182]=old[lo:hi,:182];z[lo:hi,182:184]=extra;z[lo:hi,-1]=old[lo:hi,-1]
        f[lo:hi,:243]=of[lo:hi,:243];f[lo:hi,243:245]=extra-np.float32(3.6);f[lo:hi,245:]=of[lo:hi,243:]
    z.flush();f.flush()
    for name in ['labels','users','groups']:u.link_or_copy(parent.RUN/(name+'.npy'),RUN/(name+'.npy'))
    shutil.copyfile(screen.RUN/'main-ridge.npy',RUN/'baseline.npy');shutil.copyfile(screen.RUN/'stats.npz',RUN/'stats.npz')
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(screen.REPORT/name,REPORT/name)
    result=dict(pm,code_sha256=code_hashes(),source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        prediction_columns=109,design_columns=185,nonlinear_features=335,
        baseline_rmse=next(r['rmse'] for r in evaluation['records'] if r['arm']=='main'),parent_baseline_rmse=pm['baseline_rmse'],
        new_native_predictions=NEW,raw_snapshot_pairs_complete=True,raw_full50_model_completion_claimed=False,tx_full_two_phase_completion_claimed=True,
        scope='Own107/all90 retained, add both complete own raw autoencoder snapshot predictors before the90 context components. Original folds/penalty; no numerical public predictions or qualifying outcomes. New own autoencoder predictions extend the released raw list; not exact raw-schema reproduction.')
    u.write(REPORT/'manifest.json',result);return verify_columns()

def verify_columns():
    configure();m=n.verify_prepared();rows,_=screen.verified_inputs()
    z=np.load(RUN/'design.npy',mmap_mode='r');f=np.load(RUN/'features.npy',mmap_mode='r')
    old=np.load(parent.RUN/'design.npy',mmap_mode='r');of=np.load(parent.RUN/'features.npy',mmap_mode='r')
    raw=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in rows]
    for lo in range(0,len(z),16384):
        hi=min(lo+16384,len(z))
        if not np.array_equal(z[lo:hi,:182],old[lo:hi,:182]) or not np.array_equal(z[lo:hi,-1],old[lo:hi,-1]):raise ValueError('Retained ridge changed')
        if not np.array_equal(f[lo:hi,:243].view(np.uint32),of[lo:hi,:243].view(np.uint32)) or not np.array_equal(f[lo:hi,245:].view(np.uint32),of[lo:hi,243:].view(np.uint32)):
            raise ValueError('Retained raw/context values changed')
        for j,a in enumerate(raw):
            clipped=np.clip(a[lo:hi],0,6)
            if not np.array_equal(z[lo:hi,182+j],clipped) or not np.array_equal(f[lo:hi,243+j],clipped-np.float32(3.6)):
                raise ValueError('New rating transform differs')
    for name in ['labels','users','groups']:
        if u.sha(RUN/(name+'.npy'))!=u.sha(parent.RUN/(name+'.npy')):raise ValueError('Known identities changed')
    if u.sha(RUN/'baseline.npy')!=u.sha(screen.RUN/'main-ridge.npy'):raise ValueError('Ridge vector differs')
    proof=u.read(REPORT/'ridge-verification.json')
    if proof['screen_sha256']!=u.sha(REPORT/'ridge-screen.json') or not proof['users_disjoint']:raise ValueError('Independent ridge proof differs')
    result=dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),screen_sha256=u.sha(REPORT/'ridge-screen.json'),
        source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),verifier_sha256=u.sha(Path(__file__)),
        rows=len(z),prediction_columns=109,support_columns=90,outer_users_disjoint=True,
        all_retained_and_new_columns_checked=True,independent_ridge_proof_reused=True,original_support_floats_preserved_bitwise=True,
        qualifying_labels_read=False,scope='All retained/new raw and learned context columns checked, known identities unchanged and independent ridge replay bound. No extra solve, nonlinear fit or qualifying score claimed.')
    u.write(REPORT/'cache-column-verification.json',result);print(result,flush=True);return result

if __name__=='__main__':prepare()
