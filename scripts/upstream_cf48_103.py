#!/usr/bin/env python3
"""Append our completed CF48 epoch95 to the fixed own102 development cache."""
from pathlib import Path
import shutil
import numpy as np
import upstream_cf48_102 as parent
import upstream_cf48_final_screen as screen

u,n=parent.u,parent.n
RUN=u.ROOT/'artifacts/upstream-cf48-103'
REPORT=u.ROOT/'experiments/upstream-cf48-103'
NEW=['cfnade-48_ep95']


def code_hashes():
    paths=[Path(__file__),Path(screen.__file__),u.ROOT/'scripts/verify_upstream_cf48_final_screen.py',
        u.ROOT/'scripts/upstream_remote_cf48_final.py']
    return {**parent.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,u.read(parent.REPORT/'manifest.json')['config'],code_hashes


def native():
    receipt=u.read(u.ROOT/'experiments/upstream-remote-cf48-final/paired-receipt.json')
    if not receipt['full95_model_complete'] or not receipt['reconstructed_native_terminal_and_epoch95_sha256_match']:
        raise ValueError('Complete resumed CF48 final pair required')
    return np.load(u.ROOT/receipt['pairs'][NEW[0]]['probex'],mmap_mode='r',allow_pickle=False)


def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing own103 cache preserved')
    parent.configure();pm=n.verify_prepared();proof=u.read(screen.REPORT/'ridge-verification.json')
    evaluation=u.read(screen.REPORT/'ridge-screen.json')
    if proof['state']!='complete' or not proof['users_disjoint'] or proof['screen_sha256']!=u.sha(screen.REPORT/'ridge-screen.json'):
        raise ValueError('Independent final-snapshot ridge replay required')
    sources=dict(evaluation['source_sha256'])
    for p in [screen.REPORT/'ridge-verification.json',u.ROOT/'scripts/verify_upstream_cf48_final_screen.py']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in pm['prepared_sha256'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Own103 source binding changed: '+name)
    raw=native();old=np.load(parent.RUN/'design.npy',mmap_mode='r');features=np.load(parent.RUN/'features.npy',mmap_mode='r')
    if old.shape!=(len(raw),178) or features.shape!=(len(raw),238):raise ValueError('Own102 parent layout differs')
    RUN.mkdir();REPORT.mkdir()
    z=np.lib.format.open_memmap(RUN/'design.npy',mode='w+',dtype=np.float64,shape=(len(raw),179))
    f=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(len(raw),239))
    for lo in range(0,len(raw),8192):
        hi=min(lo+8192,len(raw));extra=np.clip(raw[lo:hi],0,6)
        z[lo:hi,:177]=old[lo:hi,:177];z[lo:hi,177]=extra;z[lo:hi,178]=old[lo:hi,177]
        f[lo:hi,:238]=features[lo:hi];f[lo:hi,238]=extra-np.float32(3.6)
    z.flush();f.flush()
    for key in ['labels','users','groups']:shutil.copyfile(parent.RUN/(key+'.npy'),RUN/(key+'.npy'))
    shutil.copyfile(screen.RUN/'main-ridge.npy',RUN/'baseline.npy');shutil.copyfile(screen.RUN/'stats.npz',RUN/'stats.npz')
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(screen.REPORT/name,REPORT/name)
    result=dict(config=pm['config'],versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=len(raw),prediction_columns=103,design_columns=179,nonlinear_features=239,
        baseline_rmse=next(r['rmse'] for r in evaluation['records'] if r['arm']=='main'),
        parent_baseline_rmse=pm['baseline_rmse'],new_native_predictions=NEW,full95_model_complete=True,
        qualifying_labels_read=False,isolation=pm['isolation'],arms=pm['arms'],
        scope='Fixed own103 known-ProbeX bank, adding our completed resumed CF48 epoch95. Both published CF48 snapshots25/95 are now included. Cached own102 realization retained; production assembly remains separate. Sender final tensors were verified remotely, not independently rehashed here.')
    u.write(REPORT/'manifest.json',result);return verify_columns()


def verify_columns():
    configure();manifest=n.verify_prepared();old=np.load(parent.RUN/'design.npy',mmap_mode='r');of=np.load(parent.RUN/'features.npy',mmap_mode='r')
    z=np.load(RUN/'design.npy',mmap_mode='r');f=np.load(RUN/'features.npy',mmap_mode='r');raw=native()
    for lo in range(0,len(raw),16384):
        hi=min(lo+16384,len(raw));expected=np.clip(raw[lo:hi],0,6)
        if (not np.array_equal(z[lo:hi,:177],old[lo:hi,:177]) or not np.array_equal(z[lo:hi,-1],old[lo:hi,-1]) or
                not np.array_equal(f[lo:hi,:238],of[lo:hi]) or not np.array_equal(z[lo:hi,177],expected) or
                not np.array_equal(f[lo:hi,238],expected-np.float32(3.6))):raise ValueError('Own103 column transform differs')
    for key in ['labels.npy','users.npy','groups.npy']:
        if u.sha(RUN/key)!=u.sha(parent.RUN/key):raise ValueError('Known row/fold identities differ')
    if u.sha(RUN/'baseline.npy')!=u.sha(screen.RUN/'main-ridge.npy'):raise ValueError('Independent ridge baseline differs')
    proof=u.read(REPORT/'ridge-verification.json')
    if proof['screen_sha256']!=u.sha(REPORT/'ridge-screen.json') or not proof['users_disjoint']:raise ValueError('Independent ridge proof differs')
    result=dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),
        screen_sha256=u.sha(REPORT/'ridge-screen.json'),source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),
        verifier_sha256=u.sha(Path(__file__)),rows=len(raw),prediction_columns=103,outer_users_disjoint=True,
        all_retained_and_new_columns_checked=True,independent_ridge_proof_reused=True,
        full95_model_complete=True,qualifying_labels_read=False,
        scope='Full retained/new cache-column checks, exact row/fold bindings and already independently replayed ridge-vector binding. No extra independent ridge solve claimed.')
    u.write(REPORT/'cache-column-verification.json',result);print(result,flush=True);return result


if __name__=='__main__':prepare()
