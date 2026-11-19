#!/usr/bin/env python3
"""Fixed own105 cache: retain own103 and append the two verified split3 columns."""
from pathlib import Path
import shutil
import numpy as np
import upstream_cf48_103 as parent
import upstream_split3_105_screen as screen

u,n=parent.u,parent.n
RUN=u.ROOT/'artifacts/upstream-split3-105'
REPORT=u.ROOT/'experiments/upstream-split3-105'
NEW=['s3-attn16','s3-attn16_ep08']

def code_hashes():
    paths=[Path(__file__),Path(screen.__file__),u.ROOT/'scripts/verify_upstream_split3_105_screen.py']
    return {**parent.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}

def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,u.read(parent.REPORT/'manifest.json')['config'],code_hashes

def selected():
    rows=u.read(screen.REPORT/'plan.json')['ready_unused']
    if [r['name'] for r in rows]!=NEW:raise ValueError('Fixed split3 additions changed')
    return rows

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing own105 cache preserved')
    parent.configure();pm=n.verify_prepared()
    proof=u.read(screen.REPORT/'ridge-verification.json');evaluation=u.read(screen.REPORT/'ridge-screen.json')
    if proof['state']!='complete' or not proof['users_disjoint'] or proof['screen_sha256']!=u.sha(screen.REPORT/'ridge-screen.json'):
        raise ValueError('Independent all-two ridge replay required')
    sources=dict(evaluation['source_sha256'])
    for p in [screen.REPORT/'ridge-verification.json',u.ROOT/'scripts/verify_upstream_split3_105_screen.py']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in pm['prepared_sha256'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Source changed: '+name)
    raw=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in selected()]
    old=np.load(parent.RUN/'design.npy',mmap_mode='r');of=np.load(parent.RUN/'features.npy',mmap_mode='r')
    if old.shape!=(len(raw[0]),179) or of.shape!=(len(raw[0]),239):raise ValueError('Own103 schema mismatch')
    RUN.mkdir();REPORT.mkdir()
    z=np.lib.format.open_memmap(RUN/'design.npy',mode='w+',dtype=np.float64,shape=(len(old),181))
    f=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(len(old),241))
    for lo in range(0,len(old),8192):
        hi=min(lo+8192,len(old));extra=np.column_stack([np.clip(a[lo:hi],0,6) for a in raw])
        z[lo:hi,:178]=old[lo:hi,:178];z[lo:hi,178:180]=extra;z[lo:hi,-1]=old[lo:hi,-1]
        f[lo:hi,:239]=of[lo:hi];f[lo:hi,239:]=extra-np.float32(3.6)
    z.flush();f.flush()
    for name in ['labels','users','groups']:shutil.copyfile(parent.RUN/(name+'.npy'),RUN/(name+'.npy'))
    shutil.copyfile(screen.RUN/'main-ridge.npy',RUN/'baseline.npy');shutil.copyfile(screen.RUN/'stats.npz',RUN/'stats.npz')
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(screen.REPORT/name,REPORT/name)
    result=dict(config=pm['config'],versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=len(old),prediction_columns=105,design_columns=181,nonlinear_features=241,
        baseline_rmse=next(r['rmse'] for r in evaluation['records'] if r['arm']=='main'),parent_baseline_rmse=pm['baseline_rmse'],
        new_native_predictions=NEW,qualifying_labels_read=False,isolation=pm['isolation'],arms=pm['arms'],
        full95_model_complete=True,scope='Own103 retained exactly, plus both source-published original split3 attention columns. Verified exact ProbeX projection and absence from split3 training. Fixed nested user folds; no qualifying audit or numerical public predictions.')
    u.write(REPORT/'manifest.json',result);return verify_columns()

def verify_columns():
    configure();manifest=n.verify_prepared();raw=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in selected()]
    z=np.load(RUN/'design.npy',mmap_mode='r');f=np.load(RUN/'features.npy',mmap_mode='r')
    old=np.load(parent.RUN/'design.npy',mmap_mode='r');of=np.load(parent.RUN/'features.npy',mmap_mode='r')
    for lo in range(0,len(z),16384):
        hi=min(lo+16384,len(z))
        if not np.array_equal(z[lo:hi,:178],old[lo:hi,:178]) or not np.array_equal(z[lo:hi,-1],old[lo:hi,-1]) or not np.array_equal(f[lo:hi,:239],of[lo:hi]):
            raise ValueError('Retained cache columns differ')
        for j,a in enumerate(raw):
            clipped=np.clip(a[lo:hi],0,6)
            if not np.array_equal(z[lo:hi,178+j],clipped) or not np.array_equal(f[lo:hi,239+j],clipped-np.float32(3.6)):
                raise ValueError('Split3 transform mismatch')
    for name in ['labels','users','groups']:
        if u.sha(RUN/(name+'.npy'))!=u.sha(parent.RUN/(name+'.npy')):raise ValueError('Known row/fold identity mismatch')
    if u.sha(RUN/'baseline.npy')!=u.sha(screen.RUN/'main-ridge.npy'):raise ValueError('Ridge vector mismatch')
    proof=u.read(REPORT/'ridge-verification.json')
    if proof['screen_sha256']!=u.sha(REPORT/'ridge-screen.json') or not proof['users_disjoint']:raise ValueError('Independent ridge proof mismatch')
    result=dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),
        screen_sha256=u.sha(REPORT/'ridge-screen.json'),source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),
        verifier_sha256=u.sha(Path(__file__)),rows=len(z),prediction_columns=105,outer_users_disjoint=True,
        all_retained_and_new_columns_checked=True,independent_ridge_proof_reused=True,qualifying_labels_read=False,
        scope='Exact retained/new cache columns and row/fold identities checked; binds the independently recomputed ridge solves. No extra independent solve or qualifying score claimed.')
    u.write(REPORT/'cache-column-verification.json',result);print(result,flush=True);return result

if __name__=='__main__':prepare()
