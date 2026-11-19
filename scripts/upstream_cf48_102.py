#!/usr/bin/env python3
"""Prepare the fixed own102 cache: own93 plus ready-eight and CF48 epoch25."""
from pathlib import Path
import shutil
import numpy as np
import upstream_cf48_prefix_screen as screen

u,temporal=screen.u,screen.temporal
n=temporal.n
RUN=u.ROOT/'artifacts/upstream-cf48-102'
REPORT=u.ROOT/'experiments/upstream-cf48-102'
NEW=['dnn-24__asym-16p75','dnn-24__dnn-16','dnn-24__dnn-16p50__asym-16p75',
     'mf-61__asym-16','mf-61__asym-16_ep02','mf-61__asym-16_ep07',
     'rbmx2-0','tsvdx4-l32nb_ep03','cfnade-48_ep25']


def code_hashes():
    paths=[Path(__file__),Path(screen.__file__),u.ROOT/'scripts/verify_upstream_cf48_prefix_screen.py',
        u.ROOT/'scripts/verify_upstream_ready_public_schema.py']
    return {**temporal.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def configure():
    n.RUN,n.REPORT,n.CONFIG,n.code_hashes=RUN,REPORT,u.read(temporal.REPORT/'manifest.json')['config'],code_hashes


def selected():
    plan=u.read(screen.REPORT/'plan.json')
    rows=[r for r in plan['ready_unused'] if r['duplicate_of'] is None]
    if [r['name'] for r in rows]!=NEW:raise ValueError('Fixed own102 additions differ')
    return rows


def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing own102 preparation preserved')
    proof=u.read(screen.REPORT/'ridge-verification.json');evaluation=u.read(screen.REPORT/'ridge-screen.json')
    if (proof['state']!='complete' or not proof['users_disjoint'] or
            proof['screen_sha256']!=u.sha(screen.REPORT/'ridge-screen.json') or
            evaluation['fixed_additions']!=NEW):raise ValueError('Independent fixed all-nine replay required')
    sources=dict(evaluation['source_sha256'])
    for path in [screen.REPORT/'ridge-verification.json',u.ROOT/'scripts/verify_upstream_cf48_prefix_screen.py']:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen own102 source changed: '+name)
    old_manifest=u.read(temporal.REPORT/'manifest.json')
    for name,digest in old_manifest['prepared_sha256'].items():
        if u.sha(temporal.RUN/name)!=digest:raise ValueError('Own93 parent changed')
    rows=selected();raw=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r',allow_pickle=False) for r in rows]
    old=np.load(temporal.RUN/'design.npy',mmap_mode='r');old_features=np.load(temporal.RUN/'features.npy',mmap_mode='r')
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy')
    if old.shape!=(len(y),169) or old_features.shape!=(len(y),229):raise ValueError('Original own93 layout differs')
    RUN.mkdir();REPORT.mkdir()
    z=np.lib.format.open_memmap(RUN/'design.npy',mode='w+',dtype=np.float64,shape=(len(y),178))
    f=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(len(y),238))
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));extra=np.column_stack([np.clip(a[lo:hi],0,6) for a in raw])
        z[lo:hi,:168]=old[lo:hi,:168];z[lo:hi,168:177]=extra;z[lo:hi,-1]=old[lo:hi,-1]
        f[lo:hi,:229]=old_features[lo:hi];f[lo:hi,229:]=extra-np.float32(3.6)
    z.flush();f.flush()
    for key in ['labels','users','groups']:shutil.copyfile(temporal.RUN/(key+'.npy'),RUN/(key+'.npy'))
    shutil.copyfile(screen.RUN/'main-ridge.npy',RUN/'baseline.npy')
    gram,rhs=n.grouped_stats(z,y,groups);np.savez(RUN/'stats.npz',gram=gram,rhs=rhs)
    for name in ['plan.json','ridge-screen.json','ridge-verification.json']:shutil.copyfile(screen.REPORT/name,REPORT/name)
    score=next(r['rmse'] for r in evaluation['records'] if r['arm']=='main')
    result=dict(config=old_manifest['config'],versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=len(y),prediction_columns=102,design_columns=178,nonlinear_features=238,
        baseline_rmse=score,parent_baseline_rmse=old_manifest['baseline_rmse'],new_native_predictions=NEW,
        qualifying_labels_read=False,full95_model_complete=False,isolation=old_manifest['isolation'],
        scope='Fixed own102 known-ProbeX cache. Cached own93 realization retained; eight complete own outputs plus our original95-epoch CF48 trajectory snapshot25. Both snapshot query vectors are complete; whole95-epoch job remains in progress. No production bank or qualifying audit.',
        arms=['control','tree','nn','fixed-half-tree-nn'])
    u.write(REPORT/'manifest.json',result)
    return verify_columns()


def verify_columns():
    configure();manifest=n.verify_prepared();rows=selected()
    old=np.load(temporal.RUN/'design.npy',mmap_mode='r');old_f=np.load(temporal.RUN/'features.npy',mmap_mode='r')
    z=np.load(RUN/'design.npy',mmap_mode='r');f=np.load(RUN/'features.npy',mmap_mode='r')
    raw=[np.load(u.ROOT/r['paired_files']['probex'],mmap_mode='r') for r in rows]
    for lo in range(0,len(z),16384):
        hi=min(lo+16384,len(z))
        if not np.array_equal(z[lo:hi,:168],old[lo:hi,:168]) or not np.array_equal(z[lo:hi,-1],old[lo:hi,-1]):
            raise ValueError('Retained ridge columns differ')
        if not np.array_equal(f[lo:hi,:229],old_f[lo:hi]):raise ValueError('Retained neural features differ')
        for j,values in enumerate(raw):
            a=np.clip(values[lo:hi],0,6)
            if not np.array_equal(z[lo:hi,168+j],a) or not np.array_equal(f[lo:hi,229+j],a-np.float32(3.6)):
                raise ValueError('New native column transform differs')
    for key in ['labels.npy','users.npy','groups.npy']:
        if manifest['prepared_sha256'][key]!=u.sha(temporal.RUN/key):raise ValueError('Known row/fold identities differ')
    if u.sha(RUN/'baseline.npy')!=u.sha(screen.RUN/'main-ridge.npy'):raise ValueError('Verified ridge vector differs')
    proof=u.read(REPORT/'ridge-verification.json')
    if not proof['users_disjoint'] or proof['screen_sha256']!=u.sha(REPORT/'ridge-screen.json'):
        raise ValueError('Independent ridge proof binding differs')
    result=dict(state='complete_cache_columns',manifest_sha256=u.sha(REPORT/'manifest.json'),
        screen_sha256=u.sha(REPORT/'ridge-screen.json'),source_ridge_verification_sha256=u.sha(REPORT/'ridge-verification.json'),
        verifier_sha256=u.sha(Path(__file__)),rows=len(z),prediction_columns=102,
        outer_users_disjoint=True,all_retained_and_new_columns_checked=True,independent_ridge_proof_reused=True,
        full95_model_complete=False,qualifying_labels_read=False,
        scope='Complete cache-column and row/fold identity checks, binding the already independently replayed all-nine ridge vector. No new independent ridge solve is claimed by this cache check.')
    u.write(REPORT/'cache-column-verification.json',result);print(result,flush=True);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    verify_columns() if args.verify else prepare()
