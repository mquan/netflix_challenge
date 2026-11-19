#!/usr/bin/env python3
"""Paired own103 bank; retain the paired91 realization and append fixed12."""
from pathlib import Path
import shutil
import numpy as np
import upstream_extension_bank as parent
import upstream_cf48_103 as development

u=parent.u
RUN=u.ROOT/'artifacts/upstream-cf48-103-bank'
REPORT=u.ROOT/'experiments/upstream-cf48-103-bank'

def code_hashes():
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(parent.__file__),Path(development.__file__)]}

def build():
    manifest=REPORT/'manifest.json'
    if manifest.exists():
        result=u.read(manifest)
        if result['code_sha256']!=code_hashes():raise ValueError('Frozen paired103 implementation changed')
        for name,digest in result['source_sha256'].items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('Paired103 input changed: '+name)
        for name,digest in result['files'].items():
            if u.sha(RUN/name)!=digest:raise ValueError('Paired103 output changed: '+name)
        return result
    if RUN.exists() or REPORT.exists():raise ValueError('Partial paired103 bank preserved')
    base=parent.build()
    sources=dict(base['source_sha256'])
    def bind(path):
        rel=str(path.relative_to(u.ROOT));digest=u.sha(path)
        if rel in sources and sources[rel]!=digest:raise ValueError('Conflicting source lineage')
        sources[rel]=digest
    bind(parent.REPORT/'manifest.json')
    for name,digest in base['files'].items():
        path=parent.RUN/name
        if u.sha(path)!=digest:raise ValueError('Paired91 output differs')
        bind(path)
    dm=u.read(development.REPORT/'manifest.json')
    proof=u.read(development.REPORT/'cache-column-verification.json')
    if dm['prediction_columns']!=103 or proof['manifest_sha256']!=u.sha(development.REPORT/'manifest.json') or not proof['all_retained_and_new_columns_checked']:
        raise ValueError('Verified fixed103 development schema required')
    for path in [development.REPORT/'manifest.json',development.REPORT/'cache-column-verification.json']:
        bind(path)
    for name in ['labels','users']:
        if not np.array_equal(np.load(parent.RUN/(name+'.npy')),np.load(development.RUN/(name+'.npy'))):
            raise ValueError('Known-ProbeX row identity mismatch')
    temporal=development.parent.temporal
    proof=u.read(temporal.REPORT/'manifest.json')
    bind(temporal.REPORT/'manifest.json')
    additions=[]
    for name in temporal.NEW:
        paths={s:u.FRESH/'preds_new'/f'{name}.{s}.npy' for s in ['probex','qual']}
        for p in paths.values():
            if proof['source_manifests_sha256'][str(p.relative_to(u.ROOT))]!=u.sha(p):raise ValueError('Temporal source changed')
        additions.append((name,paths))
    bind(development.parent.screen.REPORT/'plan.json')
    for row in development.parent.selected():
        paths={s:u.ROOT/row['paired_files'][s] for s in ['probex','qual']}
        for s,p in paths.items():
            if u.sha(p)!=row['sha256'][s]:raise ValueError('Fixed ready output changed')
        additions.append((row['name'],paths))
    final=u.ROOT/'experiments/upstream-remote-cf48-final/paired-receipt.json'
    receipt=u.read(final);bind(final)
    if not receipt['full95_model_complete'] or not receipt['reconstructed_native_terminal_and_epoch95_sha256_match']:
        raise ValueError('Completed resumed CF48 pair required')
    additions.append(('cfnade-48_ep95',{s:u.ROOT/receipt['pairs']['cfnade-48_ep95'][s] for s in ['probex','qual']}))
    if len(additions)!=12:raise ValueError('Expected fixed twelve appended predictors')
    for _,paths in additions:
        for p in paths.values():bind(p)
    qual_targets=u.FRESH/'data/qual/ratings.npy'
    if np.any(np.load(qual_targets,mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    bind(qual_targets)
    RUN.mkdir();REPORT.mkdir()
    for split,count in [('probex',1128164),('qual',2817131)]:
        old=np.load(parent.RUN/(split+'-design.npy'),mmap_mode='r')
        of=np.load(parent.RUN/(split+'-features.npy'),mmap_mode='r')
        raw=[np.load(paths[split],mmap_mode='r') for _,paths in additions]
        if old.shape!=(count,167) or of.shape!=(count,227):raise ValueError('Paired91 schema mismatch')
        if any(a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all() for a in raw):raise ValueError('Invalid appended predictor')
        z=np.lib.format.open_memmap(RUN/(split+'-design.npy'),mode='w+',dtype=np.float64,shape=(count,179))
        f=np.lib.format.open_memmap(RUN/(split+'-features.npy'),mode='w+',dtype=np.float32,shape=(count,239))
        for lo in range(0,count,8192):
            hi=min(lo+8192,count);extra=np.column_stack([np.clip(a[lo:hi],0,6) for a in raw])
            z[lo:hi,:166]=old[lo:hi,:166];z[lo:hi,166:178]=extra;z[lo:hi,-1]=old[lo:hi,-1]
            f[lo:hi,:227]=of[lo:hi];f[lo:hi,227:]=extra-np.float32(3.6)
        z.flush();f.flush();del z,f
    for name in ['labels','users']:shutil.copyfile(parent.RUN/(name+'.npy'),RUN/(name+'.npy'))
    result=dict(state='complete_pending_independent_column_verification',source_pin=u.PIN,rows=1128164,qualifying_rows=2817131,
        prediction_columns=103,design_columns=179,nonlinear_features=239,code_sha256=code_hashes(),source_sha256=sources,
        fixed_additions=[name for name,_ in additions],paired_additions=[dict(name=name,files={s:str(p.relative_to(u.ROOT)) for s,p in paths.items()}) for name,paths in additions],
        files={p.name:u.sha(p) for p in RUN.iterdir()},qualifying_labels_read=False,
        scope='Paired91 native realization retained; append temporal2, ready8 and CF48 snapshots25/95. Cached development parent realization differs in previously disclosed nine columns. No new blend fitted or qualifying score computed. Tree91 production must continue using its original paired91 bank/ridge, while a future NN103 uses this separate schema.')
    u.write(manifest,result);return result

if __name__=='__main__':print(build()['state'],flush=True)
