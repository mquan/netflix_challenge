#!/usr/bin/env python3
"""Paired own107/all90 context; retain the entire verified paired50 bank."""
from pathlib import Path
import numpy as np
import upstream_rbm50_bank as parent
import upstream_latent90_features as development

u=parent.u
RUN=u.ROOT/'artifacts/upstream-latent90-bank'
REPORT=u.ROOT/'experiments/upstream-latent90-bank'

def build():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing paired90 cache preserved')
    pm=u.read(parent.REPORT/'manifest.json');pp=parent.REPORT/'column-verification.json';proof=u.read(pp)
    if proof['state']!='complete' or proof['manifest_sha256']!=u.sha(parent.REPORT/'manifest.json'):
        raise ValueError('Complete separately verified paired50 parent required')
    sources={**pm['source_sha256'],**pm['code_sha256']}
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen parent source changed')
    for name,digest in pm['files'].items():
        path=parent.RUN/name
        if u.sha(path)!=digest:raise ValueError('Paired50 parent changed')
        sources[str(path.relative_to(u.ROOT))]=digest
    dm=u.read(development.REPORT/'manifest.json');dp=development.REPORT/'cache-column-verification.json';checked=u.read(dp)
    if dm['support_columns']!=90 or checked['manifest_sha256']!=u.sha(development.REPORT/'manifest.json') or not checked['original_support_floats_preserved_bitwise']:
        raise ValueError('Complete checked full90 development schema required')
    paths,tx_sources=development.tx_inputs();sources.update(tx_sources)
    for path in [parent.REPORT/'manifest.json',pp,development.REPORT/'manifest.json',dp]:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    zeros=u.FRESH/'data/qual/ratings.npy'
    if np.any(np.load(zeros,mmap_mode='r')):raise ValueError('Qualifying targets nonzero')
    sources[str(zeros.relative_to(u.ROOT))]=u.sha(zeros)
    RUN.mkdir();REPORT.mkdir()
    for split,rows in [('probex',1128164),('qual',2817131)]:
        old=np.load(parent.RUN/(split+'-features.npy'),mmap_mode='r')
        raw=[np.load(p,mmap_mode='r') for p in paths[split]]
        if old.shape!=(rows,293) or any(a.shape!=(rows,) or a.dtype!=np.float32 for a in raw):raise ValueError('Paired source shape differs')
        f=np.lib.format.open_memmap(RUN/(split+'-features.npy'),mode='w+',dtype=np.float32,shape=(rows,333))
        for lo in range(0,rows,8192):
            hi=min(lo+8192,rows);extra=np.column_stack([a[lo:hi] for a in raw])
            if not np.isfinite(extra).all():raise ValueError('Nonfinite original Tx context')
            f[lo:hi,:293]=old[lo:hi];f[lo:hi,293:]=extra
        f.flush();del f
        u.link_or_copy(parent.RUN/(split+'-design.npy'),RUN/(split+'-design.npy'))
    for name in ['labels.npy','users.npy']:u.link_or_copy(parent.RUN/name,RUN/name)
    result=dict(state='prepared_pending_independent_column_verification',source_pin=u.PIN,
        rows=1128164,qualifying_rows=2817131,prediction_columns=107,design_columns=183,nonlinear_features=333,
        learned_support_columns=90,new_support_names=development.TX_NAMES,
        source_sha256=sources,code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(parent.__file__),Path(development.__file__)]},
        files={p.name:u.sha(p) for p in RUN.iterdir()},qualifying_labels_read=False,
        retained_parent_realization_may_differ_from_development=True,
        scope='Paired107/50 retained bitwise; append all40 original Tx trainx user/item supports without rating transforms. Original support gathers are independently checked on all rows of both queries. Ridge/known identities unchanged. No fitting, whole-Tx completion or qualifying score claimed; previous nine parent realization differences retained.')
    u.write(REPORT/'manifest.json',result);print(result['state'],flush=True)

if __name__=='__main__':build()
