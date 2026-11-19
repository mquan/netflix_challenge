#!/usr/bin/env python3
"""Paired110 raw/all90 context cache from independently completed native pairs."""
from pathlib import Path
import numpy as np
import upstream_latent90_bank as parent
import upstream_whole110 as development

u=parent.u
RUN=u.ROOT/'artifacts/upstream-whole110-bank'
REPORT=u.ROOT/'experiments/upstream-whole110-bank'

def inputs():
    pm=u.read(parent.REPORT/'manifest.json');pp=parent.REPORT/'column-verification.json';proof=u.read(pp)
    if (proof['state']!='complete' or proof['manifest_sha256']!=u.sha(parent.REPORT/'manifest.json') or
        not proof['all_retained_columns_checked'] or not proof['all_forty_new_support_columns_bitwise_checked']):
        raise ValueError('Independently checked paired90 parent required')
    dm=u.read(development.REPORT/'manifest.json');dp=development.REPORT/'cache-column-verification.json';checked=u.read(dp)
    if (dm['prediction_columns']!=110 or dm['design_columns']!=186 or dm['nonlinear_features']!=336 or
        checked['state']!='complete_cache_columns' or checked['manifest_sha256']!=u.sha(development.REPORT/'manifest.json') or
        not checked['all_retained_and_new_columns_checked']):
        raise ValueError('Verified whole110 development schema required')
    rows,sources=development.screen.verified_inputs()
    if [r['name'] for r in rows]!=['rbm50lat','tx20lat','tsvdx4p-1000lm']:raise ValueError('Fixed native order differs')
    for collection in [pm['source_sha256'],pm['code_sha256']]:
        for name,digest in collection.items():
            if name in sources and sources[name]!=digest:raise ValueError('Conflicting lineage')
            sources[name]=digest
    for name,digest in pm['files'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    for path in [parent.REPORT/'manifest.json',pp,development.REPORT/'manifest.json',dp]:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound input changed: '+name)
    return rows,sources

def build():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing paired110 cache preserved')
    rows,sources=inputs()
    zeros=u.FRESH/'data/qual/ratings.npy'
    if np.any(np.load(zeros,mmap_mode='r')):raise ValueError('Qualifying targets nonzero')
    sources[str(zeros.relative_to(u.ROOT))]=u.sha(zeros)
    RUN.mkdir();REPORT.mkdir()
    for split,count in [('probex',1128164),('qual',2817131)]:
        old=np.load(parent.RUN/(split+'-design.npy'),mmap_mode='r')
        of=np.load(parent.RUN/(split+'-features.npy'),mmap_mode='r')
        raw=[np.load(u.ROOT/r['paired_files'][split],mmap_mode='r') for r in rows]
        if old.shape!=(count,183) or of.shape!=(count,333):raise ValueError('Paired107/all90 shape differs')
        if any(a.shape!=(count,) or a.dtype!=np.float32 for a in raw):raise ValueError('Native pair shape differs')
        z=np.lib.format.open_memmap(RUN/(split+'-design.npy'),mode='w+',dtype=np.float64,shape=(count,186))
        f=np.lib.format.open_memmap(RUN/(split+'-features.npy'),mode='w+',dtype=np.float32,shape=(count,336))
        for lo in range(0,count,8192):
            hi=min(lo+8192,count);extra=np.column_stack([np.clip(a[lo:hi],0,6) for a in raw])
            if not np.isfinite(extra).all():raise ValueError('Nonfinite native pair')
            z[lo:hi,:182]=old[lo:hi,:182];z[lo:hi,182:185]=extra;z[lo:hi,-1]=old[lo:hi,-1]
            f[lo:hi,:243]=of[lo:hi,:243];f[lo:hi,243:246]=extra-np.float32(3.6);f[lo:hi,246:]=of[lo:hi,243:]
        z.flush();f.flush();del z,f
    for name in ['labels.npy','users.npy']:u.link_or_copy(parent.RUN/name,RUN/name)
    code={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(parent.__file__),Path(development.__file__)]}
    u.write(REPORT/'manifest.json',dict(state='prepared_pending_independent_column_verification',source_pin=u.PIN,
        rows=1128164,qualifying_rows=2817131,prediction_columns=110,design_columns=186,nonlinear_features=336,
        learned_support_columns=90,paired_additions=rows,code_sha256=code,source_sha256=sources,
        files={p.name:u.sha(p) for p in RUN.iterdir()},qualifying_labels_read=False,
        retained_parent_realization_may_differ_from_development=True,
        scope='Retain paired107/all90; insert all three independently completed20/20 native rating pairs in the exact whole110 development order. Prior parent realization differences retained. Own native side outputs extend the released raw schema. No fitting or qualifying score.'))

if __name__=='__main__':build()
