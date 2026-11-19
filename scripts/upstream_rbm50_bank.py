#!/usr/bin/env python3
"""Prepare paired own107 plus50 original RBM context; never read outcomes."""
from pathlib import Path
import numpy as np
import upstream_cf48_103_bank as parent
import upstream_split3_105 as split3
import upstream_ready107 as raw107
import upstream_rbm50_features as development

u=parent.u
RUN=u.ROOT/'artifacts/upstream-rbm50-bank'
REPORT=u.ROOT/'experiments/upstream-rbm50-bank'

def build():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing paired bank preserved')
    sources={}
    def bind(path,digest=None):
        actual=u.sha(path)
        if digest is not None and actual!=digest:raise ValueError('Source changed: '+str(path))
        key=str(path.relative_to(u.ROOT))
        if key in sources and sources[key]!=actual:raise ValueError('Conflicting source binding')
        sources[key]=actual
    pm=u.read(parent.REPORT/'manifest.json')
    pp=parent.REPORT/'column-verification.json';proof=u.read(pp)
    if proof['state']!='complete' or proof['manifest_sha256']!=u.sha(parent.REPORT/'manifest.json'):
        raise ValueError('Verified paired103 parent required')
    for name,digest in {**pm['code_sha256'],**pm['source_sha256']}.items():bind(u.ROOT/name,digest)
    for name,digest in pm['files'].items():bind(parent.RUN/name,digest)
    bind(parent.REPORT/'manifest.json');bind(pp)
    dm=u.read(development.REPORT/'manifest.json');dp=development.REPORT/'cache-column-verification.json'
    checked=u.read(dp)
    if (dm['prediction_columns']!=107 or dm['support_columns']!=50 or
            checked['manifest_sha256']!=u.sha(development.REPORT/'manifest.json') or
            not checked['original_support_floats_preserved_bitwise']):
        raise ValueError('Verified own107 plus50 development schema required')
    bind(development.REPORT/'manifest.json');bind(dp)
    support_paths,support_sources=development.inputs()
    for name,digest in support_sources.items():bind(u.ROOT/name,digest)
    additions=split3.selected()+raw107.selected()
    if [r['name'] for r in additions]!=['s3-attn16','s3-attn16_ep08','cfnade-96_ep10','tsvdx5-60']:
        raise ValueError('Fixed addition order changed')
    for mod in [split3,raw107]:
        path=mod.screen.REPORT/'ridge-screen.json';screen=u.read(path)
        bind(mod.screen.REPORT/'plan.json');bind(path)
        for name,digest in screen['source_sha256'].items():bind(u.ROOT/name,digest)
    for row in additions:
        for path in row['paired_files'].values():bind(u.ROOT/path)
    zeros=u.FRESH/'data/qual/ratings.npy'
    if np.any(np.load(zeros,mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    bind(zeros)
    for name in ['labels','users']:
        if not np.array_equal(np.load(parent.RUN/(name+'.npy')),np.load(development.RUN/(name+'.npy'))):
            raise ValueError('Known row identity mismatch')
    RUN.mkdir();REPORT.mkdir()
    for split,count in [('probex',1128164),('qual',2817131)]:
        old=np.load(parent.RUN/(split+'-design.npy'),mmap_mode='r')
        of=np.load(parent.RUN/(split+'-features.npy'),mmap_mode='r')
        raw=[np.load(u.ROOT/r['paired_files'][split],mmap_mode='r') for r in additions]
        supports=[np.load(p,mmap_mode='r') for p in support_paths[split]]
        if old.shape!=(count,179) or of.shape!=(count,239):raise ValueError('Paired103 shape mismatch')
        if any(a.shape!=(count,) or a.dtype!=np.float32 for a in raw+supports):raise ValueError('Invalid native query shape')
        z=np.lib.format.open_memmap(RUN/(split+'-design.npy'),mode='w+',dtype=np.float64,shape=(count,183))
        f=np.lib.format.open_memmap(RUN/(split+'-features.npy'),mode='w+',dtype=np.float32,shape=(count,293))
        for lo in range(0,count,8192):
            hi=min(lo+8192,count)
            extra=np.column_stack([np.clip(a[lo:hi],0,6) for a in raw])
            context=np.column_stack([a[lo:hi] for a in supports])
            if not np.isfinite(extra).all() or not np.isfinite(context).all():raise ValueError('Nonfinite native output')
            z[lo:hi,:178]=old[lo:hi,:178];z[lo:hi,178:182]=extra;z[lo:hi,-1]=old[lo:hi,-1]
            f[lo:hi,:239]=of[lo:hi];f[lo:hi,239:243]=extra-np.float32(3.6);f[lo:hi,243:]=context
        z.flush();f.flush();del z,f
    for name in ['labels','users']:u.link_or_copy(parent.RUN/(name+'.npy'),RUN/(name+'.npy'))
    code={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),Path(parent.__file__),Path(development.__file__)]}
    result=dict(state='prepared_pending_independent_column_verification',source_pin=u.PIN,
        rows=1128164,qualifying_rows=2817131,prediction_columns=107,design_columns=183,nonlinear_features=293,
        rbm_user_support_columns=50,code_sha256=code,source_sha256=sources,
        paired_additions=additions,support_names=development.SUPPORT_NAMES,
        files={p.name:u.sha(p) for p in RUN.iterdir()},qualifying_labels_read=False,
        retained_parent_realization_may_differ_from_development=True,
        scope='Paired103 retained; append split3 terminal/ep08, CF96 ep10, temporal60 and50 unchanged original trainx RBM user supports. No fit, promotion or qualifying score. Prior nine parent realization differences retained. Whole RBM two-phase completion is not claimed by the support receipt.')
    u.write(REPORT/'manifest.json',result);print(result['state'],flush=True)

if __name__=='__main__':build()
