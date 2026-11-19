#!/usr/bin/env python3
"""Append complete fresh CFNADE12 to the actual paired 85-predictor bank."""
from pathlib import Path
import os
import shutil
import numpy as np
import upstream_phase_bank as parent
import upstream_cf12_context as development
from upstream_extension_bank import blocks
from upstream_full_reproduce import receipt_valid

u=parent.u
RUN=u.ROOT/'artifacts/upstream-cf12-bank'
REPORT=u.ROOT/'experiments/upstream-cf12-bank'


def code_hashes():
    paths=[Path(__file__),Path(parent.__file__),parent.CORE_FILE,
        Path(development.__file__),u.ROOT/'scripts/upstream_extension_bank.py']
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}


def build():
    if (REPORT/'manifest.json').exists():
        result=u.read(REPORT/'manifest.json')
        if result['code_sha256']!=code_hashes():raise ValueError('CF12 paired bank code changed')
        for name,digest in {**result['source_sha256'],**{str((RUN/k).relative_to(u.ROOT)):v for k,v in result['files'].items()}}.items():
            if u.sha(u.ROOT/name)!=digest:raise ValueError('CF12 paired bank input changed: '+name)
        return result
    if RUN.exists():raise ValueError('Partial CF12 paired bank preserved')
    fs=os.statvfs(u.ROOT)
    if fs.f_bavail*fs.f_frsize<26*2**30:raise ValueError('Insufficient CF12 paired bank disk headroom')
    base=parent.build();development.configure();dm=development.n.verify_prepared();u.validate_fresh()
    if dm['prediction_columns']!=86 or dm['design_columns']!=162 or dm['nonlinear_features']!=222:
        raise ValueError('CF12 development schema differs')
    jobs=u.read(u.BASE/'registry.json')['jobs']
    if not receipt_valid('cfnade-12',jobs):raise ValueError('Complete paired CF12 source receipt required')
    sources=dict(base['source_sha256'])
    for path in [parent.REPORT/'manifest.json',development.REPORT/'manifest.json',u.BASE/'fresh-receipts/cfnade-12.json']:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in base['files'].items():sources[str((parent.RUN/name).relative_to(u.ROOT))]=digest
    for key in ['labels','users']:
        if not np.array_equal(np.load(parent.RUN/(key+'.npy')),np.load(development.RUN/(key+'.npy'))):
            raise ValueError('CF12 production/development identities differ')
    arrays={}
    for split,count in [('probex',base['rows']),('qual',base['qualifying_rows'])]:
        path=u.FRESH/f'preds_new/cfnade-12.{split}.npy';a=np.load(path,mmap_mode='r',allow_pickle=False)
        if a.shape!=(count,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Malformed paired CF12 prediction')
        if not np.array_equal(a,np.load(u.FRESH/f'preds_new/cfnade-12_ep40.{split}.npy',mmap_mode='r')):
            raise ValueError('CF12 terminal/epoch40 mismatch')
        sources[str(path.relative_to(u.ROOT))]=u.sha(path);arrays[split]=a
    REPORT.mkdir(parents=True,exist_ok=True);RUN.mkdir()
    for split,count in [('probex',base['rows']),('qual',base['qualifying_rows'])]:
        old=np.load(parent.RUN/(split+'-design.npy'),mmap_mode='r')
        features=np.load(parent.RUN/(split+'-features.npy'),mmap_mode='r')
        if old.shape!=(count,161) or features.shape!=(count,221):raise ValueError('Parent paired layout differs')
        z=np.lib.format.open_memmap(RUN/(split+'-design.npy'),mode='w+',dtype=np.float64,shape=(count,162))
        f=np.lib.format.open_memmap(RUN/(split+'-features.npy'),mode='w+',dtype=np.float32,shape=(count,222))
        for lo in range(0,count,8192):
            hi=min(lo+8192,count)
            z[lo:hi],f[lo:hi]=blocks(old[lo:hi],features[lo:hi],arrays[split][lo:hi,None])
        z.flush();f.flush()
    for key in ['labels','users']:shutil.copyfile(parent.RUN/(key+'.npy'),RUN/(key+'.npy'))
    result=dict(state='complete',source_pin=u.PIN,rows=base['rows'],qualifying_rows=base['qualifying_rows'],
        prediction_columns=86,design_columns=162,nonlinear_features=222,code_sha256=code_hashes(),
        source_sha256=sources,files={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        parent_bank_manifest_sha256=u.sha(parent.REPORT/'manifest.json'),qualifying_labels_read=False,
        scope='Actual paired85 bank plus complete own CF12. Original85 mean/centering retained; appended predictor before ridge intercept and after neural features. Only the new NN consumes this86-predictor schema; its reused tree consumes the original85 bank.')
    u.write(REPORT/'manifest.json',result);return result


if __name__=='__main__':build()
