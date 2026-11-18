#!/usr/bin/env python3
"""Append the same fixed six native columns to the paired 85-column bank."""
import os
from pathlib import Path
import shutil
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import upstream_phase_bank as parent
import upstream_source_extension as development

u=parent.u
RUN=u.ROOT/'artifacts/upstream-extension-bank'
REPORT=u.ROOT/'experiments/upstream-extension-bank'


def code_hashes():
    paths=[Path(__file__),Path(parent.__file__),parent.CORE_FILE,Path(development.__file__)]
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}


def blocks(design,features,added):
    added=np.clip(np.asarray(added,dtype=np.float32),0,6)
    return (np.column_stack([design[:,:-1],added,design[:,-1:]]),
            np.column_stack([features,added-3.6]).astype(np.float32))


def build():
    if (REPORT/'manifest.json').exists():
        result=u.read(REPORT/'manifest.json')
        if result['extension_code_sha256']!=code_hashes():raise ValueError('Extension bank code changed')
        for path,digest in {**result['source_sha256'],**{str((RUN/name).relative_to(u.ROOT)):v for name,v in result['files'].items()}}.items():
            if u.sha(u.ROOT/path)!=digest:raise ValueError('Extension bank input changed: '+path)
        return result
    if RUN.exists():raise ValueError('Partial extension bank preserved')
    disk=os.statvfs(u.ROOT)
    if disk.f_bavail*disk.f_frsize<26*2**30:raise ValueError('Insufficient extension bank disk headroom')
    base=parent.build();development.configure();dm=development.n.verify_prepared()
    if dm['new_native_predictions']!=development.NEW or dm['prediction_columns']!=91:
        raise ValueError('Development extension recipe differs')
    added,sources=development.added_values(base['rows'])
    sources.update(base['source_sha256'])
    for path in [parent.REPORT/'manifest.json',development.REPORT/'manifest.json']:
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    sources.update({str((parent.RUN/name).relative_to(u.ROOT)):digest for name,digest in base['files'].items()})
    y=np.load(parent.RUN/'labels.npy');users=np.load(parent.RUN/'users.npy')
    if not np.array_equal(y,np.load(development.RUN/'labels.npy')) or not np.array_equal(users,np.load(development.RUN/'users.npy')):
        raise ValueError('Paired extension development row identities differ')
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    for split,rows in [('probex',base['rows']),('qual',base['qualifying_rows'])]:
        old=np.load(parent.RUN/(split+'-design.npy'),mmap_mode='r')
        features=np.load(parent.RUN/(split+'-features.npy'),mmap_mode='r')
        arrays=[np.load(u.FRESH/'preds_new'/(name+'.qual.npy'),mmap_mode='r') for name in development.NEW] if split=='qual' else None
        z=np.lib.format.open_memmap(RUN/(split+'-design.npy'),mode='w+',dtype=np.float64,shape=(rows,167))
        f=np.lib.format.open_memmap(RUN/(split+'-features.npy'),mode='w+',dtype=np.float32,shape=(rows,227))
        for lo in range(0,rows,8192):
            hi=min(lo+8192,rows)
            extra=added[lo:hi] if arrays is None else np.column_stack([a[lo:hi] for a in arrays])
            if not np.isfinite(extra).all():raise ValueError('Nonfinite paired extension predictor')
            z[lo:hi],f[lo:hi]=blocks(old[lo:hi],features[lo:hi],extra)
        z.flush();f.flush()
    for key in ['labels','users']:shutil.copyfile(parent.RUN/(key+'.npy'),RUN/(key+'.npy'))
    result=dict(state='complete',source_pin=u.PIN,rows=base['rows'],qualifying_rows=base['qualifying_rows'],
        prediction_columns=91,new_native_predictions=development.NEW,extension_code_sha256=code_hashes(),
        source_sha256=sources,files={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        qualifying_labels_read=False,design_columns=167,nonlinear_features=227,
        scope='Same paired 85-column schema plus all six fixed fresh native models, preserving parent transforms/intercept. All final weights fit only known ProbeX; no recovered outcomes in assembly.')
    u.write(REPORT/'manifest.json',result);return result


if __name__=='__main__':build()
