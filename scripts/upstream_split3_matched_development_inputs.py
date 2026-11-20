#!/usr/bin/env python3
"""Admit complete newly generated probe features without reusing old qual data."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
from multiprocessing import shared_memory, resource_tracker
from pathlib import Path
import subprocess
import numpy as np
import upstream_repro as u
import upstream_split3_matched_data as data

class Inputs(data.Inputs):
    def __init__(self):
        proof=u.read(data.VOTES);plan_path=data.VOTES.parent/'plan.json';plan=u.read(plan_path)
        if proof['state']!='complete_original_full_two_phase_voting_features_in_ram' or proof['plan_sha256']!=u.sha(plan_path) or proof['qualifying_labels_read']:
            raise ValueError('Complete original full two-phase native execution required')
        names=proof['feature_names'];records=proof['records']
        if (len(names)!=130 or len(set(names))!=130 or len(records)!=260 or
            [(r['query'],r['name'],r['rows']) for r in records]!=[(q,name,count) for q,count in [('probe_3',data.ROWS),('qual',2817131)] for name in names]):
            raise ValueError('Every full native feature frame required')
        mismatches=[r['name'] for r in records if r['query']=='qual' and not r['reference_payload_bitwise_equal']]
        if mismatches!=['vf022_ordinal_svd_std']:raise ValueError('Original preserved feature-parity outcome changed')
        cache={}
        for key,h in plan['source_sha256'].items():
            p=u.ROOT/key;s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
            if identity not in cache:cache[identity]=u.sha(p)
            if cache[identity]!=h:raise ValueError('Original native feature source changed')
        status=subprocess.run(['ps','-p',str(proof['shared_memory']['owner_pid']),'-o','stat='],capture_output=True,text=True).stdout.strip()
        if not status or status.startswith('Z'):raise ValueError('Actual complete RAM owner must remain live')
        if proof['shared_memory']['shape']!=[130,data.ROWS] or proof['shared_memory']['dtype']!='<f4':raise ValueError('Full feature-major RAM shape differs')
        self.segment=shared_memory.SharedMemory(name=proof['shared_memory']['name']);resource_tracker.unregister(self.segment._name,'shared_memory')
        self.voting=np.ndarray((130,data.ROWS),dtype='<f4',buffer=self.segment.buf)
        if data.digest(self.voting)!=proof['probe3_feature_major_payload_sha256']:raise ValueError('Every complete original probe3 RAM bit must match')
        if any(data.digest(self.voting[i])!=records[i]['payload_sha256'] for i in range(130)):raise ValueError('Full per-feature probe stream hashes differ')
        bank=u.read(data.BANK)
        if bank['raw_columns']!=data.RAW:raise ValueError('Complete original own raw bank required')
        self.raw=[np.load(u.ROOT/r['probe3'],mmap_mode='r') for r in bank['columns']]
        self.context=np.load(data.RUN/'context.npy',mmap_mode='r')

def stats():
    manifest=u.read(data.REPORT/'manifest.json')
    if (data.REPORT/'statistics-verification.json').exists():raise ValueError('Existing complete statistics preserved')
    for name,h in manifest['prepared_sha256'].items():
        if u.sha(data.RUN/name)!=h:raise ValueError('Original matched metadata changed')
    inputs=Inputs();y=np.load(data.RUN/'labels.npy');groups=np.load(data.RUN/'groups.npy');canonical=np.load(data.RUN/'canonical.npy')
    grams=np.zeros((2,10,data.DESIGN,data.DESIGN));rhs=np.zeros((2,10,data.DESIGN));hashes=[hashlib.sha256(),hashlib.sha256()]
    try:
        for lo in range(0,data.ROWS,8192):
            hi=min(lo+8192,data.ROWS);z,f=inputs.block(lo,hi)
            hashes[0].update(z.tobytes(order='C'));hashes[1].update(f.tobytes(order='C'))
            if not np.isfinite(z).all() or not np.isfinite(f).all():raise ValueError('Nonfinite full original input')
            for ai in range(2):
                for g in range(10):
                    keep=(groups[lo:hi]==g)&(canonical[lo:hi] if ai==0 else True);a=z[keep]
                    grams[ai,g]+=a.T@a;rhs[ai,g]+=a.T@y[lo:hi][keep]
        np.savez(data.RUN/'stats.npz',gram=grams,rhs=rhs)
        policy=u.ROOT/'SPLIT3_FEATURE_ADMISSION.md'
        u.write(data.REPORT/'statistics-verification.json',dict(state='complete_all_original_full_input_blocks_and_both_arm_group_statistics',
            manifest_sha256=u.sha(data.REPORT/'manifest.json'),voting_verification_sha256=u.sha(data.VOTES),stats_sha256=u.sha(data.RUN/'stats.npz'),
            design_payload_sha256=hashes[0].hexdigest(),feature_payload_sha256=hashes[1].hexdigest(),
            source_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),policy]},
            rows=data.ROWS,all130_new_probe_features_retained=True,old_qualifying_features_consumed=False,
            qualifying_production_requires_complete_new_fulltrain_feature_capture=True,qualifying_labels_read=False,models_fitted=0))
    finally:inputs.close()

if __name__=='__main__':stats()
