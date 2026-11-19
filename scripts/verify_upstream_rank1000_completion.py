#!/usr/bin/env python3
"""Separately bind the original full-rank execution and all native query outputs."""
import argparse
import math
from pathlib import Path
import re
import time
import numpy as np
import upstream_parallel_fullscale as producer

u=producer.u
REPORT=u.ROOT/'experiments/upstream-rank1000-completion'

def verify():
    rp=producer.REPORT/'native-receipt.json';r=u.read(rp)
    proof,prepared=producer.verify_candidate()
    job=u.read(u.BASE/'registry.json')['jobs'][producer.JOB]
    if (r['source_commit']!=u.PIN or r['job']!=producer.JOB or r['command']!=job['command'] or
        r['proof_sha256']!=u.sha(proof) or r['storage']!=prepared or r['factors']!=1000 or
        r['epochs_per_phase']!=20 or r['logical_chunks']!=16 or r['physical_workers']!=4 or
        not r['qualifying_outcomes_zero'] or r['script_sha256']!=u.sha(Path(producer.__file__)) or
        r['binary_sha256']!=u.sha(producer.BASE/'target/release/tsvdx4p-new')):
        raise ValueError('Complete original-rank execution provenance differs')
    if not set(job['outputs']).issubset(r['outputs']):raise ValueError('Declared native outputs missing')
    bindings={}
    for p in [rp,proof,producer.BASE/'parallel-preparation.json',Path(producer.__file__),Path(__file__),u.BASE/'registry.json',producer.BASE/'target/release/tsvdx4p-new']:
        bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in r['outputs'].items():
        p=producer.TREE/name
        if u.sha(p)!=digest:raise ValueError('Native output changed: '+name)
        bindings[str(p.relative_to(u.ROOT))]=digest
    log=producer.TREE/'preds_new'/f'{producer.JOB}.out'
    text=log.read_text()
    if [int(v) for v in re.findall(r'^Epoch\s+(\d+)\s*\|',text,re.M)]!=list(range(1,21))*2:
        raise ValueError('Both original full20-epoch traces required')
    for split,rows in [('trainx',99352343),('probex',1128164),('fulltrain',100480507),('qual',2817131)]:
        directory=producer.TREE/'data'/split
        if directory.resolve()!=(u.FRESH/'data'/split).resolve():raise ValueError('Own data alias differs')
        for p in directory.glob('*.npy'):bindings[str((u.FRESH/'data'/split/p.name).relative_to(u.ROOT))]=u.sha(p)
        p=producer.TREE/'preds_new'/f'{producer.JOB}.{split}.npy';a=np.load(p,mmap_mode='r')
        if str(p.relative_to(producer.TREE)) not in r['outputs'] or a.shape!=(rows,) or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError('Complete native vector invalid')
    for phase in ['trainx','fulltrain']:
        p=producer.TREE/'preds_new'/f'{producer.JOB}.ifeat.{phase}.npy'
        if p.exists():
            a=np.load(p,mmap_mode='r')
            if str(p.relative_to(producer.TREE)) not in r['outputs'] or a.shape!=(17770,1000) or a.dtype!=np.float32 or not np.isfinite(a).all():
                raise ValueError('Original item-factor output invalid')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying targets')
    pred=np.load(producer.TREE/'preds_new'/f'{producer.JOB}.probex.npy',mmap_mode='r')
    labels=np.load(u.FRESH/'data/probex/ratings.npy',mmap_mode='r')
    score=math.sqrt(math.fsum((float(p)-int(y))**2 for p,y in zip(pred,labels))/len(labels))
    printed=re.findall(r'^probex RMSE\s+([\d.]+)',text,re.M)
    if len(printed)!=1 or abs(score-float(printed[0]))>5.1e-6:raise ValueError('Known-query score differs from rounded native trace')
    result=dict(state='complete_original_two_phase_rank1000',observed=producer.dt.datetime.now(producer.dt.timezone.utc).isoformat(),
        source_pin=u.PIN,factors=1000,epochs_per_phase=20,logical_chunks=16,physical_workers=4,
        both_full_epoch_traces=True,native_outputs_verified=len(r['outputs']),standalone_known_probex_rmse=score,
        unchanged_original_data_aliases_verified=True,qualifying_targets_zero=True,
        public_prediction_values_read=False,qualifying_labels_read=False,bindings=bindings,
        scope='Separate original source/storage-parity/binary/data/output binding, full20/20 trace, full native vector shape/finiteness and compensated known-query score checks. Original training is not rerun; no standalone qualifying outcome or blend promotion claimed.')
    u.write(REPORT/'verification.json',result);print({k:v for k,v in result.items() if k!='bindings'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.watch:
        while not (producer.REPORT/'native-receipt.json').exists():time.sleep(30)
    verify()
