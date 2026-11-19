#!/usr/bin/env python3
"""Check the actual original23/23 temporal job, all outputs and input bindings."""
from pathlib import Path
import re
import numpy as np
import upstream_repro as u
from upstream_full_reproduce import receipt_valid

def main():
    u.validate_fresh();name='tsvdx5-60';jobs=u.read(u.BASE/'registry.json')['jobs']
    if not receipt_valid(name,jobs):raise ValueError('Complete original native receipt required')
    receiptpath=u.BASE/'fresh-receipts'/(name+'.json');receipt=u.read(receiptpath)
    log=u.FRESH/'preds_new'/f'{name}.out';text=log.read_text();parts=text.split('fulltrain => qual\n')
    if len(parts)!=2 or 'trainx => probex\n' not in parts[0]:raise ValueError('Original two-phase trace required')
    for part in parts:
        if [int(v) for v in re.findall(r'^Epoch\s+(\d+)(?:\s|$)',part,re.M)]!=list(range(1,24)):
            raise ValueError('Both original23-epoch phases required')
    bindings={str(receiptpath.relative_to(u.ROOT)):u.sha(receiptpath)};arrays={}
    for rel,digest in receipt['outputs'].items():
        path=u.FRESH/rel
        if u.sha(path)!=digest:raise ValueError('Native output changed')
        bindings[str(path.relative_to(u.ROOT))]=digest
        if path.suffix!='.npy':continue
        split=path.name.split('.')[-2];target=u.FRESH/'data'/split/'ratings.npy'
        a=np.load(path,mmap_mode='r');y=np.load(target,mmap_mode='r')
        if a.shape!=y.shape or a.dtype!=np.float32:raise ValueError('Original output shape/dtype mismatch')
        for lo in range(0,len(a),262144):
            if not np.isfinite(a[lo:lo+262144]).all():raise ValueError('Nonfinite native output')
        bindings[str(target.relative_to(u.ROOT))]=u.sha(target)
        arrays[split]=dict(rows=len(a),dtype=str(a.dtype),all_finite=True)
    if set(arrays)!={'trainx','probex','fulltrain','qual'}:raise ValueError('Original output coverage incomplete')
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets nonzero')
    for rel in ['src/bin/tsvdx5-new.rs','src/tx.rs','src/lib.rs']:
        path=u.FRESH/rel
        if u.sha(path)!=u.sha(u.SOURCE/rel):raise ValueError('Original model source differs')
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    result=dict(state='complete_original_temporal_native_outputs',job=name,source_pin=u.PIN,
        command=receipt['command'],epochs_per_phase=23,arrays=arrays,bindings=bindings,
        verifier_sha256=u.sha(Path(__file__)),model_training_performed=False,qualifying_labels_read=False,
        scope='Actual original execution receipt, full23/23 log trace, every output hash and all four finite float32 vector shapes checked against native dataset rows. No saved-model inference rerun or qualifying accuracy claim.')
    u.write(u.ROOT/'experiments/upstream-tsvdx5-60-completion/verification.json',result);print(result['state'],arrays,flush=True)

if __name__=='__main__':main()
