#!/usr/bin/env python3
"""Compare the completed independent phase to all31 original native outputs."""
import datetime as dt
from pathlib import Path
import numpy as np
import upstream_temporal_phase as phase
import upstream_remote_temporal_pair as remote

u=phase.u


def verify():
    paired=phase.verify_receipt();recovered=remote.verify_receipt()
    original=u.ROOT/'experiments/upstream-remote-12h/tsvdx5-150o/native-receipt.json'
    if u.sha(original)!=remote.RECEIPT_SHA:raise ValueError('Original own native receipt changed')
    native=u.read(original)
    expected={name:digest for name,digest in native['outputs'].items()
              if name.endswith('.qual.npy') or name.endswith('.fulltrain.npy')}
    tree=phase.mapped.TREE/phase.PRED
    if len(expected)!=31 or {p.name for p in tree.iterdir()}!={Path(name).name for name in expected}:
        raise ValueError('Exactly31 completed native phase outputs required')
    rows={split:len(np.load(u.FRESH/'data'/split/'ratings.npy',mmap_mode='r',allow_pickle=False))
          for split in ['probex','qual','fulltrain']}
    if rows['qual']!=2817131 or np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):
        raise ValueError('Qualifying targets must remain zero')
    records=[];bindings={}
    for relative,digest in expected.items():
        path=tree/Path(relative).name;actual=u.sha(path)
        if actual!=digest:raise ValueError('Local phase differs from original native output: '+relative)
        values=np.load(path,mmap_mode='r',allow_pickle=False);split=path.name.rsplit('.',2)[-2]
        if values.shape!=(rows[split],) or values.dtype!=np.dtype('<f4'):
            raise ValueError('Native output shape/dtype differs')
        for lo in range(0,len(values),262144):
            if not np.isfinite(values[lo:lo+262144]).all():raise ValueError('Nonfinite native output')
        bindings[str(path.relative_to(u.ROOT))]=digest
        records.append(dict(name=path.name,rows=len(values),sha256=digest,exact_original_native_hash=True))
    pairs=[]
    for name in phase.NAMES:
        for split in ['probex','qual']:
            left=u.ROOT/paired['pairs'][name][split];right=u.ROOT/recovered['pairs'][name][split]
            a=np.load(left,mmap_mode='r',allow_pickle=False);b=np.load(right,mmap_mode='r',allow_pickle=False)
            if a.shape!=b.shape or a.shape!=(rows[split],):raise ValueError('Recovered pair shape differs')
            for lo in range(0,len(a),262144):
                if not np.array_equal(a[lo:lo+262144],b[lo:lo+262144]):
                    raise ValueError('Independently completed pair differs from recovered own pair')
            pairs.append(dict(name=name,split=split,rows=len(a),values_equal=True))
    for path in [Path(__file__),original,phase.REPORT/'plan.json',phase.REPORT/'paired-receipt.json',
                 remote.REPORT/'paired-receipt.json']:
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    for relative,digest in bindings.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Completion evidence changed during scan')
    result=dict(state='complete',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_pin=u.PIN,original_native_receipt_sha256=remote.RECEIPT_SHA,
        completed_original_phase_epochs=25,records=records,selected_pairs=pairs,
        outputs_matching_original_native_hash=31,selected_pair_values_equal=8,
        bindings=bindings,qualifying_labels_read=False,qualifying_targets_zero=True,goal_complete=False,
        scope='Completed independent fulltrain/qual phase reproduces all31 native NPY outputs of our original remote job exactly, including the full training residual vector. The eight selected split vectors equal the already admitted own recovered pairs. No new qualifying score or full second two-phase pipeline is claimed.')
    u.write(phase.REPORT/'completion-verification.json',result)
    print({k:result[k] for k in ['state','outputs_matching_original_native_hash','selected_pair_values_equal','goal_complete']})


if __name__=='__main__':verify()
