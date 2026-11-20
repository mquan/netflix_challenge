#!/usr/bin/env python3
"""Complete original eligible-row normalization parity without unused-row buffers."""
import os
for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='2'
import gc
import hashlib
from pathlib import Path
import numpy as np
import upstream_own167_context218_features as data
import upstream_exact_neural_scaling as exact
import upstream_nonlinear as original

u=data.u
RUN=u.ROOT/'artifacts/upstream-own167-context218-normalization'
REPORT=u.ROOT/'experiments/upstream-own167-context218-normalization'

def digest(a):return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

def selected_inputs(mask,baseline):
    _,arrays=data.sources();out=np.empty((int(mask.sum()),522),dtype=np.float32);offset=0
    for lo in range(0,len(mask),8192):
        hi=min(lo+8192,len(mask));keep=mask[lo:hi];count=int(keep.sum())
        z,f=data.block(arrays,lo,hi)
        # Original conversion happens on the complete block before row selection.
        reference=np.column_stack([f,baseline[lo:hi]]).astype(np.float32)[keep]
        out[offset:offset+count,:-1]=f[keep];out[offset:offset+count,-1]=baseline[lo:hi][keep]
        if not np.array_equal(out[offset:offset+count].view('u4'),reference.view('u4')):
            raise ValueError('Every selected input bit must match original full-block conversion')
        offset+=count
    if offset!=len(out):raise ValueError('Incomplete ordered eligible rows')
    return out

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing complete parity evidence preserved')
    RUN.mkdir();REPORT.mkdir()
    proof=data.REPORT/'independent-verification.json';checked=u.read(proof)
    if checked['state']!='complete_independent_all_columns_and_original_group_statistics':raise ValueError('Original all-column/group-stat parity required')
    sources,arrays=data.sources();sources[str(proof.relative_to(u.ROOT))]=u.sha(proof)
    for p in [Path(__file__),Path(exact.__file__),Path(original.__file__),data.REPORT/'manifest.json']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    manifest=u.read(data.REPORT/'manifest.json')
    for name,h in manifest['prepared_sha256'].items():
        if u.sha(data.RUN/name)!=h:raise ValueError('Prepared identities/statistics differ')
    groups=np.load(data.RUN/'groups.npy');z=np.empty((len(groups),243),dtype=np.float64)
    old=arrays[0];raw=arrays[2]
    for lo in range(0,len(z),8192):
        hi=min(lo+8192,len(z))
        z[lo:hi,:241]=old[lo:hi,:241];z[lo:hi,241]=np.clip(raw[lo:hi],0,6);z[lo:hi,242]=old[lo:hi,241]
    if digest(z)!=manifest['virtual_c_order_sha256'][0]:raise ValueError('Complete original C-order ridge design differs')
    del arrays,old,raw;gc.collect()
    selected=np.arange(10)//2!=0
    with np.load(data.RUN/'stats.npz') as stats:
        inner=original.ridge_weights(stats['gram'],stats['rhs'],selected&(np.arange(10)%2==0),1000)
        full=original.ridge_weights(stats['gram'],stats['rhs'],selected,1000)
    baselines={'inner':np.clip(z@inner,1.05,4.95),'full':np.clip(z@full,1.05,4.95)}
    del z;gc.collect();records=[]
    for arm in ['inner','full']:
        mask=(groups//2!=0)&((groups%2==0) if arm=='inner' else True)
        x=selected_inputs(mask,baselines[arm]);input_hash=digest(x)
        lo,hi=np.percentile(x,[1,99],axis=0)
        clipped=np.asarray(np.clip(x,lo,hi),dtype=np.float64,order='C')
        del x;gc.collect()
        reference=dict(lo=lo,hi=hi,mean=clipped.mean(0),std=np.maximum(clipped.std(0),1e-6))
        del clipped;gc.collect()
        # Reconstruct the identical eligible input instead of retaining it during std.
        x=selected_inputs(mask,baselines[arm])
        if digest(x)!=input_hash:raise ValueError('Eligible source values changed')
        actual=exact.fit_scaler(x,np.ones(len(x),dtype=bool))
        for key in reference:
            if not np.array_equal(reference[key].view('u8'),actual[key].view('u8')):raise ValueError('Original complete eligible statistical bits differ: '+key)
        normalized=hashlib.sha256()
        for start in range(0,len(x),10007):
            block=x[start:start+10007]
            expected,_=original.scaled(block,None,scaler=reference)
            got=exact.apply_scaler(block,actual,block_rows=8192)
            if not np.array_equal(expected.view('u4'),got.view('u4')):raise ValueError('Actual522-input normalized float bits differ')
            normalized.update(got.tobytes(order='C'))
        np.savez(RUN/(arm+'-scaler.npz'),**actual)
        records.append(dict(arm=arm,rows=len(x),columns=522,ordered_eligible_input_sha256=input_hash,
            original_all_statistical_bits_equal=True,every_normalized_float32_bit_equal=True,
            normalized_c_order_sha256=normalized.hexdigest()))
        del x;gc.collect()
    for name,h in sources.items():
        if u.sha(u.ROOT/name)!=h:raise ValueError('Original source changed during parity')
    u.write(REPORT/'verification.json',dict(state='complete_actual522_input_eligible_statistics_and_normalization_parity',
        records=records,source_sha256=sources,fold=0,qualifying_labels_read=False,models_fitted=0,
        exact_original_row_order_and_full_eligible_statistic_dimensions_preserved=True,
        scope='Actual inner/full eligible matrices for outerfold0, all522 columns; complete original percentiles/mean/std and every normalized f32 bit. No row/rank/capacity reduction, no large persistent cache. Remaining folds follow the same frozen transform; actual model fitting/replay remains.'))
    print('Complete actual522-input original statistics and all normalized bits match',flush=True)

if __name__=='__main__':main()
