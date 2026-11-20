#!/usr/bin/env python3
"""Complete own330 ridge using bounded original NPY row reads, not a full cache."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='2'
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_repro as u
import upstream_nonlinear as n

RUN=u.ROOT/'artifacts/upstream-own330-native149-data'
REPORT=u.ROOT/'experiments/upstream-own330-native149-data'
CONTROL=u.ROOT/'experiments/upstream-own181-recent15-data'
INVENTORY=u.ROOT/'experiments/upstream-unused-native-pairs/verification.json'
ROWS=1128164
DESIGN=406


def check(bindings):
    for rel,digest in bindings.items():
        if u.sha(u.ROOT/rel)!=digest:
            raise ValueError('Original complete own source changed: '+rel)


def sources():
    old=u.read(CONTROL/'manifest.json');proof=u.read(CONTROL/'independent-verification.json')
    extra=u.read(INVENTORY)
    if (proof['state']!='complete_all_own181_retained_added_operator_bits_original_statistics_and_ridge_replay' or
        proof['manifest_sha256']!=u.sha(CONTROL/'manifest.json') or old['design_columns']!=257):
        raise ValueError('Complete own181 control required')
    if extra['state']!='complete_all_unused_local_original_native_pairs_and_query_payloads' or len(extra['rows'])!=149:
        raise ValueError('Fixed complete149 native input admission required')
    bindings=dict(old['source_sha256'])
    for rel,digest in extra['source_sha256'].items():
        if rel in bindings and bindings[rel]!=digest:
            raise ValueError('Conflicting original input bindings')
        bindings[rel]=digest
    for p in [Path(__file__),Path(n.__file__),CONTROL/'manifest.json',CONTROL/'independent-verification.json',
              CONTROL/'canonical-novelty-verification.json',INVENTORY,
              u.ROOT/'OWN330_NATIVE149_PLAN.md',u.ROOT/'scripts/verify_upstream_own330_native149_data.py']:
        bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in old['prepared_sha256'].items():
        path=u.ROOT/'artifacts/upstream-own181-recent15-data'/name
        bindings[str(path.relative_to(u.ROOT))]=digest
    return bindings,old['recent_rows'],extra['rows']


class Rows:
    """Regular-file window: full original values with a bounded resident set."""
    def __init__(self,path,shape,dtype):
        self.path=Path(path);self.stream=self.path.open('rb')
        try:
            version=np.lib.format.read_magic(self.stream)
            if version!=(1,0):
                raise ValueError('Explicit original NPY1.0 header required')
            actual,fortran,actual_dtype=np.lib.format.read_array_header_1_0(self.stream)
            if actual!=shape or fortran or actual_dtype!=np.dtype(dtype):
                raise ValueError('Exact original row shape/dtype/order required: '+str(path))
            self.offset=self.stream.tell();self.shape=shape;self.dtype=actual_dtype
            self.width=int(np.prod(shape[1:])) if len(shape)>1 else 1
            expected=self.offset+int(np.prod(shape))*self.dtype.itemsize
            if self.path.stat().st_size!=expected:
                raise ValueError('Complete NPY payload length differs')
        except BaseException:
            self.stream.close();raise

    def read(self,lo,hi):
        if not 0<=lo<=hi<=self.shape[0]:
            raise ValueError('Original full row bounds required')
        self.stream.seek(self.offset+lo*self.width*self.dtype.itemsize)
        value=np.fromfile(self.stream,dtype=self.dtype,count=(hi-lo)*self.width)
        if value.size!=(hi-lo)*self.width:
            raise ValueError('Short original row read')
        return value.reshape((hi-lo,)+self.shape[1:])

    def close(self):self.stream.close()


class Inputs:
    def __init__(self,recent,extra):
        self.old=Rows(u.ROOT/'artifacts/upstream-warm166/design.npy',(ROWS,242),np.float64)
        self.raw=[]
        try:
            self.raw=[Rows(u.ROOT/r['paired_files']['probex'],(ROWS,),np.float32) for r in recent+extra]
        except BaseException:
            self.close();raise
        if len(self.raw)!=164:
            self.close();raise ValueError('All fifteen plus149 inputs required')

    def block(self,lo,hi):
        old=self.old.read(lo,hi)
        added=np.column_stack([np.clip(a.read(lo,hi),0,6) for a in self.raw])
        z=np.column_stack([old[:,:-1],added,old[:,-1:]])
        control=np.column_stack([old[:,:-1],added[:,:15],old[:,-1:]])
        if z.shape!=(hi-lo,DESIGN) or control.shape!=(hi-lo,257) or not np.isfinite(z).all():
            raise ValueError('Full finite own330 design required')
        return z,control,added[:,15:]

    def close(self):
        self.old.close()
        for value in self.raw:value.close()


def score(prediction,labels):
    error=np.clip(np.asarray(prediction,dtype=np.float64),1.05,4.95)-labels
    return math.sqrt(math.fsum(float(v)*float(v) for v in error)/len(labels))


def main():
    if RUN.exists() or REPORT.exists():
        raise ValueError('Existing complete or partial own330 evidence preserved')
    bindings,recent,extra=sources();check(bindings)
    RUN.mkdir();REPORT.mkdir()
    for key in ['labels','users','groups']:
        u.link_or_copy(u.ROOT/'artifacts/upstream-own181-recent15-data'/(key+'.npy'),RUN/(key+'.npy'))
    labels=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy')
    grams=np.zeros((10,DESIGN,DESIGN));rhs=np.zeros((10,DESIGN))
    full_hash=hashlib.sha256();control_hash=hashlib.sha256();new_hash=[hashlib.sha256() for _ in extra]
    inputs=Inputs(recent,extra)
    try:
        for lo in range(0,ROWS,8192):
            hi=min(lo+8192,ROWS);z,control,added=inputs.block(lo,hi)
            full_hash.update(z.tobytes(order='C'));control_hash.update(control.tobytes(order='C'))
            for j,h in enumerate(new_hash):h.update(added[:,j].tobytes(order='C'))
            for group in range(10):
                keep=groups[lo:hi]==group;a=z[keep]
                grams[group]+=a.T@a;rhs[group]+=a.T@labels[lo:hi][keep]
        if control_hash.hexdigest()!=u.read(CONTROL/'manifest.json')['virtual_c_order_sha256'][0]:
            raise ValueError('Every retained original181 design bit must match')
        if [h.hexdigest() for h in new_hash]!=[r['canonical_clipped_payload_sha256'] for r in extra]:
            raise ValueError('Every complete original149 scalar payload must match')
        weights=np.stack([n.ridge_weights(grams,rhs,np.arange(10)//2!=fold,1000) for fold in range(5)])
        prediction=np.empty(ROWS,dtype='<f4')
        for lo in range(0,ROWS,8192):
            hi=min(lo+8192,ROWS);z,_,_=inputs.block(lo,hi)
            for fold in range(5):
                keep=groups[lo:hi]//2==fold
                prediction[lo:hi][keep]=np.clip(z[keep]@weights[fold],1.05,4.95)
    finally:inputs.close()
    np.savez(RUN/'stats.npz',gram=grams,rhs=rhs);np.save(RUN/'weights.npy',weights)
    prediction.tofile(RUN/'ridge.f32');check(bindings)
    u.write(REPORT/'manifest.json',dict(state='complete_original330_input_bits_group_statistics_and_ridge_diagnostic',
            source_sha256=bindings,rows=ROWS,raw_predictions=330,design_columns=DESIGN,
            added_rows=extra,control_rows=recent,full_design_c_order_sha256=full_hash.hexdigest(),
            original181_control_c_order_sha256=control_hash.hexdigest(),
            complete_added_payload_sha256=[h.hexdigest() for h in new_hash],
            prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},ridge_rmse=score(prediction,labels),
            original181_ridge_rmse=u.read(CONTROL/'independent-verification.json')['ridge_rmse'],
            ridge_fold_rmse=[score(prediction[groups//2==fold],labels[groups//2==fold]) for fold in range(5)],
            qualifying_labels_read=False,neural_models_fitted=0,
            scope='Complete own181 ridge retained plus all149 fixed unused own predictors. Bounded regular-file transport, full original operators and user-fold ridge only. Independent replay still required; no neural/global or qualifying promotion.'))
    print('Own330 full canonical ridge:',score(prediction,labels),'independent replay required',flush=True)


if __name__=='__main__':main()
