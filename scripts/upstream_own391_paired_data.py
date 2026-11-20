#!/usr/bin/env python3
"""Matched paired-B391 scalar/context operators and known-only ridge control."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
from pathlib import Path
import numpy as np
import upstream_full_fwls_inputs as inventory
import upstream_own352_fold_bag as old

u,n=inventory.u,old.study.n
RUN=u.ROOT/'artifacts/upstream-own391-paired-data'
REPORT=u.ROOT/'experiments/upstream-own391-paired-data'
DECL=u.ROOT/'OWN391_PAIRED_NEURAL_PLAN.md'
ROWS,DESIGN,FEATURES=1128164,467,617
check,score=inventory.checks.verify_bindings,old.study.data.score


def sources():
    mp=inventory.REPORT/'manifest.json';vp=inventory.REPORT/'independent-verification.json'
    m=u.read(mp);v=u.read(vp)
    if (v['state']!='complete_independent_all413_original_input_streams_and_exact_complete_paired_aliases' or
        v['manifest_sha256']!=u.sha(mp) or not v['all_known_and_qualifying_operator_bits_equal'] or
        v['qualifying_labels_read'] is not False or v['models_fitted']!=0 or
        v['source_sha256']!=m['source_sha256'] or not set(range(352)).issubset(m['selected_indices'])):
        raise ValueError('Complete original native inventory/alias admission required')
    added=[m['entries'][i] for i in m['selected_indices'] if 352<=i<400]
    if len(added)!=39 or any(r['transform']!='clip0to6' for r in added):raise ValueError('Every eligible native/TX scalar addition required')
    paired=u.read(old.base.REPORT/'declaration.json');proof=u.read(old.base.REPORT/'independent-verification.json')
    temporal=u.read(old.study.data.REPORT/'manifest.json')['recent_rows']
    if (proof['state']!='complete_both_full_paired_own330_source_column_reconstructions' or
        proof['declaration_sha256']!=u.sha(old.base.REPORT/'declaration.json') or
        len(paired['rows'])!=223 or len(temporal)!=22):raise ValueError('Complete matched paired B parent/temporal sources required')
    bindings=dict(v['source_sha256'])
    def bind(path,expected=None):
        key=str(path.relative_to(u.ROOT));value=u.sha(path) if expected is None else expected
        if key in bindings and bindings[key]!=value:raise ValueError('Original paired source expectation conflicts')
        bindings[key]=value
    for key,value in paired['source_sha256'].items():bind(u.ROOT/key,value)
    for path in [Path(__file__),u.ROOT/'scripts/verify_upstream_own391_paired_data.py',DECL,mp,vp,
        old.base.REPORT/'declaration.json',old.base.REPORT/'transport.json',old.base.REPORT/'independent-verification.json']:
        bind(path)
    check(bindings)
    return bindings,paired['rows'],temporal,added


class Inputs:
    def __init__(self,query,original,temporal,added):
        if query not in old.base.COUNTS or len(original)!=223 or len(temporal)!=22 or len(added)!=39:
            raise ValueError('Complete original paired B391 query required')
        self.query=query;self.parent=old.base.Inputs(query,original);self.temporal=temporal;self.added=added

    def block(self,lo,hi):
        z,f=self.parent.block(lo,hi);columns=[]
        for row in self.temporal+self.added:
            reader=old.base.raw.Rows(u.ROOT/row['paired_files'][self.query],(old.base.COUNTS[self.query],),np.float32)
            try:columns.append(np.clip(reader.read(lo,hi),0,6))
            finally:reader.close()
        extra=np.column_stack(columns)
        result=(np.column_stack([z[:,:-1],extra,z[:,-1:]]),np.column_stack([f,extra-np.float32(3.6)]))
        if (result[0].shape!=(hi-lo,467) or result[1].shape!=(hi-lo,617) or result[0].dtype!=np.float64 or
            result[1].dtype!=np.float32 or not np.isfinite(result[0]).all() or not np.isfinite(result[1]).all()):
            raise ValueError('Every complete finite paired391 operator required')
        return result

    def close(self):self.parent.close()


def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing paired391 data study retained')
    bindings,original,temporal,added=sources();RUN.mkdir();REPORT.mkdir()
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,original_rows=original,temporal_rows=temporal,new_rows=added,
        declaration_sha256=u.sha(DECL),worker_sha256=u.sha(Path(__file__)),
        verifier_sha256=u.sha(u.ROOT/'scripts/verify_upstream_own391_paired_data.py'),qualifying_labels_read=False,
        input_realization='Matched paired production B on both complete known and qualifying query populations.'))
    for name in ['labels.npy','users.npy','groups.npy']:u.link_or_copy(old.study.RUN/name,RUN/name)
    labels=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy')
    gram=np.zeros((10,467,467));rhs=np.zeros((10,467));streams={}
    for query,count in old.base.COUNTS.items():
        hashes=[hashlib.sha256(),hashlib.sha256()];retained=[hashlib.sha256(),hashlib.sha256()]
        reader=Inputs(query,original,temporal,added)
        try:
            for lo in range(0,count,8192):
                hi=min(lo+8192,count);z,f=reader.block(lo,hi)
                hashes[0].update(z.tobytes());hashes[1].update(f.tobytes())
                retained[0].update(np.column_stack([z[:,:405],z[:,-1:]]).tobytes());retained[1].update(f[:,:556].tobytes())
                if query=='probex':
                    for group in range(10):
                        keep=groups[lo:hi]==group;a=z[keep]
                        gram[group]+=a.T@a;rhs[group]+=a.T@labels[lo:hi][keep]
            if [h.hexdigest() for h in retained]!=u.read(old.base.REPORT/'transport.json')['c_order_sha256'][query]:
                raise ValueError('Every original paired B330 parent operator bit must remain unchanged')
        finally:reader.close()
        streams[query]=[h.hexdigest() for h in hashes];print('Complete matched B391 input stream:',query,flush=True)
    selected=list(range(427))+[466]
    cg=gram[:,selected][:,:,selected];cr=rhs[:,selected]
    weights=np.stack([n.ridge_weights(gram,rhs,np.arange(10)//2!=fold,1000) for fold in range(5)])
    control=np.stack([n.ridge_weights(cg,cr,np.arange(10)//2!=fold,1000) for fold in range(5)])
    predictions=[np.empty(ROWS,dtype='<f4'),np.empty(ROWS,dtype='<f4')]
    reader=Inputs('probex',original,temporal,added)
    try:
        for lo in range(0,ROWS,8192):
            hi=min(lo+8192,ROWS);z,_=reader.block(lo,hi)
            for fold in range(5):
                held=groups[lo:hi]//2==fold
                predictions[0][lo:hi][held]=np.clip(z[held]@weights[fold],1.05,4.95).astype('<f4')
                predictions[1][lo:hi][held]=np.clip(z[held][:,selected]@control[fold],1.05,4.95).astype('<f4')
    finally:reader.close()
    np.savez(RUN/'stats.npz',gram=gram,rhs=rhs);np.save(RUN/'weights.npy',weights);np.save(RUN/'control352-weights.npy',control)
    predictions[0].tofile(RUN/'ridge.f32');predictions[1].tofile(RUN/'control352.f32');check(bindings)
    u.write(REPORT/'manifest.json',dict(state='complete_matched_paired391_operators_and_known_ridge_pending_independent_replay',
        source_sha256=bindings,plan_sha256=u.sha(REPORT/'plan.json'),original_rows=original,temporal_rows=temporal,new_rows=added,
        rows=ROWS,qualifying_rows=2817131,raw_predictions=391,context_columns=90,design_columns=467,nonlinear_columns=617,nn_inputs=618,
        matched_B_known_and_qualifying=True,complete_original_B330_operator_bits_preserved=True,
        virtual_c_order_sha256=streams,ridge_rmse=score(predictions[0],labels),control352_rmse=score(predictions[1],labels),
        ridge_fold_rmse=[score(predictions[0][groups//2==fold],labels[groups//2==fold]) for fold in range(5)],
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},qualifying_labels_read=False,neural_models_fitted=0,
        nonlinear_component_OOF_predictions_used_as_features=False))
    print('Matched B352/B391 known ridge:',score(predictions[1],labels),score(predictions[0],labels),flush=True)


if __name__=='__main__':main()
