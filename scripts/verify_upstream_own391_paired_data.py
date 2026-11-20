#!/usr/bin/env python3
"""Rebuild both complete paired-B391 streams, systems and original ridge fits."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_own391_paired_data as experiment

u=experiment.u


def block(parent,query,temporal,added,lo,hi):
    z,f=parent.block(lo,hi);extra=np.empty((hi-lo,61),dtype=np.float32)
    for column,row in enumerate(temporal+added):
        array=np.load(u.ROOT/row['paired_files'][query],mmap_mode='r',allow_pickle=False)
        if array.shape!=(experiment.old.base.COUNTS[query],) or array.dtype!=np.dtype('<f4'):raise ValueError('Complete original native scalar shape/dtype required')
        extra[:,column]=np.clip(array[lo:hi],0,6);del array
    zz=np.ones((hi-lo,467),dtype=np.float64);ff=np.empty((hi-lo,617),dtype=np.float32)
    zz[:,:405]=z[:,:405];zz[:,405:466]=extra;zz[:,-1]=z[:,-1]
    ff[:,:556]=f
    for column in range(61):ff[:,556+column]=extra[:,column]-np.float32(3.6)
    if not np.isfinite(zz).all() or not np.isfinite(ff).all():raise ValueError('Complete finite matched paired operators required')
    return zz,ff


def main():
    report,root=experiment.REPORT,experiment.RUN;output=report/'independent-verification.json'
    if output.exists():raise ValueError('Existing complete paired391 verification retained')
    p=u.read(report/'plan.json');m=u.read(report/'manifest.json')
    if (m['state']!='complete_matched_paired391_operators_and_known_ridge_pending_independent_replay' or
        m['plan_sha256']!=u.sha(report/'plan.json') or p['worker_sha256']!=u.sha(Path(experiment.__file__)) or
        p['verifier_sha256']!=u.sha(Path(__file__)) or p['declaration_sha256']!=u.sha(experiment.DECL) or
        m['qualifying_labels_read'] is not False or m['neural_models_fitted']!=0 or
        not m['matched_B_known_and_qualifying'] or m['nonlinear_component_OOF_predictions_used_as_features'] is not False):
        raise ValueError('Original complete matched-B391 declaration required')
    bindings,original,temporal,added=experiment.sources()
    if (bindings!=p['source_sha256'] or bindings!=m['source_sha256'] or original!=m['original_rows'] or
        temporal!=m['temporal_rows'] or added!=m['new_rows'] or original!=p['original_rows'] or
        temporal!=p['temporal_rows'] or added!=p['new_rows']):raise ValueError('Original full matched source list differs')
    for name,digest in m['prepared_sha256'].items():
        if u.sha(root/name)!=digest:raise ValueError('Actual complete data/statistic/prediction artifact changed')
    labels=np.load(root/'labels.npy');groups=np.load(root/'groups.npy');users=np.load(root/'users.npy')
    order=np.argsort(users,kind='stable')
    if labels.shape!=(1128164,) or set(np.unique(groups))!=set(range(10)) or np.any((users[order][1:]==users[order][:-1])&(groups[order][1:]//2!=groups[order][:-1]//2)):
        raise ValueError('Complete canonical known labels and held users required')
    gram=np.zeros((10,467,467));rhs=np.zeros((10,467));streams={}
    for query,count in experiment.old.base.COUNTS.items():
        parent=experiment.old.base.Inputs(query,original);digests=[hashlib.sha256(),hashlib.sha256()]
        retained=[hashlib.sha256(),hashlib.sha256()]
        try:
            for lo in range(0,count,8192):
                hi=min(lo+8192,count);z,f=block(parent,query,temporal,added,lo,hi)
                digests[0].update(z.tobytes());digests[1].update(f.tobytes())
                retained[0].update(np.column_stack([z[:,:405],z[:,-1:]]).tobytes());retained[1].update(f[:,:556].tobytes())
                if query=='probex':
                    for group in range(10):
                        keep=groups[lo:hi]==group;a=z[keep];gram[group]+=a.T@a;rhs[group]+=a.T@labels[lo:hi][keep]
            if [h.hexdigest() for h in retained]!=u.read(experiment.old.base.REPORT/'transport.json')['c_order_sha256'][query]:raise ValueError('Every complete paired B330 parent bit must match')
        finally:parent.close()
        streams[query]=[h.hexdigest() for h in digests]
        print('Independently reconstructed complete matched B391:',query,flush=True)
    saved=np.load(root/'stats.npz')
    if streams!=m['virtual_c_order_sha256'] or not np.array_equal(gram.view('u8'),saved['gram'].view('u8')) or not np.array_equal(rhs.view('u8'),saved['rhs'].view('u8')):
        raise ValueError('Every complete matched operator and statistic bit must match')
    selected=list(range(427))+[466];cg=gram[:,selected][:,:,selected];cr=rhs[:,selected]
    coefficients=[]
    for g,r,name in [(gram,rhs,'weights.npy'),(cg,cr,'control352-weights.npy')]:
        actual=np.load(root/name);expected=[];penalty=np.eye(r.shape[1])*1000.;penalty[-1,-1]=0
        for fold in range(5):
            fit=np.arange(10)//2!=fold;expected.append(np.linalg.solve(g[fit].sum(0)+penalty,r[fit].sum(0)))
        expected=np.stack(expected)
        if actual.shape!=expected.shape or not np.array_equal(actual.view('u8'),expected.view('u8')):raise ValueError('Every eligible ridge coefficient bit must match')
        coefficients.append(actual)
    arrays=[np.fromfile(root/name,dtype='<f4') for name in ['ridge.f32','control352.f32']]
    if any(a.shape!=(1128164,) or not np.isfinite(a).all() for a in arrays):raise ValueError('Complete finite held predictions required')
    parent=experiment.old.base.Inputs('probex',original)
    try:
        for lo in range(0,len(labels),8192):
            hi=min(lo+8192,len(labels));z,_=block(parent,'probex',temporal,added,lo,hi)
            for fold in range(5):
                held=groups[lo:hi]//2==fold
                for k,design in enumerate([z,z[:,selected]]):
                    expected=np.clip(design[held]@coefficients[k][fold],1.05,4.95).astype('<f4')
                    if not np.array_equal(expected.view('u4'),arrays[k][lo:hi][held].view('u4')):raise ValueError('Every actual held prediction bit must replay')
    finally:parent.close()
    scores=[math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(values.astype(np.float64),1.05,4.95),labels))/len(labels)) for values in arrays]
    if abs(scores[0]-m['ridge_rmse'])>1e-12 or abs(scores[1]-m['control352_rmse'])>1e-12:raise ValueError('Full compensated canonical known score differs')
    experiment.check(bindings)
    u.write(output,dict(state='complete_independent_matched_B391_all_operators_statistics_ridge_and_stored_predictions',
        observed=inventory_stamp(),manifest_sha256=u.sha(report/'manifest.json'),plan_sha256=u.sha(report/'plan.json'),
        verifier_sha256=u.sha(Path(__file__)),source_sha256=bindings,raw_predictions=391,design_columns=467,
        nonlinear_columns=617,nn_inputs=618,all_original_paired_B330_bits_preserved=True,matched_B_known_and_qualifying=True,
        every_full_operator_statistic_solve_and_prediction_bit_equal=True,ridge_rmse=scores[0],control352_rmse=scores[1],
        ridge_prediction_sha256=u.sha(root/'ridge.f32'),qualifying_labels_read=False,neural_models_fitted=0))
    print('Complete matched B352/B391 ridge:',scores[1],scores[0],flush=True)


def inventory_stamp():return experiment.inventory.checks.stamp()


if __name__=='__main__':main()
