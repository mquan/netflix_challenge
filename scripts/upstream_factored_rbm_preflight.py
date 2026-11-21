"""Guarded device-math preflight; no production data, fitting or CLI.

CPU-only or reduced checks never qualify the production device. Real MPS must
be explicitly configured without fallback, and the full stress case must pass.
"""
import os
import resource
import sys
import time
import numpy as np
import upstream_factored_rbm_reference as reference
import upstream_factored_rbm_torch as kernel
from upstream_factored_rbm_prediction import predict

SMALL_ATOL=2e-5
SMALL_RTOL=2e-5
STRESS_ATOL=5e-4
STRESS_RTOL=5e-5
REQUIRED_STRESS=dict(users=500,items=17770,hidden=400,factors=50,ordinary_rows=200,
                     training_rows=117570,conditional_rows=118069,query_rows=1000,max_user_rows=17770)


def fixture():
    n_users,n_items,hidden,factors=3,5,4,3
    rng=np.random.default_rng(311)
    shapes=dict(p=(n_items,5,factors),q=(factors,hidden),d=(n_items,hidden),bv=(n_items,5),bh=(hidden,),bu=(n_users,5),but=(5,5))
    model={k:rng.normal(0,.1,size=s).astype(np.float32) for k,s in shapes.items()}
    batch=dict(items=np.array([0,2,1,2,3,0]),users=np.array([0,0,1,1,2,2]),ratings=np.array([0,4,2,1,3,0]),
        days=np.array([0,1,2,2,3,4]),conditional_items=np.array([0,2,4,1,2,0,3]),
        conditional_users=np.array([0,0,0,1,1,2,2]),day_owner=np.array([0,0,1,2,2]))
    return model,batch,rng.random((3,n_users,hidden)).astype(np.float32),rng.random((3,6)).astype(np.float32)


def stress_fixture(n_users=500,n_items=17770,hidden=400,factors=50,ordinary_rows=200):
    if n_users<2 or n_items<ordinary_rows+6 or ordinary_rows<2:raise ValueError('Nonempty heavy-user stress dimensions required')
    rng=np.random.default_rng(140215);counts=np.r_[n_items,np.full(n_users-1,ordinary_rows)]
    item=[];user=[];rating=[];days=[];ci=[];cu=[];qi=[];qu=[];qd=[]
    for u,count in enumerate(counts):
        base=u*53%n_items;i=(base+np.arange(count))%n_items;extra=(base+ordinary_rows+5)%n_items
        item.append(i);user.append(np.full(count,u));rating.append((np.arange(count)+u)%5)
        days.append(u*2+(np.arange(count)>=count//2));condition=np.unique(np.r_[i,extra])
        ci.append(condition);cu.append(np.full(len(condition),u));qi.extend([i[0],extra]);qu.extend([u,u]);qd.extend([u*2,-1])
    cat=lambda xs:np.concatenate(xs).astype(np.int64)
    batch=dict(items=cat(item),users=cat(user),ratings=cat(rating),days=cat(days),
        conditional_items=cat(ci),conditional_users=cat(cu),day_owner=np.repeat(np.arange(n_users,dtype=np.int64),2))
    model=dict(p=rng.normal(0,.01,(n_items,5,factors)).astype(np.float32),q=rng.normal(0,.01/np.sqrt(factors),(factors,hidden)).astype(np.float32),
        d=rng.normal(0,.01,(n_items,hidden)).astype(np.float32),bv=np.zeros((n_items,5),dtype=np.float32),bh=np.zeros(hidden,dtype=np.float32),
        bu=np.zeros((n_users,5),dtype=np.float32),but=np.zeros((n_users*2,5),dtype=np.float32))
    query=dict(items=np.asarray(qi,dtype=np.int64),users=np.asarray(qu,dtype=np.int64),days=np.asarray(qd,dtype=np.int64),day_owner=batch['day_owner'])
    # Valid, unambiguous fixed draws stress all large indexed reductions without
    # near-CDF-boundary random divergence between f32 and f64 implementations.
    hu=np.zeros((3,n_users,hidden),dtype=np.float32);vu=np.zeros((3,len(batch['items'])),dtype=np.float32)
    observed=dict(users=n_users,items=n_items,hidden=hidden,factors=factors,ordinary_rows=ordinary_rows,
        training_rows=len(batch['items']),conditional_rows=len(batch['conditional_items']),query_rows=len(query['items']),max_user_rows=int(counts.max()))
    return model,batch,hu,vu,query,observed


def reference_prediction(model,hidden,query):
    qi,qu,day=[query[k] for k in ('items','users','days')]
    bias=model['bu'][qu].copy();known=day>=0;bias[known]+=model['but'][day[known]]
    logits=model['bv'][qi]+bias+np.einsum('lkf,lf->lk',model['p'][qi],(hidden@model['q'].T)[qu])
    ex=np.exp(logits-logits.max(axis=1,keepdims=True));prob=ex/ex.sum(axis=1,keepdims=True)
    return (prob*np.arange(1,6)).sum(axis=1)


def check_case(model,batch,hu,vu,guard,device='cpu',query=None,cold_user=False,atol=SMALL_ATOL,rtol=SMALL_RTOL,tile_rows=16384):
    import torch
    guard();started=time.perf_counter();cpu={k:v.astype(np.float64) for k,v in model.items()}
    expected=reference.batch_gradients(cpu,batch,hu.astype(np.float64),vu.astype(np.float64),tile_rows)
    guard();tm={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in model.items()}
    tb={k:torch.tensor(v,dtype=torch.int64,device=device) for k,v in batch.items()}
    got=kernel.batch_gradients(tm,tb,torch.tensor(hu,device=device),torch.tensor(vu,device=device),tile_rows)
    errors={}
    for key,value in expected.items():
        if key=='batch_users':
            if got[key]!=value:raise ValueError('Complete nonempty-user count differs')
        else:
            observed=got[key].cpu().numpy()
            if key=='visible_state':np.testing.assert_array_equal(observed,value)
            else:np.testing.assert_allclose(observed,value,atol=atol,rtol=rtol)
            errors[key]=float(np.max(np.abs(observed-value)))
    velocities={k:torch.zeros_like(v) for k,v in tm.items()}
    values,speeds=kernel.update(tm,velocities,got)
    ev,es=reference.update(cpu,{k:np.zeros_like(v) for k,v in cpu.items()},expected)
    for key in cpu:
        np.testing.assert_allclose(values[key].cpu().numpy(),ev[key],atol=atol,rtol=rtol)
        np.testing.assert_allclose(speeds[key].cpu().numpy(),es[key],atol=atol,rtol=rtol)
    if query is None:
        users=len(cpu['bu']);owner=batch['day_owner'];days=[]
        for u in range(users):days.extend([int(np.flatnonzero(owner==u)[0]),-1])
        query=dict(items=np.arange(users*2,dtype=np.int64)%len(cpu['p']),users=np.repeat(np.arange(users,dtype=np.int64),2),
                   days=np.array(days,dtype=np.int64),day_owner=owner)
    hidden=expected['positive_expected'];support=dict(batch);pm=dict(cpu)
    if cold_user:
        u=len(pm['bu']);pm['bu']=np.vstack((pm['bu'],np.zeros((1,5))))
        support['conditional_items']=np.r_[support['conditional_items'],1];support['conditional_users']=np.r_[support['conditional_users'],u]
        hidden=np.vstack((hidden,reference.expected(pm['bh']+pm['d'][1])))
        query=dict(query,items=np.r_[query['items'],1,1],users=np.r_[query['users'],u,u],days=np.r_[query['days'],-1,-1])
    wanted=reference_prediction(pm,hidden,query)
    models={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in pm.items()}
    coords={k:torch.tensor(v,dtype=torch.int64,device=device) for k,v in support.items()}
    queries={k:torch.tensor(v,dtype=torch.int64,device=device) for k,v in query.items()}
    repeats=[]
    for _ in range(3):
        guard();observed=predict(models,coords,queries,tile_rows)['prediction'].cpu().numpy()
        np.testing.assert_allclose(observed,wanted,atol=atol,rtol=rtol);repeats.append(observed.copy())
    for value in repeats[1:]:np.testing.assert_array_equal(value,repeats[0])
    guard();return dict(gradient_max_absolute_errors=errors,prediction_max_absolute_error=float(np.max(np.abs(repeats[0]-wanted))),
        same_device_prediction_repeats_bitwise_equal=True,query_rows=len(wanted),cold_user_checked=cold_user,
        atol=atol,rtol=rtol,elapsed_seconds=time.perf_counter()-started)


def run(guard,device='mps',stress=True):
    if device not in ('cpu','mps') or type(stress) is not bool or not callable(guard):raise ValueError('Explicit device/scope/resource guard required')
    if device=='mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0'):
        raise ValueError('Explicit MPS fallback/fast-math disable before runtime import required')
    if device=='mps' and 'torch' in sys.modules:
        raise ValueError('Fresh MPS worker required; Torch was already imported before startup verification')
    guard()
    import torch
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Actual MPS device unavailable; no CPU substitution')
    old=torch.are_deterministic_algorithms_enabled();old_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True,warn_only=False)
    try:
        model,batch,hu,vu=fixture();small=check_case(model,batch,hu,vu,guard,device,cold_user=True,tile_rows=2)
        results=dict(small=small)
        if stress:
            guard();model,batch,hu,vu,query,observed=stress_fixture()
            if observed!=REQUIRED_STRESS:raise ValueError('Full predeclared production-width stress case required')
            large=check_case(model,batch,hu,vu,guard,device,query=query,atol=STRESS_ATOL,rtol=STRESS_RTOL)
            results['stress']=dict(large,shape=observed)
        memory=None
        if device=='mps':
            torch.mps.synchronize();memory=dict(current_allocated_bytes=torch.mps.current_allocated_memory(),driver_allocated_bytes=torch.mps.driver_allocated_memory())
        guard();return dict(state='complete_device_math_checks_only',device=device,actual_MPS_tested=device=='mps',
            full_required_stress_tested=stress,passed_for_production_device=device=='mps' and stress,
            deterministic_algorithms_during_checks=True,fresh_mps_runtime_import=device=='mps',cases=results,mps_memory=memory,
            torch_version=torch.__version__,numpy_version=np.__version__,production_data_read=False,qualifying_labels_read=False,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
            limits='Device preflight only; actual full-data/profile/ownership and15+15 model/query replay remain required.')
    finally:torch.use_deterministic_algorithms(old,warn_only=old_warn)
