"""Owned retained-cache reference recovery and full actual-model replay; no training."""
import argparse
import numpy as np
import upstream_rbm400_reference_replay_execution as execution
import upstream_rbm400_latent_provider as provider
import upstream_rbm400_latent_cache_reference_v2 as cache
import upstream_rbm400_latent_context_math as math
import upstream_factored_rbm_terminal_metal as terminal
import upstream_factored_rbm_admission as admission
import upstream_factored_rbm_worker_metal as original_worker


def base(plan_sha):
    return dict(execution_plan_sha256=plan_sha,qualifying_labels_read=False,
                models_fitted=0,production_provider_admitted=False,
                observed=execution.stamp(),actual_MPS_predictions_replayed=True,
                all_native_prediction_bits_match=True)


def load_one(row,dtype,shape,guard):
    relative=str(execution.Path(row['path']).relative_to(execution.ROOT))
    records=execution.inputs.verify_files(execution.ROOT,{relative:row['sha256']},guard)
    return cache.mapped(execution.ROOT/relative,records[relative],dtype,shape,guard)


def aligned(raw,known_items):
    maps=math.alignment(known_items,raw['items'])
    users=np.full_like(raw['users'],np.nan)
    users[raw['covered']],items=math.align_qualifying(raw['users'][raw['covered']],raw['items'],maps)
    return dict(users=users,items=items,covered=raw['covered']),maps


def equal_bits(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def features_and_dots(context,raw,maps,query,guard):
    """Every real query's feature bits plus a measured casting-error bound.

    The bound separates the dual-map float64 residual, float32 cache casting,
    and float64 roundoff. No threshold is fitted to outcomes.
    """
    max_error=0.;max_bound=0.;rows=0
    a,b=(maps['item_map'],maps['user_map']) if maps else (np.eye(50),np.eye(50))
    dual_error=float(np.max(np.abs(b@a.T-np.eye(50))))
    for lo in range(0,len(query['users']),65536):
        guard();uid=query['users'][lo:lo+65536];iid=query['items'][lo:lo+65536]
        u=raw['users'][uid];i=raw['items'][iid]
        if maps:
            # Reuse the declared per-user/per-item cache arithmetic. A batched
            # row-wise transform could choose a different BLAS summation path.
            cu=context.users[uid];ci=context.items[iid]
        else:cu=u;ci=i
        reference=np.empty((len(uid),150),dtype='<f4')
        reference[:,:50]=cu;reference[:,50:100]=ci
        for f in range(50):reference[:,100+f]=cu[:,f]*ci[:,f]
        if not equal_bits(context.values(uid,iid),reference):
            raise ValueError('Every independently constructed query feature bit required')
        uf=u.astype(np.float64);it=i.astype(np.float64)
        ideal_u=uf@b;ideal_i=it@a
        stored_u=cu.astype(np.float64);stored_i=ci.astype(np.float64)
        before=np.zeros(len(uid));after=np.zeros(len(uid));cast=np.zeros(len(uid))
        for f in range(50):
            before+=uf[:,f]*it[:,f];after+=stored_u[:,f]*stored_i[:,f]
            cast+=np.abs(stored_u[:,f]-ideal_u[:,f])*np.abs(ideal_i[:,f])
            cast+=np.abs(stored_i[:,f]-ideal_i[:,f])*np.abs(ideal_u[:,f])
            cast+=np.abs(stored_u[:,f]-ideal_u[:,f])*np.abs(stored_i[:,f]-ideal_i[:,f])
        residual=dual_error*np.abs(uf).sum(axis=1)*np.abs(it).sum(axis=1)
        rounding=1e-12*(1+np.abs(ideal_u*ideal_i).sum(axis=1)+np.abs(uf*it).sum(axis=1))
        bound=cast+residual+rounding;error=np.abs(after-before)
        if not np.isfinite(bound).all() or np.any(error>bound):
            raise ValueError('Full query dual-map/cast dot-product invariance bound required')
        max_error=max(max_error,float(error.max()));max_bound=max(max_bound,float(bound.max()));rows+=len(uid)
    return dict(all_query_feature_bits_match=True,query_rows=rows,
                max_dot_product_error=max_error,max_dot_product_error_bound=max_bound,
                dual_map_max_identity_error=dual_error)


def verify(phase,plan_sha,guard):
    qreceipt=execution.read(execution.STUDY/'qualifying-extraction.json')
    execution.validate_result('qualifying',qreceipt,plan_sha)
    raw,bindings=provider.extract_phase(phase,guard)
    if (raw['metadata']['actual_MPS_predictions_replayed'] is not True
            or raw['metadata']['original_full_source_and_own16_lifecycle_verified'] is not True
            or raw['metadata']['all_native_queries_verified'] is not True):
        raise ValueError('Independent actual complete MPS extraction required')
    if bindings!=qreceipt['source_sha256']:
        raise ValueError('Original complete source closure must remain unchanged')
    row=qreceipt['manifest'];relative=str(execution.Path(row['path']).relative_to(execution.ROOT))
    with cache.load(execution.ROOT,relative,row['sha256'],guard) as (contexts,manifest):
        context=contexts[phase];maps=None
        if phase=='qualifying':
            value,maps=aligned(raw,contexts['known'].items)
            for name in ('item_map','user_map'):
                declared=manifest['arrays'][name]
                with load_one(dict(path=str(execution.ROOT/declared['path']),sha256=declared['sha256']),'<f8',(50,50),guard) as actual:
                    if not equal_bits(actual,maps[name]):raise ValueError('Every independently recomputed alignment-map bit required')
        else:value=raw
        for name,actual in [('users',context.users),('items',context.items),('covered',context.covered)]:
            if not equal_bits(actual,value[name]):raise ValueError('Every actual-model cache bit and coverage mask required')
        _,declaration,_=original_worker.data_receipt()
        _,paths=admission.phase_paths(declaration,phase)
        data=execution.read(provider.rbm.STUDY/'data-admission.json')
        selected={'users':paths['user_idxs'],'items':paths['item_idxs']}
        with admission.arrays(execution.ROOT,selected,guard,data['bound_file_records']) as query:
            feature_proof=features_and_dots(context,raw,maps,query,guard)
    return dict(base(plan_sha),state=execution.RESULTS['verify_'+phase][1],phase=phase,
                manifest_sha256=row['sha256'],all_actual_cache_bits_match=True,**feature_proof)


def adopt(plan_sha,guard):
    import os
    import upstream_rbm400_latent_execution as original
    expected=execution.read(execution.REPORT/'plan.json')['retained_metadata_sha256']
    execution.retained_metadata()
    copies={}
    for phase in ('known','qualifying'):
        path=original.STUDY/original.RESULTS[phase][0];relative=str(path.relative_to(execution.ROOT))
        records=execution.inputs.verify_files(execution.ROOT,{relative:expected[relative]},guard)
        with path.open('rb') as source:
            before=os.fstat(source.fileno())
            if admission.file_identity(before)!=records[relative]['identity']:raise ValueError('Original retained receipt descriptor required')
            def write(output):
                while True:
                    guard();block=source.read(1048576)
                    if not block:break
                    output.write(block)
            try:row=terminal.durable_payload(execution.STUDY/path.name,write,execution.ROOT,guard)
            finally:
                guard()
                if (admission.file_identity(before)!=admission.file_identity(os.fstat(source.fileno()))
                        or admission.file_identity(before)!=admission.file_identity(path.stat())):
                    raise ValueError('Original retained extraction receipt changed during adoption')
        if row['sha256']!=expected[relative]:raise ValueError('Every original retained receipt byte required')
        copies[phase]=row
    return dict(base(plan_sha),state=execution.RESULTS['adopt'][1],copies=copies,
        actual_MPS_predictions_replayed=False,all_native_prediction_bits_match=False,
        cache_arrays_rewritten=False,models_retrained=False,original_failure_preserved=True,
        remaining='Both complete actual saved-model/native/cache/query-feature replays required')


def assemble(plan_sha,guard):
    proofs={}
    for phase in ('known','qualifying','verify_known','verify_qualifying'):
        if phase in ('known','qualifying'):
            import upstream_rbm400_latent_execution as original
            name=original.RESULTS[phase][0]
        else:name=execution.RESULTS[phase][0]
        path=execution.STUDY/name;record=execution.read(path)
        execution.validate_result(phase,record,plan_sha)
        proofs[str(path.relative_to(execution.ROOT))]=execution.inputs.sha(path)
    receipt=execution.read(execution.STUDY/'qualifying-extraction.json');row=receipt['manifest']
    for phase in ('known','qualifying'):
        if execution.read(execution.STUDY/(phase+'-independent-verification.json'))['manifest_sha256']!=row['sha256']:
            raise ValueError('Both actual complete phase replays must bind the same unchanged retained cache')
    with cache.load(execution.ROOT,str(execution.Path(row['path']).relative_to(execution.ROOT)),row['sha256'],guard):pass
    return dict(base(plan_sha),state=execution.RESULTS['assemble'][1],manifest=row,proof_sha256=proofs,
        adoption_sha256=execution.inputs.sha(execution.STUDY/execution.RESULTS['adopt'][0]),
        actual_complete_models_reopened=2,independent_full_phase_replays=2,
        retained_cache_arrays_rewritten=False,original_failure_preserved=True,
        remaining='Versioned full context-data/neural/bag/own17 successors must consume this actual proof')


def main(phase):
    execution.authorize(phase);guard=execution.WorkerGuard(phase);guard()
    plan_sha=execution.inputs.sha(execution.REPORT/'plan.json')
    if phase=='adopt':value=adopt(plan_sha,guard)
    elif phase in ('verify_known','verify_qualifying'):value=verify(phase[7:],plan_sha,guard)
    elif phase=='assemble':value=assemble(plan_sha,guard)
    else:raise ValueError('Only retained metadata adoption and full original read-only replays declared')
    execution.validate_result(phase,value,plan_sha)
    execution.exclusive_json(execution.STUDY/execution.RESULTS[phase][0],value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=execution.PHASES,required=True)
    main(parser.parse_args().phase)
