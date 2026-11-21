"""Read-only representation extraction core; no files, labels, fitting or CLI.

Actual production admission/ownership must be supplied by a separate caller.
CPU fixtures never admit a production provider. Original native predictions
must replay bitwise before any latent cache can be returned.
"""
from dataclasses import asdict
import os
import numpy as np
import upstream_factored_rbm_terminal_metal as terminal
from upstream_factored_rbm_training_metal import Config,SHARED
from upstream_factored_rbm_prediction_metal import predict
from upstream_rbm400_latent_context_math import item_contrast,FACTORS


def extract(record,reader,phase,admission_sha256,expected_config,expected_native,guard,device='mps'):
    if device not in ('cpu','mps') or not callable(guard):
        raise ValueError('Explicit test/actual device and external ownership guard required')
    if device=='mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0'):
        raise ValueError('Actual MPS extraction requires fallback/fast-math0')
    guard()
    import torch
    if device=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('Actual MPS unavailable; no CPU substitution')
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise ValueError('Strict deterministic MPS runtime required')
    config=terminal.validate(record,reader,phase,admission_sha256,expected_config)
    if asdict(config)!=asdict(Config()) or config.factors!=FACTORS:
        raise ValueError('Complete original400hidden50factor15epoch recipe required')
    expected=np.asarray(expected_native)
    if expected.dtype!=np.dtype('<f4') or expected.shape!=(len(reader.qr['user_idxs']),) or not np.isfinite(expected).all():
        raise ValueError('Complete finite original native float32 prediction vector required')
    guard()
    shared={k:record['model'][k].to(device) for k in SHARED}
    user_factors=np.full((reader.n_users,FACTORS),np.nan,dtype='<f4')
    covered=np.zeros(reader.n_users,dtype=bool)
    seen=np.zeros(len(expected),dtype=bool)
    required=np.flatnonzero(reader.qr['user_cnts'])
    items=item_contrast(record['model']['p'].numpy())
    guard()
    for lo in range(0,len(required),config.batch_users):
        guard();batch=reader.batch(required[lo:lo+config.batch_users]);model=dict(shared)
        for key,indices in [('bu',batch['global_users']),('but',batch['global_days'])]:
            index=torch.tensor(indices,dtype=torch.int64,device='cpu')
            model[key]=record['model'][key].index_select(0,index).to(device)
        support={k:torch.tensor(v,dtype=torch.int64,device=device) for k,v in batch['support'].items()}
        query={k:torch.tensor(v,dtype=torch.int64,device=device) for k,v in batch['query'].items()}
        result=predict(model,support,query,config.tile_rows)
        values=result['prediction'].cpu().numpy();rows=batch['query_rows'];users=batch['global_users']
        if (rows.ndim!=1 or rows.dtype.kind not in 'iu' or np.any(rows<0) or np.any(rows>=len(expected))
                or len(np.unique(rows))!=len(rows) or not np.array_equal(users,required[lo:lo+config.batch_users])):
            raise ValueError('Exact unique original query rows and canonical user batch required')
        if (len(values)!=len(rows) or np.any(seen[rows]) or not np.isfinite(values).all()
                or values.dtype!=np.dtype('<f4')
                or not np.array_equal(values.view('<u4'),expected[rows].view('<u4'))):
            raise ValueError('Every original native query prediction must replay bitwise')
        with torch.no_grad():factors=(result['hidden_expected']@shared['q'].T).cpu().numpy()
        if (factors.dtype!=np.float32 or factors.shape!=(len(users),FACTORS)
                or np.any(covered[users]) or not np.isfinite(factors).all()):
            raise ValueError('Every finite original query-user posterior required once')
        user_factors[users]=factors;covered[users]=True;seen[rows]=True;guard()
    if not seen.all() or not np.array_equal(covered,reader.qr['user_cnts']>0):
        raise ValueError('Complete exact canonical query/user coverage required')
    if device=='mps':torch.mps.synchronize()
    guard()
    return dict(users=user_factors,items=items,covered=covered,
                metadata=dict(phase=phase,device=device,config=asdict(config),admission_sha256=admission_sha256,
                    users=reader.n_users,items=reader.n_items,native_query_rows=len(expected),
                    covered_users=int(covered.sum()),cold_query_users=int(np.count_nonzero(covered&(reader.positive_counts==0))),
                    raw_prediction_bits_verified=True,all_native_queries_verified=True,
                    exact_query_user_coverage=True,actual_MPS_predictions_replayed=device=='mps',
                    production_provider_admitted=False,qualifying_labels_read=False,models_fitted=0))
