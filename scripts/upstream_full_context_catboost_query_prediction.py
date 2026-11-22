"""Saved-model qualifying block arithmetic component; no fit or publication.

The future owner must first authenticate both complete known studies and the
known-only selected arm, then admit the original qualifying operator stream.
This component retains all five folds and every zero/positive seed slot.
"""
from contextlib import contextmanager,ExitStack
from pathlib import Path
import numpy as np
import upstream_full_context_catboost as backend
import upstream_full_context_catboost_consumer as consumer
import verify_upstream_full_context_catboost_fold as replay

SEEDS=(2026,7,13)


@contextmanager
def models(folds,width,guard):
    consumer.no_torch()
    if len(folds)!=5 or type(width) is not int or width not in backend.WIDTHS or not callable(guard):
        raise ValueError('Every original fold, raw feature column and owned whole-family guard required')
    with ExitStack() as stack:
        loaded=[];records=[]
        for fold,row in enumerate(folds):
            counts=row['chosen_iterations']
            if (row['fold']!=fold or row['seed_order']!=list(SEEDS) or row['seed_offset']!=fold
                    or len(counts)!=3 or any(type(c) is not int or not 0<=c<=2000 for c in counts)
                    or row['qualifying_labels_read'] is not False
                    or set(row['models'])!={str(s) for s,c in zip(SEEDS,counts) if c}):
                raise ValueError('Exact original five-fold positive models and zero seed slots required')
            values={}
            for seed,count in zip(SEEDS,counts):
                if not count:continue
                record=row['models'][str(seed)];path=Path(record['path']);signature=record['signature']
                expected=dict(backend.PARAMETERS,random_seed=seed+fold,iterations=count)
                if (record['seed']!=seed+fold or record['iterations']!=count or record['width']!=width
                        or signature['tree_count']!=count or len(signature['feature_names'])!=width
                        or any(signature['flat_parameters'].get(k)!=v for k,v in expected.items())):
                    raise ValueError('Exact original selected seed/count/configuration saved model required')
                replay.authenticate_model(path,record,guard)
                values[seed]=stack.enter_context(backend.actual_model(path,record,signature,width,guard));records.append((path,record))
            loaded.append(values)
        try:
            yield tuple(loaded)
        finally:
            for path,record in records:replay.authenticate_model(path,record,guard)
            consumer.no_torch();guard()


def predict(features,baselines,loaded,guard):
    consumer.no_torch()
    if (not callable(guard) or features.ndim!=2 or not 0<len(features)<=8192
            or features.shape[1] not in backend.WIDTHS or features.dtype!=np.dtype('<f4')
            or not features.flags.c_contiguous or baselines.shape!=(5,2,len(features))
            or baselines.dtype!=np.dtype('<f8') or len(loaded)!=5
            or not np.isfinite(features).all() or np.any(features[:,-1]!=0)
            or not np.isfinite(baselines).all() or np.any(baselines<1.05) or np.any(baselines>4.95)
            or any(not set(row).issubset(set(SEEDS)) for row in loaded)):
        raise ValueError('Bounded full original raw block, ten eligible baselines and all five model folds required')
    values=[];total=np.zeros(len(features),dtype='<f8')
    # A single block copy may replace only its final full-eligible baseline
    # feature. Original stream features/baselines remain unchanged.
    matrix=features.copy(order='C')
    for fold,row in enumerate(loaded):
        guard();matrix[:,-1]=baselines[fold,1].astype('<f4')
        correction=np.zeros(len(features),dtype='<f8')
        for seed in SEEDS:
            if seed in row:correction+=backend.predict(row[seed],matrix,guard)/3
        held=np.clip(baselines[fold,1]+correction,1.05,4.95).astype('<f4')
        values.append(held);total+=held.astype('<f8')
    bag=np.clip(total/5,1.05,4.95).astype('<f4');guard()
    return tuple(values),bag
