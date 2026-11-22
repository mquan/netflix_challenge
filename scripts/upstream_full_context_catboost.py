"""CPU-only high-capacity tree backend; no owner, data admission or CLI.

The future owned full-context study must establish masks, complete operators,
eligible baselines, provenance and resources before calling these primitives.
Importing this module never imports CatBoost/Torch or starts training.
"""
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import numpy as np

VERSION='1.2.10'
MAX_ROUNDS=2000
PATIENCE=100
WIDTHS=(713,714,863,864)
PARAMETERS=dict(loss_function='RMSE',eval_metric='RMSE',depth=8,learning_rate=.03,
    l2_leaf_reg=10.,border_count=64,bootstrap_type='Bernoulli',subsample=.8,rsm=.9,
    thread_count=2,task_type='CPU',grow_policy='SymmetricTree',allow_writing_files=False,verbose=False)


def runtime():
    if 'torch' in sys.modules:raise ValueError('Tree process must remain isolated from Torch')
    if importlib.metadata.version('catboost')!=VERSION:raise ValueError('Pinned CatBoost CPU runtime required')
    from catboost import CatBoostRegressor
    return CatBoostRegressor


class GuardCallback:
    def __init__(self,guard):
        if not callable(guard):raise ValueError('External same-frame owned resource guard required')
        self.guard=guard

    def after_iteration(self,info):
        self.guard()
        return True  # Resource limits suspend the process, never shorten training.


def arrays(features,targets,guard):
    if (not callable(guard) or features.ndim!=2 or features.shape[1] not in WIDTHS
            or features.dtype!=np.float32 or not features.flags.c_contiguous or not len(features)
            or targets.shape!=(len(features),) or targets.dtype!=np.float64):
        raise ValueError('Complete contiguous declared-width float32 inputs and float64 residuals required')
    for lo in range(0,len(features),8192):
        guard();hi=min(lo+8192,len(features))
        if not np.isfinite(features[lo:hi]).all() or not np.isfinite(targets[lo:hi]).all():
            raise ValueError('Every original feature and known residual must be finite')


def train(features,targets,seed,iterations,guard,validation=None):
    arrays(features,targets,guard)
    if type(seed) is not int or not 0<=seed<2**31 or type(iterations) is not int or not 1<=iterations<=MAX_ROUNDS:
        raise ValueError('Original positive selected budget and explicit seed required')
    if validation is not None:
        if iterations!=MAX_ROUNDS:raise ValueError('Inner selection must retain the full declared maximum')
        arrays(*validation,guard)
        if validation[0].shape[1]!=features.shape[1]:raise ValueError('Exact same inner/validation raw schema required')
    model=runtime()(**PARAMETERS,random_seed=seed,iterations=iterations)
    kwargs=dict(callbacks=[GuardCallback(guard)],save_snapshot=False)
    if validation is not None:kwargs.update(eval_set=validation,early_stopping_rounds=PATIENCE,use_best_model=True)
    else:kwargs['use_best_model']=False
    guard();model.fit(features,targets,**kwargs);guard()
    if validation is None and model.tree_count_!=iterations:raise ValueError('Every original selected full-refit tree required')
    return model


def selected_iterations(model,validation_residuals):
    if (validation_residuals.ndim!=1 or validation_residuals.dtype!=np.float64
            or not len(validation_residuals) or not np.isfinite(validation_residuals).all()):
        raise ValueError('Complete finite original inner validation residuals required')
    count=int(model.get_best_iteration())+1
    best=float(model.get_best_score()['validation']['RMSE'])
    baseline=float(np.mean(validation_residuals**2))
    if not 1<=count<=MAX_ROUNDS or not np.isfinite(best) or best<0 or model.tree_count_!=count:
        raise ValueError('Complete original selected inner tree inventory and metric required')
    chosen=count if best*best<baseline else 0
    return chosen,dict(best_iteration_zero_based=count-1,selected_iterations=chosen,
        best_validation_RMSE=best,zero_correction_validation_MSE=baseline,
        max_rounds=MAX_ROUNDS,patience=PATIENCE,qualifying_labels_read=False)


def model_signature(model):
    """Exact persisted configuration/schema; no float-comparison tolerance.

    In1.2.10 get_all_params rounds some floats after reload and n_features_in_
    becomes zero. The persisted flat parameters and feature names remain exact.
    """
    return dict(flat_parameters=json.loads(model.get_metadata()['params'])['flat_params'],
                feature_names=list(model.feature_names_),tree_count=int(model.tree_count_))


def predict(model,features,guard):
    if (features.ndim!=2 or features.shape[1] not in WIDTHS or features.dtype!=np.float32
            or not features.flags.c_contiguous or features.shape[1]!=len(model.feature_names_)):
        raise ValueError('Every original saved-model input column required')
    output=np.empty(len(features),dtype=np.float64)
    for lo in range(0,len(features),8192):
        guard();hi=min(lo+8192,len(features))
        if not np.isfinite(features[lo:hi]).all():raise ValueError('Finite full prediction features required')
        values=model.predict(features[lo:hi],thread_count=2)
        if values.shape!=(hi-lo,) or values.dtype!=np.float64 or not np.isfinite(values).all():
            raise ValueError('Every original float64 tree prediction required')
        output[lo:hi]=values
    guard();return output


@contextmanager
def actual_model(path,record,expected_signature,width,guard):
    """Load authenticated model bytes; no fitting or snapshot-based restart."""
    from upstream_factored_rbm_admission import file_identity
    path=Path(path);guard()
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if path.resolve()!=path or file_identity(before)!=record['identity']:
            raise ValueError('Exact authenticated original saved CPU tree required')
        # /dev/fd keeps the public filename loader attached to the authenticated
        # open descriptor without reopening the replaceable original pathname.
        try:
            model=runtime()();model.load_model(fname='/dev/fd/%s'%stream.fileno(),format='cbm')
            if model_signature(model)!=expected_signature or len(model.feature_names_)!=width:
                raise ValueError('Exact original saved tree configuration, width and selected count required')
            guard();yield model
        finally:
            guard()
            if file_identity(before)!=file_identity(os.fstat(stream.fileno())) or file_identity(before)!=file_identity(path.stat()):
                raise ValueError('Original CPU tree model changed during replay')
