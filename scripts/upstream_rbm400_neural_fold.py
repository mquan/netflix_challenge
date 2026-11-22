"""Original-budget fold scheduling with one full representation matrix at a time.

All three independent inner selections precede their fresh outer refits. Each
backend call resets the original seed; no model, optimizer, RNG or budget is
shared between calls. Production MPS/ownership and durable model callbacks
must be supplied by a separately reviewed worker. This module has no CLI.
"""
import hashlib
import numpy as np
import upstream_rbm400_latent_neural as architecture
import upstream_rbm400_neural_scaler as scaling


def masks(users,groups,fold):
    if (type(fold) is not int or not 0<=fold<5 or users.ndim!=1 or groups.shape!=users.shape
            or users.dtype.kind not in 'iu' or groups.dtype!=np.int8 or np.any(users<0)
            or set(np.unique(groups))!=set(range(10))):
        raise ValueError('Complete original ten user groups and a declared outer fold required')
    order=np.argsort(users,kind='stable')
    if np.any((users[order][1:]==users[order][:-1])&(groups[order][1:]!=groups[order][:-1])):
        raise ValueError('Every user must retain one original inner/outer group')
    train=groups//2!=fold;held=~train;fit=train&(groups%2==0);valid=train&(groups%2==1)
    if not all(v.any() for v in (held,fit,valid)):raise ValueError('Every original nonempty fold partition required')
    return dict(train=train,held=held,fit=fit,valid=valid)


def digest(values,guard):
    h=hashlib.sha256()
    for start in range(0,len(values),8192):guard();h.update(values[start:start+8192].tobytes())
    return h.hexdigest()


def run(arm,fold,labels,users,groups,inner_weights,full_weights,*,build,train_nn,predict,
        save_preprocessing,save_model,release,guard):
    if arm not in architecture.CONFIGS:raise ValueError('Exact declared scalar/latent H128 arm required')
    if (labels.shape!=users.shape or labels.dtype!=np.float64 or not np.isin(labels,[1,2,3,4,5]).all()
            or any(not callable(f) for f in (build,train_nn,predict,save_preprocessing,save_model,release,guard))):
        raise ValueError('Complete original known outcomes and external owned callbacks required')
    cfg=architecture.CONFIGS[arm];partition=masks(users,groups,fold)
    if (inner_weights.shape not in ((562,),(563,)) or full_weights.shape!=inner_weights.shape
            or inner_weights.dtype!=np.float64 or full_weights.dtype!=np.float64
            or not np.isfinite(inner_weights).all() or not np.isfinite(full_weights).all()):
        raise ValueError('Complete eligible original scalar-ridge coefficient vectors required')
    expected_width=architecture.ops.dimensions(len(inner_weights)==563,arm=='latent')['nn_inputs']
    counts=[];preprocessing={}

    def prepare(stage,weights,eligible):
        guard();features,baseline,proof=build(stage,weights)
        if (features.shape!=(len(labels),expected_width) or baseline.shape!=labels.shape or baseline.dtype!=np.float64
                or not np.isfinite(baseline).all() or not features.flags.c_contiguous
                or features.dtype!=np.float32 or features.shape[1] not in scaling.WIDTHS
                or not np.array_equal(features[:,-1].view('<u4'),baseline.astype('<f4').view('<u4'))
                or proof['raw_neural_input_sha256']!=digest(features,guard)):
            raise ValueError('Every row of original complete raw input and float32 baseline coordinate required')
        scaler=scaling.fit_scaler(features,eligible,guard=guard)
        scaling.apply_scaler(features,scaler,guard=guard,output=features)
        record=dict(raw=proof,normalized_input_sha256=digest(features,guard),eligible_rows=int(eligible.sum()))
        save_preprocessing(stage,weights,scaler,record);preprocessing[stage]=record
        return features,baseline

    inner,ib=prepare('inner',inner_weights,partition['fit'])
    for seed in cfg['seeds']:
        guard()
        selected,count=train_nn(inner[partition['fit']],labels[partition['fit']]-ib[partition['fit']],
            cfg,seed+fold,cfg['nn_epochs'],valid=(inner[partition['valid']],labels[partition['valid']]-ib[partition['valid']]))
        if type(count) is not int or not 0<=count<=cfg['nn_epochs']:raise ValueError('Actual original known-selected iteration count required')
        counts.append(count);del selected;release();guard()
    del inner,ib;release();guard()

    full,fb=prepare('full',full_weights,partition['train'])
    correction=np.zeros(int(partition['held'].sum()),dtype=np.float64)
    for seed,count in zip(cfg['seeds'],counts):
        if count:
            guard();model,actual=train_nn(full[partition['train']],labels[partition['train']]-fb[partition['train']],
                                         cfg,seed+fold,count)
            if type(actual) is not int or actual!=count:raise ValueError('Fresh refit must complete every selected original epoch')
            values=predict(model,full[partition['held']])
            if values.dtype!=np.float32 or values.shape!=correction.shape or not np.isfinite(values).all():
                raise ValueError('Every finite actual-model held prediction required')
            correction+=values/3
            save_model(model,seed,count);del model;release();guard()
    prediction=np.clip(fb[partition['held']]+correction,1.05,4.95).astype('<f4')
    del full,fb;release();guard()
    return dict(fold=fold,chosen_iterations=counts,prediction=prediction,preprocessing=preprocessing,
                seed_order=list(cfg['seeds']),seed_offset=fold,
                schedule='all_original_independent_inner_selections_then_fresh_outer_refits',
                full_representation_matrices_resident_at_once=1)
