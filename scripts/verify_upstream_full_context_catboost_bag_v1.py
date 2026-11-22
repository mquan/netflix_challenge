"""Separate full selected-tree raw/ridge/model/fold/bag replay; no CLI/fit.

Rebuild the full stream in a new isolated source role. This verifier implements
its own saved-model prediction and rounding loops, independently re-solves all
ten eligible ridge systems and checks every known and qualifying output bit.
No recovered outcomes or training occur here; the future owner remains needed.
"""
from contextlib import contextmanager,ExitStack
import hashlib
import os
from pathlib import Path
import numpy as np
import upstream_full_context_catboost_bag_v1 as bag
import upstream_full_context_catboost as backend
import verify_upstream_full_context_catboost_fold as fold_replay
import verify_upstream_full_context_catboost_study as known_replay

study=bag.study
consumer=bag.consumer
STATE='complete_independent_selected_original_tree_raw_ridge_models_all_fold_bag_and_known_bits'


def code_bindings():
    return {str(Path(__file__).relative_to(study.ROOT)):study.sha(Path(__file__))}


@contextmanager
def vector(path,record,rows,guard):
    path=Path(path);guard()
    with path.open('rb') as source:
        before=consumer.file_identity(os.fstat(source.fileno()))
        if path.resolve()!=path or before!=record['identity'] or before[2]!=rows*4:
            raise ValueError('Every original full canonical raw float32 vector required')
        values=np.memmap(source,dtype='<f4',mode='r',shape=(rows,))
        try:
            if not np.isfinite(values).all():raise ValueError('Every original finite full raw vector required')
            yield values
        finally:
            values._mmap.close();guard()
            if before!=consumer.file_identity(os.fstat(source.fileno())) or before!=consumer.file_identity(path.stat()):
                raise ValueError('Original full raw vector changed during independent use')


def ridge(known,metadata,guard):
    root=study.ROOT;source=known['arms'][known['arm']];m=source['manifest'];d=m['dimensions']['design_columns']
    weights=Path(m['outputs']['weights']['path']);bindings={str(weights.relative_to(root)):m['outputs']['weights']['sha256']}
    for name in ('gram.npy','rhs.npy'):
        relative=str(Path(consumer.ARRAYS)/name)
        if relative not in metadata['source_sha256']:raise ValueError('Complete original ridge statistics required in qualifying source closure')
        bindings[relative]=metadata['source_sha256'][relative]
    records=consumer.verify_files(root,bindings,guard)
    with ExitStack() as stack:
        gram=stack.enter_context(consumer.cache.mapped(root/consumer.ARRAYS/'gram.npy',records[str(Path(consumer.ARRAYS)/'gram.npy')],'<f8',(10,d,d),guard))
        rhs=stack.enter_context(consumer.cache.mapped(root/consumer.ARRAYS/'rhs.npy',records[str(Path(consumer.ARRAYS)/'rhs.npy')],'<f8',(10,d),guard))
        w=stack.enter_context(consumer.cache.mapped(weights,records[str(weights.relative_to(root))],'<f8',(5,2,d),guard))
        residuals=known_replay.coefficients(gram,rhs,w,guard)
        if hashlib.sha256(w.tobytes()).hexdigest()!=metadata['weights_sha256']:
            raise ValueError('Exact all-ten independently solved known/qualifying coefficient vectors required')
    if consumer.verify_files(root,bindings,guard)!=records:raise ValueError('Original ridge system identities changed during replay')
    return residuals,bindings


@contextmanager
def models(folds,width,guard):
    if len(folds)!=5:raise ValueError('All five original saved fold inventories required')
    records=[];loaded=[]
    with ExitStack() as stack:
        for fold,row in enumerate(folds):
            counts=row['chosen_iterations'];values={}
            if (row['fold']!=fold or row['seed_order']!=[2026,7,13] or row['seed_offset']!=fold
                    or len(counts)!=3 or any(type(v) is not int or not 0<=v<=2000 for v in counts)
                    or set(row['models'])!={str(s) for s,n in zip((2026,7,13),counts) if n}):
                raise ValueError('Exact original ordered seed counts, positive models and zero denominator slots required')
            for seed,count in zip((2026,7,13),counts):
                if not count:continue
                record=row['models'][str(seed)];signature=record['signature'];params=dict(backend.PARAMETERS,random_seed=seed+fold,iterations=count)
                if (record['seed']!=seed+fold or record['iterations']!=count or record['width']!=width
                        or signature['tree_count']!=count or len(signature['feature_names'])!=width
                        or any(signature['flat_parameters'].get(k)!=v for k,v in params.items())):
                    raise ValueError('Original complete saved model configuration and selected count required')
                path=Path(record['path']);fold_replay.authenticate_model(path,record,guard)
                values[seed]=stack.enter_context(backend.actual_model(path,record,signature,width,guard));records.append((path,record))
            loaded.append(values)
        try:yield loaded,len(records)
        finally:
            for path,record in records:fold_replay.authenticate_model(path,record,guard)


def verify(completion_sha256,source,*,guard,producer_exit_gate):
    consumer.no_torch();root=study.ROOT;art,report=bag.paths()
    if (report/'replay-started.json').exists() or (report/'independent-verification.json').exists():
        raise ValueError('Original independent bag replay namespace preserved; no retry')
    if not callable(guard) or not callable(producer_exit_gate):raise ValueError('External original family guard and successful source exit gate required')
    value=consumer.verified_json(report/'completion.json',completion_sha256,root,guard)
    if (value['state']!=bag.STATE or value['known_rows']!=study.ROWS or value['qualifying_rows']!=study.QUALIFYING_ROWS
            or value['original_fold_slots']!=5 or value['original_seed_slots']!=15 or value['zero_slots_retained_in_denominator'] is not True
            or value['every_original_qualifying_row_inferred'] is not True or value['models_fitted']!=0
            or value['qualifying_labels_read'] is not False or value['production_data_admitted'] is not False
            or value['correction_arithmetic']!='ordered_float64_sum_of_each_seed_prediction_divided_by_three'
            or value['fold_arithmetic']!='float64_baseline_plus_correction_clip_then_float32'
            or value['bag_arithmetic']!='ordered_float64_sum_of_five_float32_folds_divided_by_five_clip_then_float32'
            or set(value['outputs'])!={'known','qualifying',*('fold-'+str(i) for i in range(5))}):
        raise ValueError('Complete original selected pair and five-fold/fifteen-slot arithmetic required')
    known=bag.selected.load(value['selection_sha256'],guard)
    metadata,contract=bag.selected.qualifying_metadata(known,Path(value['query_metadata_path']),value['query_metadata_sha256'],guard)
    if value['arm']!=known['arm'] or value['query_source_admission_sha256']!=contract['admission_sha256']:
        raise ValueError('Exact original known-only selected arm and logical source admission required')
    bindings=dict(known['source_sha256']);bag.selected.merge(bindings,value['source_sha256']);code=code_bindings();bag.selected.merge(bindings,code)
    bag.selected.merge(bindings,{str((report/'completion.json').relative_to(root)):completion_sha256})
    outputs={}
    for name,row in value['outputs'].items():
        expected=art/('known-prediction.f32' if name=='known' else 'prediction.f32' if name=='qualifying' else name+'.f32')
        rows=study.ROWS if name=='known' else study.QUALIFYING_ROWS
        if Path(row['path'])!=expected or row['bytes']!=rows*4:raise ValueError('Exact original complete raw output path and byte count required')
        relative=str(expected.relative_to(root));outputs[relative]=row['sha256']
    bag.selected.merge(bindings,outputs)
    residuals,systems=ridge(known,metadata,guard);bag.selected.merge(bindings,systems)
    records=consumer.verify_files(root,bindings,guard)
    study.save_json(report/'replay-started.json',dict(completion_sha256=completion_sha256,qualifying_labels_read=False),guard)
    next_row=0;models_reopened=0
    try:
        with ExitStack() as stack:
            vectors={name:stack.enter_context(vector(Path(row['path']),records[str(Path(row['path']).relative_to(root))],study.ROWS if name=='known' else study.QUALIFYING_ROWS,guard)) for name,row in value['outputs'].items()}
            path=study.paths(known['arm'])[0]/'prediction.npy';relative=str(path.relative_to(root))
            with consumer.cache.mapped(path,records[relative],'<f4',(study.ROWS,),guard) as original:
                for lo in range(0,study.ROWS,8192):
                    guard();hi=min(lo+8192,study.ROWS)
                    if original[lo:hi].tobytes()!=vectors['known'][lo:hi].tobytes():raise ValueError('Every original matched known tree prediction bit required')
            chosen=known['arms'][known['arm']]
            with models(chosen['folds'],chosen['manifest']['dimensions']['nn_inputs'],guard) as (loaded,models_reopened):
                def consume(lo,hi,features,baselines):
                    nonlocal next_row
                    if lo!=next_row or hi>study.QUALIFYING_ROWS:raise ValueError('Every original qualifying row once in original order required')
                    # Independent direct native prediction and rounding loops;
                    # do not call the production block-prediction implementation.
                    matrix=features.copy(order='C');total=np.zeros(hi-lo,dtype='<f8')
                    for fold in range(5):
                        matrix[:,-1]=baselines[fold,1].astype('<f4');correction=np.zeros(hi-lo,dtype='<f8')
                        for seed in (2026,7,13):
                            if seed not in loaded[fold]:continue
                            guard();predicted=loaded[fold][seed].predict(matrix,thread_count=2);guard()
                            if predicted.dtype!=np.dtype('<f8') or predicted.shape!=(hi-lo,) or not np.isfinite(predicted).all():
                                raise ValueError('Every original native float64 saved-model prediction required')
                            correction+=predicted/3
                        expected=np.clip(baselines[fold,1]+correction,1.05,4.95).astype('<f4')
                        stored=vectors['fold-'+str(fold)][lo:hi]
                        if expected.tobytes()!=stored.tobytes():raise ValueError('Every independent original qualifying fold prediction bit required')
                        total+=stored.astype('<f8')
                    combined=np.clip(total/5,1.05,4.95).astype('<f4')
                    if combined.tobytes()!=vectors['qualifying'][lo:hi].tobytes():raise ValueError('Every independent original qualifying bag prediction bit required')
                    next_row=hi
                wire=bag.stream.receive(source,contract,consume,guard=guard)
                if next_row!=study.QUALIFYING_ROWS or wire!=value['full_wire_proof']:
                    raise ValueError('Every original complete raw feature/baseline/footer/EOF bit must replay')
                gate=producer_exit_gate(wire)
                if not isinstance(gate,dict) or set(gate)!={'path','sha256','producer'}:raise ValueError('Actual original independent source exit required')
                consumer.producer_exit(gate['path'],gate['sha256'],gate['producer'],contract,contract['admission_sha256'],root,guard)
                relative=str(Path(gate['path']).relative_to(root));bag.selected.merge(bindings,{relative:gate['sha256']})
                records.update(consumer.verify_files(root,{relative:gate['sha256']},guard))
        if code_bindings()!=code:raise ValueError('Independent original bag verifier implementation changed')
    finally:
        if consumer.verify_files(root,bindings,guard)!=records:raise ValueError('Every original saved input/model/output identity must survive independent bag replay')
    proof=dict(state=STATE,arm=known['arm'],completion_sha256=completion_sha256,selection_sha256=value['selection_sha256'],
        known_rows=study.ROWS,qualifying_rows=study.QUALIFYING_ROWS,fold_slots=5,seed_slots=15,actual_saved_models_reopened=models_reopened,
        ridge_equations=10,ridge_residuals=residuals,every_known_and_all_five_qualifying_fold_and_bag_bit_verified=True,
        every_original_full_raw_stream_and_baseline_bit_replayed=True,independent_arithmetic_does_not_call_production_predict=True,
        actual_independent_producer_zero_exit_verified=True,models_fitted=0,qualifying_labels_read=False,production_data_admitted=False,
        source_sha256=bindings,verifier_code_sha256=code,remaining='Owned final pair admission; own18 full calibration, preaudit and fixed recovered canonical audits')
    study.save_json(report/'independent-verification.json',proof,guard);return proof
