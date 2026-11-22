"""Complete selected tree known/qualifying pair assembly; no CLI or owner.

Outputs remain pending a separate full raw/operator/model/fold/bag replay.
Only the future reviewed owner can authorize a source role and full family
resource budget. No fitting, qualifying outcomes or score promotion occurs.
"""
from pathlib import Path
import numpy as np
import upstream_full_context_catboost_study as study
import upstream_full_context_catboost_consumer as consumer
import upstream_full_context_catboost_selected_sources as selected
import upstream_full_context_catboost_query_prediction as prediction
import upstream_full_context_catboost_query_stream as stream

STATE='complete_selected_original_tree_known_and_qualifying_bag_pending_independent_full_replay'


def paths():
    return study.ROOT/'artifacts/upstream-full-context-catboost-selected-bag-v1',study.ROOT/'experiments/upstream-full-context-catboost-selected-bag-v1'


def code_bindings():
    root=study.ROOT
    return {str(Path(p).relative_to(root)):study.sha(Path(p)) for p in
        (__file__,selected.__file__,prediction.__file__,stream.__file__)}


def raw(path,values,guard):
    if values.ndim!=1 or values.dtype!=np.dtype('<f4') or not np.isfinite(values).all():
        raise ValueError('Every finite original float32 prediction row required')
    def write(output):
        for lo in range(0,len(values),8192):
            guard();output.write(values[lo:lo+8192].tobytes())
    return study.durable_payload(path,write,study.ROOT,guard)


def infer(selection_sha256,query_metadata_path,query_metadata_sha256,source,*,guard,producer_exit_gate):
    consumer.no_torch();root=study.ROOT;art,report=paths()
    if art.exists() or report.exists():raise ValueError('Existing partial/full selected bag preserved; no retry')
    if not callable(guard) or not callable(producer_exit_gate):raise ValueError('Owned full-family guard and actual producer-exit gate required')
    known=selected.load(selection_sha256,guard)
    metadata,expected=selected.qualifying_metadata(known,Path(query_metadata_path),query_metadata_sha256,guard)
    arm=known['arm'];chosen=known['arms'][arm];bindings=dict(known['source_sha256']);code=code_bindings()
    selected.merge(bindings,code);selected.merge(bindings,{str(Path(query_metadata_path).relative_to(root)):query_metadata_sha256})
    records=consumer.verify_files(root,bindings,guard)
    study.durable_directory(art,root,guard);study.durable_directory(report,root,guard)
    study.save_json(report/'started.json',dict(selection_sha256=selection_sha256,query_metadata_sha256=query_metadata_sha256,
        query_source_admission_sha256=expected['admission_sha256'],models_fitted=0,qualifying_labels_read=False,production_data_admitted=False),guard)
    folds=np.empty((5,study.QUALIFYING_ROWS),dtype='<f4');bag=np.empty(study.QUALIFYING_ROWS,dtype='<f4');next_row=0
    try:
        with prediction.models(chosen['folds'],chosen['manifest']['dimensions']['nn_inputs'],guard) as loaded:
            def consume(lo,hi,features,baselines):
                nonlocal next_row
                if lo!=next_row or hi>study.QUALIFYING_ROWS:raise ValueError('Every original qualifying row exactly once in order required')
                values,combined=prediction.predict(features,baselines,loaded,guard)
                for fold in range(5):folds[fold,lo:hi]=values[fold]
                bag[lo:hi]=combined;next_row=hi
            wire=stream.receive(source,expected,consume,guard=guard)
            if next_row!=study.QUALIFYING_ROWS:raise ValueError('Complete original qualifying population required')
            gate=producer_exit_gate(wire)
            if not isinstance(gate,dict) or set(gate)!={'path','sha256','producer'}:
                raise ValueError('Actual source producer successful-exit receipt required')
            consumer.producer_exit(gate['path'],gate['sha256'],gate['producer'],expected,expected['admission_sha256'],root,guard)
            relative=str(Path(gate['path']).relative_to(root));selected.merge(bindings,{relative:gate['sha256']})
            records.update(consumer.verify_files(root,{relative:gate['sha256']},guard))
        # A successful byte-complete stream and actual source exit precede all
        # prediction publication. Inference before that point is provisional.
        outputs={}
        for fold in range(5):outputs['fold-'+str(fold)]=raw(art/('fold-'+str(fold)+'.f32'),folds[fold],guard)
        outputs['qualifying']=raw(art/'prediction.f32',bag,guard)
        path=study.paths(arm)[0]/'prediction.npy';relative=str(path.relative_to(root))
        with consumer.cache.mapped(path,records[relative],'<f4',(study.ROWS,),guard) as values:
            outputs['known']=raw(art/'known-prediction.f32',values,guard)
        if code_bindings()!=code:raise ValueError('Original bag assembly implementation changed during inference')
        value=dict(state=STATE,arm=arm,selection_sha256=selection_sha256,query_metadata_path=str(query_metadata_path),
            query_metadata_sha256=query_metadata_sha256,query_source_admission_sha256=expected['admission_sha256'],known_rows=study.ROWS,qualifying_rows=study.QUALIFYING_ROWS,
            original_fold_slots=5,original_seed_slots=15,zero_slots_retained_in_denominator=True,
            correction_arithmetic='ordered_float64_sum_of_each_seed_prediction_divided_by_three',
            fold_arithmetic='float64_baseline_plus_correction_clip_then_float32',
            bag_arithmetic='ordered_float64_sum_of_five_float32_folds_divided_by_five_clip_then_float32',
            every_original_qualifying_row_inferred=True,full_wire_proof=wire,
            actual_producer_exit_receipt=dict(path=str(gate['path']),sha256=gate['sha256'],producer=gate['producer']),
            source_sha256=bindings,outputs=outputs,models_fitted=0,qualifying_labels_read=False,
            production_data_admitted=False,remaining='Separate full raw/ridge/model/fold/bag replay; own18 calibration and fixed audits')
    finally:
        if consumer.verify_files(root,bindings,guard)!=records:
            raise ValueError('Every original known study/model/source-metadata/exit identity must survive full bag assembly')
    study.save_json(report/'completion.json',value,guard);return value
