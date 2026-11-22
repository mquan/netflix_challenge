"""Complete independent tree-arm replay component; no CLI or production owner.

The future owned worker must freshly rebuild the complete operator stream in
an isolated producer and supply its actual zero-exit receipt. This verifier
checks every original ridge system, saved fold/model bit and OOF join before
publishing a proof. No fitting or qualifying outcomes are used here.
"""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import numpy as np
import upstream_full_context_catboost_study as study
import upstream_full_context_catboost_consumer as consumer
import upstream_full_context_catboost_wire as wire
import verify_upstream_full_context_catboost_fold as fold_replay

STATE = 'complete_independent_full_context_tree_all_folds_raw_inputs_ridge_systems_saved_models_and_OOF_bits'


def code_bindings():
    root=study.ROOT
    return {str(p.relative_to(root)):study.sha(p) for p in (Path(__file__),Path(fold_replay.__file__))}


def coefficients(gram, rhs, stored, guard):
    if rhs.ndim!=2:raise ValueError('Complete original two-dimensional ridge targets required')
    d=rhs.shape[1]
    if (d not in (562,563) or gram.shape!=(10,d,d) or rhs.shape!=(10,d)
            or stored.shape!=(5,2,d) or any(a.dtype!=np.dtype('<f8') or not np.isfinite(a).all() for a in (gram,rhs,stored))):
        raise ValueError('Complete original full-width group systems and ten saved ridge vectors required')
    residuals=[]
    for outer in range(5):
        for population in range(2):
            guard()
            keep=np.asarray([g//2!=outer and (population==1 or g%2==0) for g in range(10)],dtype=bool)
            regularizer=np.eye(d,dtype=np.float64)*1000.;regularizer[-1,-1]=0.
            matrix=gram[keep].sum(axis=0)+regularizer;target=rhs[keep].sum(axis=0)
            fresh=np.linalg.solve(matrix,target)
            if fresh.tobytes()!=stored[outer,population].tobytes():
                raise ValueError('Every independently solved eligible ridge coefficient bit required')
            residual=float(np.max(np.abs(matrix@stored[outer,population]-target)))
            if residual>1e-12*(1+float(np.max(np.abs(target)))):
                raise ValueError('Every original ridge1000 equation and unpenalized intercept required')
            residuals.append(residual)
    return residuals


def bindings(arm, manifest_sha256, fit_sha256, guard):
    root=study.ROOT;run,report=study.paths(arm)
    manifest,metadata,expected=study.load(arm,manifest_sha256,guard)
    path=report/'fit-completion.json';fitted=consumer.verified_json(path,fit_sha256,root,guard)
    if (fitted['state']!=study.FITTED or fitted['arm']!=arm or fitted['manifest_sha256']!=manifest_sha256
            or fitted['all_five_folds_complete'] is not True or fitted['original_seed_slots']!=15
            or fitted['zero_slots_retained_in_denominator'] is not True or fitted['qualifying_labels_read'] is not False
            or fitted['production_study_admitted'] is not False
            or set(fitted['fold_proof_sha256'])!={str((report/'folds'/('%s.json'%f)).relative_to(root)) for f in range(5)}):
        raise ValueError('Every original completed fold, seed slot and source-bound fit receipt required')
    closure={str(path.relative_to(root)):fit_sha256,str((report/'manifest.json').relative_to(root)):manifest_sha256}
    folds=[];count=0
    for fold in range(5):
        path=report/'folds'/('%s.json'%fold);relative=str(path.relative_to(root));digest=fitted['fold_proof_sha256'][relative]
        row=consumer.verified_json(path,digest,root,guard);closure[relative]=digest
        if (row['fold']!=fold or row['manifest_sha256']!=manifest_sha256 or row['seed_order']!=list(study.SEEDS)
                or len(row['chosen_iterations'])!=3 or row['qualifying_labels_read'] is not False):
            raise ValueError('Exact source-bound original fold and ordered seed inventory required')
        counts=row['chosen_iterations']
        if any(type(c) is not int or not 0<=c<=2000 for c in counts):
            raise ValueError('Original full-budget selected tree counts required')
        positive={str(s) for s,c in zip(study.SEEDS,counts) if c}
        required={'inner-pipeline.json','full-pipeline.json','prediction.npy'}|{'tree-%s.cbm'%s for s in positive}
        directory=run/'folds'/str(fold)
        if (set(row['models'])!=positive or set(row['outputs'])!=required
                or row['actual_saved_model_count']!=len(positive) or {p.name for p in directory.iterdir()}!=required):
            raise ValueError('Exact original positive models, zero slots and no extra/missing payloads required')
        for name,record in row['outputs'].items():
            actual=directory/name
            if Path(record['path'])!=actual or actual.resolve()!=actual or actual.stat().st_size!=record['bytes']:
                raise ValueError('Exact original canonical saved fold payload path and size required')
            closure[str(actual.relative_to(root))]=record['sha256']
        for seed in positive:
            if row['models'][seed]!=row['outputs']['tree-%s.cbm'%seed]:
                raise ValueError('Model inventory must bind the same actual saved payload record')
        for stage in ('inner','full'):
            p=directory/(stage+'-pipeline.json')
            actual=consumer.verified_json(p,row['outputs'][p.name]['sha256'],root,guard)
            if row['populations'][stage]!=actual:
                raise ValueError('Exact independently recorded original raw population pipeline required')
        folds.append(row);count+=len(positive)
    if fitted['actual_saved_model_count']!=count:
        raise ValueError('Every actual original saved model required in total inventory')
    prediction=fitted['prediction'];path=run/'prediction.npy'
    if Path(prediction['path'])!=path or path.resolve()!=path or path.stat().st_size!=prediction['bytes']:
        raise ValueError('Original complete OOF payload path and size required')
    closure[str(path.relative_to(root))]=prediction['sha256']
    return manifest,metadata,expected,fitted,folds,closure


def verify(arm,manifest_sha256,fit_sha256,source,*,guard,producer_exit_gate):
    consumer.no_torch();root=study.ROOT;run,report=study.paths(arm)
    if (report/'independent-verification.json').exists():
        raise ValueError('Prior complete independent proof retained; no automatic retry')
    if not callable(guard) or not callable(producer_exit_gate):
        raise ValueError('Owned whole-group guard and actual producer-exit gate required')
    manifest,metadata,expected,fitted,folds,closure=bindings(arm,manifest_sha256,fit_sha256,guard)
    # Authenticate every model/pipeline/prediction, independently of fit code.
    outputs=consumer.verify_files(root,closure,guard);code=code_bindings()
    features,baselines,stream_proof=wire.receive(source,expected,guard=guard)
    gate=producer_exit_gate(stream_proof)
    if set(gate)!={'path','sha256','producer'}:
        raise ValueError('Actual original isolated replay-producer receipt required')
    consumer.producer_exit(gate['path'],gate['sha256'],gate['producer'],expected,manifest['source_metadata_sha256'],root,guard)
    gate_path=Path(gate['path']);relative=str(gate_path.relative_to(root))
    if relative in closure and closure[relative]!=gate['sha256']:raise ValueError('Conflicting original exit receipt binding')
    closure[relative]=gate['sha256'];outputs.update(consumer.verify_files(root,{relative:gate['sha256']},guard))
    sources=consumer.authenticate_sources(metadata,manifest['source_metadata_sha256'],root,guard)
    d=metadata['dimensions']['design_columns'];equations=[];proofs=[];seen=np.zeros(study.ROWS,dtype=bool)
    reconstructed=np.empty(study.ROWS,dtype='<f4')
    try:
        with ExitStack() as stack:
            values={}
            locations={name:root/consumer.ARRAYS/name for name in ('gram.npy','rhs.npy')}
            for name,shape in [('gram.npy',(10,d,d)),('rhs.npy',(10,d))]:
                path=locations[name];relative=str(path.relative_to(root))
                if relative not in sources:raise ValueError('All original complete group statistics must be in admitted source closure')
                values[name]=stack.enter_context(consumer.cache.mapped(path,sources[relative],'<f8',shape,guard))
            weight_path=run/'weights.npy';weight_hash=manifest['outputs']['weights']['sha256']
            weight_records=consumer.verify_files(root,{str(weight_path.relative_to(root)):weight_hash},guard)
            stored=stack.enter_context(consumer.cache.mapped(weight_path,weight_records[str(weight_path.relative_to(root))],'<f8',(5,2,d),guard))
            equations=coefficients(values['gram.npy'],values['rhs.npy'],stored,guard)
            if hashlib.sha256(stored.tobytes()).hexdigest()!=metadata['weights_sha256']:
                raise ValueError('All ten original coefficient vectors must match the raw stream declaration')
            with consumer.known_arrays(metadata,sources,root,guard) as (labels,users,groups):
                for fold,row in enumerate(folds):
                    guard();held=groups//2==fold;path=run/'folds'/str(fold)/'prediction.npy';relative=str(path.relative_to(root))
                    with consumer.cache.mapped(path,outputs[relative],'<f4',(int(held.sum()),),guard) as prediction:
                        proof=fold_replay.verify(features,baselines[fold],users,groups,row,row['models'],prediction,guard)
                        if seen[held].any():raise ValueError('Every held user population exactly once required')
                        reconstructed[held]=prediction;seen[held]=True;proofs.append(proof)
                path=run/'prediction.npy'
                with consumer.cache.mapped(path,outputs[str(path.relative_to(root))],'<f4',(study.ROWS,),guard) as prediction:
                    if not seen.all() or reconstructed.tobytes()!=prediction.tobytes():
                        raise ValueError('Every original full OOF row must equal the independently replayed held-fold join')
            if consumer.verify_files(root,{str(weight_path.relative_to(root)):weight_hash},guard)!=weight_records:
                raise ValueError('Original complete saved ridge systems changed during replay')
        study.load(arm,manifest_sha256,guard)
        result=dict(state=STATE,arm=arm,manifest_sha256=manifest_sha256,fit_completion_sha256=fit_sha256,
            rows=study.ROWS,qualifying_rows=study.QUALIFYING_ROWS,original_seed_slots=15,
            actual_saved_models_reopened=sum(p['actual_saved_models_reopened'] for p in proofs),
            every_original_raw_population_and_held_prediction_bit_verified=True,every_full_OOF_join_bit_verified=True,
            every_eligible_ridge_equation_and_coefficient_bit_verified=True,ridge_equations=10,max_ridge_equation_residuals=equations,
            fold_replays=proofs,replay_stream=stream_proof,output_sha256=closure,
            actual_replay_producer_exit_receipt=dict(path=str(gate_path),sha256=gate['sha256'],producer=gate['producer']),
            verifier_code_sha256=code,
            qualifying_labels_read=False,production_study_admitted=False,
            remaining='Separate compensated canonical known-only scoring; full selected qualifying bag and own18 audits')
    finally:
        try:
            if consumer.authenticate_sources(metadata,manifest['source_metadata_sha256'],root,guard)!=sources:
                raise ValueError('Every original input identity must survive full independent tree replay')
        finally:
            if consumer.verify_files(root,closure,guard)!=outputs:
                raise ValueError('Every saved model/pipeline/prediction identity must survive full independent replay')
            if code_bindings()!=code:
                raise ValueError('Independent replay implementation changed during full verification')
    study.save_json(report/'independent-verification.json',result,guard)
    return result
