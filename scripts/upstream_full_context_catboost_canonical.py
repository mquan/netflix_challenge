"""Compensated canonical known-only tree scores and predeclared arm choice.

No training, qualifying prediction access, outcome audit or CLI occurs here.
The future owner must authenticate complete original fit/replay receipts first.
"""
import datetime
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_full_context_catboost_study as study
import upstream_full_context_catboost_consumer as consumer
import verify_upstream_full_context_catboost_study as replay

STATE = 'complete_separate_full_context_tree_compensated_canonical_known_CV_score'
SELECTION_STATE = 'complete_both_full_context_tree_arms_known_only_canonical_selection'


def code_bindings():
    root=study.ROOT
    return {str(Path(__file__).relative_to(root)):study.sha(Path(__file__))}


def score(prediction, labels):
    if (prediction.ndim!=1 or prediction.shape!=labels.shape or not len(prediction)
            or prediction.dtype!=np.dtype('<f4') or labels.dtype!=np.dtype('<f8')
            or not np.isfinite(prediction).all() or not np.isin(labels,[1,2,3,4,5]).all()):
        raise ValueError('Every original finite float32 held prediction and known integer outcome required')
    def errors():
        for p,y in zip(prediction,labels):
            delta=min(4.95,max(1.05,float(p)))-float(y)
            yield delta*delta
    squared=math.fsum(errors())
    return math.sqrt(squared/len(prediction))


def choose(scores):
    if set(scores)!={'scalar','latent'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in scores.values()):
        raise ValueError('Both complete finite original known-only canonical scores required')
    return 'scalar' if scores['scalar']<=scores['latent']+1e-12 else 'latent'


def verify(arm,manifest_sha256,fit_sha256,replay_sha256,guard):
    consumer.no_torch();root=study.ROOT;run,report=study.paths(arm)
    if (report/'independent-canonical-verification.json').exists():
        raise ValueError('Existing canonical proof preserved; no automatic retry')
    manifest,metadata,expected=study.load(arm,manifest_sha256,guard)
    path=report/'independent-verification.json';proof=consumer.verified_json(path,replay_sha256,root,guard)
    if (proof['state']!=replay.STATE or proof['arm']!=arm or proof['manifest_sha256']!=manifest_sha256
            or proof['fit_completion_sha256']!=fit_sha256 or proof['rows']!=study.ROWS
            or proof['qualifying_rows']!=study.QUALIFYING_ROWS or proof['original_seed_slots']!=15
            or proof['ridge_equations']!=10 or proof['qualifying_labels_read'] is not False
            or proof['every_original_raw_population_and_held_prediction_bit_verified'] is not True
            or proof['every_full_OOF_join_bit_verified'] is not True
            or proof['every_eligible_ridge_equation_and_coefficient_bit_verified'] is not True):
        raise ValueError('Complete original actual-model/raw-operator/ridge replay required before canonical scoring')
    records=consumer.verify_files(root,proof['output_sha256'],guard)
    sources=consumer.authenticate_sources(metadata,manifest['source_metadata_sha256'],root,guard)
    code=code_bindings()
    try:
        prediction_path=run/'prediction.npy';relative=str(prediction_path.relative_to(root))
        if relative not in records:raise ValueError('Original complete replay-bound OOF vector required')
        with consumer.cache.mapped(prediction_path,records[relative],'<f4',(study.ROWS,),guard) as predictions:
            with consumer.known_arrays(metadata,sources,root,guard) as (labels,users,groups):
                result=score(predictions,labels)
        value=dict(state=STATE,observed=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,
            manifest_sha256=manifest_sha256,fit_completion_sha256=fit_sha256,independent_replay_sha256=replay_sha256,
            prediction_sha256=proof['output_sha256'][relative],rows=study.ROWS,tree_only_CV_rmse=result,
            prediction_representation='float32 stored OOF, independently reclipped in float64',
            accumulation='math.fsum over every original squared known error',canonical_code_sha256=code,
            qualifying_labels_read=False,score_is_complete_five_user_fold_CV=True,production_study_admitted=False)
    finally:
        try:
            if consumer.verify_files(root,proof['output_sha256'],guard)!=records:
                raise ValueError('Every original replay-bound output must remain unchanged during scoring')
        finally:
            if consumer.authenticate_sources(metadata,manifest['source_metadata_sha256'],root,guard)!=sources:
                raise ValueError('Every original known source must remain unchanged during canonical scoring')
    if code_bindings()!=code:
        raise ValueError('Canonical scoring implementation changed during scoring')
    study.save_json(report/'independent-canonical-verification.json',value,guard);return value


def select(canonical_sha256,guard):
    consumer.no_torch();root=study.ROOT
    if set(canonical_sha256)!=set(study.ARMS):raise ValueError('Both completed canonical arm proofs required')
    proofs={};scores={}
    for arm in study.ARMS:
        path=study.paths(arm)[1]/'independent-canonical-verification.json'
        p=consumer.verified_json(path,canonical_sha256[arm],root,guard)
        if (p['state']!=STATE or p['arm']!=arm or p['rows']!=study.ROWS or p['qualifying_labels_read'] is not False
                or p['score_is_complete_five_user_fold_CV'] is not True):
            raise ValueError('Both complete original known-only tree CV proofs required')
        # Revalidate every linked original artifact rather than accepting a
        # rehashed partial/foreign "complete" score receipt.
        run,report=study.paths(arm)
        replay_proof=consumer.verified_json(report/'independent-verification.json',p['independent_replay_sha256'],root,guard)
        study.load(arm,p['manifest_sha256'],guard)
        if (replay_proof['state']!=replay.STATE or replay_proof['rows']!=study.ROWS
                or replay_proof['manifest_sha256']!=p['manifest_sha256']
                or replay_proof['fit_completion_sha256']!=p['fit_completion_sha256']
                or replay_proof['every_full_OOF_join_bit_verified'] is not True
                or replay_proof['qualifying_labels_read'] is not False
                or replay_proof['output_sha256'][str((run/'prediction.npy').relative_to(root))]!=p['prediction_sha256']):
            raise ValueError('Canonical score must bind complete original actual-model replay and exact vector')
        if (replay_proof['qualifying_rows']!=study.QUALIFYING_ROWS or replay_proof['original_seed_slots']!=15
                or replay_proof['ridge_equations']!=10
                or replay_proof['every_original_raw_population_and_held_prediction_bit_verified'] is not True
                or replay_proof['every_eligible_ridge_equation_and_coefficient_bit_verified'] is not True):
            raise ValueError('Complete both-arm original stream/model/ridge replay required for selection')
        _,_,_,_,_,original=replay.bindings(arm,p['manifest_sha256'],p['fit_completion_sha256'],guard)
        if any(replay_proof['output_sha256'].get(name)!=digest for name,digest in original.items()):
            raise ValueError('Every canonical score must retain the full original saved study closure')
        consumer.verify_files(root,replay_proof['output_sha256'],guard)
        consumer.verify_files(root,replay_proof['verifier_code_sha256'],guard)
        consumer.verify_files(root,p['canonical_code_sha256'],guard)
        proofs[arm]=p;scores[arm]=p['tree_only_CV_rmse']
    arm=choose(scores);path=root/'experiments/upstream-full-context-catboost-selection/selection.json'
    result=dict(state=SELECTION_STATE,observed=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_arm=arm,
        tree_only_known_CV_scores=scores,tie_tolerance=1e-12,tie_rule='scalar',canonical_sha256=canonical_sha256,
        selected_known_prediction_sha256=proofs[arm]['prediction_sha256'],selection_uses_only_complete_known_CV=True,
        qualifying_labels_read=False,production_study_admitted=False,
        remaining='Complete selected original qualifying bag and independent actual-model/operator replay, then own18 calibration/audits')
    study.save_json(path,result,guard);return result
