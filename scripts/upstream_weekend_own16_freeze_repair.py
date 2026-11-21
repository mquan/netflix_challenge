"""Correct full16 weekend freeze, retaining original fit, models and verifier.

Original frozen modules are imported without editing or replacing their globals.
Only the idle-coordinator handoff and new owned queue can dispatch this repair.
"""
import json
from pathlib import Path
import numpy as np
import upstream_own16_weekend_trajectory_calibration as original
import upstream_factored_rbm_execution as rbm
import upstream_factored_rbm_inputs as inputs
import upstream_factored_rbm_terminal as terminal

ROOT=original.u.ROOT
REPORT=original.REPORT
ART=original.RUN
DECL=ROOT/'WEEKEND_OWN16_FREEZE_REPAIR.md'
BASE_IDS=original.BASE_IDS
FAMILY_ID='nn-weekend-trajectory-fold-bag'
CANDIDATES=original.CANDIDATES
COUNTS=original.COUNTS


def protocol_bindings():
    import upstream_weekend_own16_queue_v2 as queue
    paths=[Path(__file__),Path(queue.__file__),DECL,
        ROOT/'experiments/upstream-weekend-own16-freeze-repair/protocol-verification.json',
        ROOT/'experiments/upstream-weekend-own16-freeze-repair/idle-handoff.json']
    paths.extend(queue.REPORT/name for name in ('plan.json','monitor.json','prerequisites.json','first-scientific-baseline.json','freeze-gate.json','freeze-ownership.json'))
    return {str(path.relative_to(ROOT)):inputs.sha(path) for path in paths}


def json_output(path,value,guard):
    return terminal.durable_payload(path,lambda stream:stream.write((json.dumps(value,indent=2)+'\n').encode()),original.u.ROOT,guard)


def freeze(guard):
    if (original.REPORT/'frozen.json').exists():raise ValueError('Original full16 freeze preserved; no retry')
    done,cal,plan=[original.u.read(original.REPORT/name) for name in ('known-completion.json','calibration.json','plan.json')]
    if (done['state']!='complete_original16_known_only_coefficients_before_candidate_freeze'
            or done['calibration_sha256']!=original.sha(original.REPORT/'calibration.json')
            or done['plan_sha256']!=original.sha(original.REPORT/'plan.json') or done['qualifying_labels_read'] is not False
            or set(done['known_predictions'])!={'known-simplex','known-ridge10','known-fixed-half'}
            or plan['families']!=16 or plan['primary']!=CANDIDATES[0]
            or cal['family_ids']!=BASE_IDS+[FAMILY_ID] or [row['id'] for row in cal['families']]!=cal['family_ids']
            or len(cal['simplex_weights'])!=16 or len(cal['ridge_weights'])!=16
            or cal['qualifying_labels_read'] is not False or cal['known_score_is_nested_cross_validation'] is not False):
        raise ValueError('Original complete all16 known-only fit and fixed primary required')
    if {path.name for path in original.RUN.iterdir()}!={name+'.f32' for name in done['known_predictions']}:
        raise ValueError('Original known artifacts/partial outputs preserved; no retry')
    for name,digest in done['known_predictions'].items():
        guard()
        if original.sha(original.RUN/(name+'.f32'))!=digest:raise ValueError('Original known prediction changed')
    proof_path=original.BAG/'independent-verification.json';proof=original.u.read(proof_path)
    if (proof['state']!='complete_independent_complete_trajectory_actual_fold_models_all_query_streams_and_bag'
            or proof['seed_slots']!=15 or proof['rows']!=original.COUNTS['qual']
            or proof['all_original_model_and_input_streams_replayed'] is not True
            or proof['uniform_five_fold_bag_bitwise_equal'] is not True or proof['matched_B_raw_operators_on_both_queries'] is not True
            or proof['known_OOF_and_qualifying_bag_constructions_distinct'] is not True
            or proof['qualifying_labels_read'] is not False or proof['models_fitted']!=0 or proof['qualifying_score_claimed'] is not False):
        raise ValueError('All original fifteen saved models and complete bag replay required')
    bindings=dict(cal['source_sha256'])
    for relative,digest in proof['source_sha256'].items():original.bind(bindings,original.u.ROOT/relative,digest)
    original.bind(bindings,proof_path)
    original.bind(bindings,original.BAG_RUN/'nn-bag.f32',proof['prediction_sha256'])
    for relative,digest in protocol_bindings().items():original.bind(bindings,original.u.ROOT/relative,digest)
    guard();original.verify_bindings(bindings);guard()
    rows=cal['families'];arrays=[]
    for row in rows:
        guard();arrays.append(np.fromfile(original.u.ROOT/row['qualifying_prediction'],dtype='<f4'))
    if len(rows)!=16 or any(value.shape!=(original.COUNTS['qual'],) or not np.isfinite(value).all() for value in arrays):
        raise ValueError('Every original sixteen complete finite qualifying family required')
    specs=[(CANDIDATES[0],arrays,cal['simplex_weights'],0.),
           (CANDIDATES[1],arrays,cal['ridge_weights'],cal['ridge_intercept']),
           (CANDIDATES[2],[arrays[-1],arrays[5]],[.5,.5],0.)]
    outputs={}
    for name,columns,weights,intercept in specs:
        guard();values=original.combine(columns,weights,intercept);guard()
        outputs[name]=terminal.durable_payload(original.RUN/(name+'.f32'),lambda stream:stream.write(memoryview(values)),original.u.ROOT,guard)
    guard();original.verify_bindings(bindings);guard()
    for name,digest in done['known_predictions'].items():
        if original.sha(original.RUN/(name+'.f32'))!=digest:raise ValueError('Original known fit must remain unchanged')
    json_output(original.REPORT/'frozen.json',dict(state='complete_all_three_own16_candidates_frozen_before_any_new_audit',
        observed=original.stamp(),primary=CANDIDATES[0],plan_sha256=original.sha(original.REPORT/'plan.json'),
        calibration_sha256=original.sha(original.REPORT/'calibration.json'),known_completion_sha256=original.sha(original.REPORT/'known-completion.json'),
        source_sha256=bindings,predictions={**done['known_predictions'],**{name:row['sha256'] for name,row in outputs.items()}},
        qualifying_labels_read=False,independent_coefficient_and_arithmetic_verification_remaining=True,
        freeze_cardinality_repair=dict(original_family_count_check=15,correct_full_family_count=16,numerical_recipe_changed=False,
            models_refitted=0,coefficients_refitted=False,original_known_outputs_unchanged=True,original_independent_verifier_retained=True)),guard)


def main():
    import upstream_weekend_own16_queue_v2 as queue
    queue.authorize('freeze');guard=queue.Guard('freeze');guard();freeze(guard)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('freeze',));parser.parse_args();main()
