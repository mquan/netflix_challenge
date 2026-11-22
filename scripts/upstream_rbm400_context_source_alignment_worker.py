"""Owned full readonly replay; no assembly, fitting or qualifying outcomes."""
import argparse
import upstream_rbm400_context_source_alignment_execution as execution
import verify_upstream_rbm400_context_source_alignment as verification


def main(phase):
    if phase!='verify':raise ValueError('Only complete readonly replay is authorized')
    execution.authorize(phase);guard=execution.WorkerGuard(phase);guard()
    result=verification.verify(guard)
    if (result['state']!=verification.STATE or result['every_both_arm_operator_bit_verified'] is not True
            or result['every_group_statistic_and_ridge_prediction_bit_verified'] is not True
            or result['both_query_alias_independently_verified'] is not True):
        raise ValueError('Complete actual independent original operator/statistic/ridge replay required')
    value=dict(state=execution.RESULTS[phase][1],
        execution_plan_sha256=execution.inputs.sha(execution.REPORT/'plan.json'),
        verification_sha256=execution.inputs.sha(verification.REPORT/'independent-verification.json'),
        all_original_and_both_arm_operator_statistic_and_ridge_bits_verified=True,
        neural_models_fitted=0,qualifying_labels_read=False,observed=execution.stamp())
    execution.validate_result(phase,value,value['execution_plan_sha256'])
    execution.exclusive_json(execution.STUDY/execution.RESULTS[phase][0],value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=execution.PHASES,required=True)
    main(parser.parse_args().phase)
