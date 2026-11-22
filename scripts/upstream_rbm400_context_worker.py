"""Owned complete CPU context-data phases; no MPS work or recovered audit."""
import argparse
import upstream_rbm400_context_execution as execution
import upstream_rbm400_context_data as data
import verify_upstream_rbm400_context_data as verification


def main(phase):
    execution.authorize(phase);guard=execution.WorkerGuard(phase);guard()
    plan_sha=execution.inputs.sha(execution.REPORT/'plan.json')
    if phase=='assemble':
        result=data.admit(guard)
        if result['neural_models_fitted']!=0 or result['qualifying_labels_read'] is not False:
            raise ValueError('Full known-only context data assembly required')
        proof=data.REPORT/'manifest.json'
        value=dict(state=execution.RESULTS[phase][1],data_manifest_sha256=execution.inputs.sha(proof))
    elif phase=='verify':
        result=verification.verify(guard);proof=data.REPORT/'independent-verification.json'
        if (result['state']!=verification.STATE
                or result['every_both_arm_operator_bit_verified'] is not True
                or result['every_group_statistic_and_ridge_prediction_bit_verified'] is not True):
            raise ValueError('Complete independent both-arm full-data/ridge proof required')
        value=dict(state=execution.RESULTS[phase][1],verification_sha256=execution.inputs.sha(proof),
                   all_original_and_both_arm_operator_statistic_and_ridge_bits_verified=True)
    else:raise ValueError('Declared complete CPU phase required')
    value.update(execution_plan_sha256=plan_sha,neural_models_fitted=0,
                 qualifying_labels_read=False,observed=execution.stamp())
    execution.validate_result(phase,value,plan_sha)
    execution.exclusive_json(execution.STUDY/execution.RESULTS[phase][0],value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=execution.PHASES,required=True)
    main(parser.parse_args().phase)
