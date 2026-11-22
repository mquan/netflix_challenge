"""Owned full selected-arm qualifying inference or independent replay."""
import argparse
import upstream_rbm400_bag_execution_reference_v4 as execution
import upstream_rbm400_bag_reference_v4 as bag
import verify_upstream_rbm400_bag_reference_v4 as replay


def main(phase):
    execution.authorize(phase);guard=execution.WorkerGuard(phase);guard()
    plan_sha=execution.inputs.sha(execution.REPORT/'plan.json')
    if phase=='infer':bag.infer(guard);path=bag.REPORT/'inference-completion.json'
    elif phase=='verify':replay.verify(guard);path=bag.REPORT/'independent-verification.json'
    else:raise ValueError('Only complete inference and independent replay are declared')
    value=dict(state=execution.RESULTS[phase][1],execution_plan_sha256=plan_sha,
        scientific_proof_sha256=execution.inputs.sha(path),qualifying_labels_read=False,observed=execution.stamp())
    execution.validate_result(phase,value,plan_sha)
    execution.exclusive_json(execution.STUDY/execution.RESULTS[phase][0],value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=execution.PHASES,required=True)
    main(parser.parse_args().phase)
