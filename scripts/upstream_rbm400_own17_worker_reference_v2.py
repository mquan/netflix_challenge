"""Owned original17-family calibration, full preaudit and fixed recovered audits."""
import argparse
import upstream_rbm400_own17_execution_reference_v2 as execution
import upstream_rbm400_own17_calibration_reference_v2 as study
import verify_upstream_rbm400_own17_calibration_reference_v2 as replay


def main(phase):
    execution.authorize(phase);guard=execution.WorkerGuard(phase);guard()
    plan_sha=execution.inputs.sha(execution.REPORT/'plan.json')
    if phase=='known':
        study.terminal.durable_directory(study.REPORT,study.ROOT,guard)
        study.terminal.durable_directory(study.ART,study.ROOT,guard)
        study.known(guard);name='known-completion.json'
    elif phase=='freeze':study.frozen(guard);name='frozen.json'
    elif phase=='preaudit':replay.verify(guard);name='independent-preaudit-verification.json'
    elif phase=='audit':replay.audit(guard);name='independent-recovered-audit.json'
    elif phase=='canonical':replay.canonical(guard);name='independent-canonical-verification.json'
    else:raise ValueError('Only original full known/freeze/preaudit/audit/canonical phases required')
    value=dict(state=execution.RESULTS[phase][1],execution_plan_sha256=plan_sha,
        scientific_proof_sha256=execution.inputs.sha(study.REPORT/name),
        qualifying_labels_read=phase in ('audit','canonical'),observed=execution.stamp())
    execution.validate_result(phase,value,plan_sha)
    execution.exclusive_json(execution.STUDY/execution.RESULTS[phase][0],value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=execution.PHASES,required=True)
    main(parser.parse_args().phase)
