"""Source-bound owned neural phases; only the reviewed supervisor dispatches."""
import argparse
import upstream_rbm400_neural_execution_reference_v3 as execution
import upstream_rbm400_neural_study_reference_v3 as study
import verify_upstream_rbm400_neural_study_reference_v3 as replay
import upstream_rbm400_neural_canonical_reference_v3 as canonical


def main(phase):
    execution.authorize(phase);guard=execution.WorkerGuard(phase);guard()
    plan_sha=execution.inputs.sha(execution.REPORT/'plan.json')
    if phase=='prepare':
        value=dict(manifests=study.prepare(guard))
    elif phase=='select':
        canonical.select(guard)
        path=execution.ROOT/'experiments/upstream-own-rbm400-neural-selection-reference-v3/selection.json'
        value=dict(scientific_proof_sha256=execution.inputs.sha(path))
    else:
        kind,arm=phase.split('_');report=study.paths(arm)[1]
        if kind=='fit':study.fit(arm,guard);path=report/'fit-completion.json'
        elif kind=='replay':replay.verify(arm,guard);path=report/'independent-verification.json'
        elif kind=='canonical':canonical.canonical(arm,guard);path=report/'independent-canonical-verification.json'
        else:raise ValueError('Exact declared scientific phase required')
        value=dict(scientific_proof_sha256=execution.inputs.sha(path))
    value.update(state=execution.RESULTS[phase][1],execution_plan_sha256=plan_sha,
                 qualifying_labels_read=False,observed=execution.stamp())
    execution.validate_result(phase,value,plan_sha)
    execution.exclusive_json(execution.STUDY/execution.RESULTS[phase][0],value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=execution.PHASES,required=True)
    main(parser.parse_args().phase)
