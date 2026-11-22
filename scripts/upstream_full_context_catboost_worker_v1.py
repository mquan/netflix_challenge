"""Owned original tree root/source phases, separate from the external owner."""
import argparse
import sys
import upstream_full_context_catboost_execution_v1 as execution


def source(phase,root_pid):
    captured=execution.source_bootstrap(phase,root_pid)
    guard=execution.supervision.Guard(execution.control(phase,captured,'producer'));guard()
    # Import Torch-capable source adapter only after original authorization.
    import upstream_full_context_catboost_producer_v2 as producer
    arm=phase.split('_')[1];prepared=producer.prepare(arm,guard)
    report=execution.study.paths(arm)[1];manifest=execution.read(report/'manifest.json')
    if (execution.pipeline.hashlib.sha256(producer.wire.encoded(prepared['metadata'])).hexdigest()!=manifest['source_metadata_sha256']
            or execution.study.sha(report/'source.json')!=manifest['outputs']['source']['sha256']):
        raise ValueError('Fresh original source metadata must match the stored arm preparation exactly')
    producer.send(prepared,sys.stdout.buffer,guard)


def main(phase):
    e=execution;captured=e.authorize(phase);context=e.control(phase,captured);guard=e.supervision.Guard(context);guard()
    if phase=='select':
        hashes={arm:e.study.sha(e.study.paths(arm)[1]/'independent-canonical-verification.json') for arm in e.study.ARMS}
        e.canonical.select(hashes,guard)
    else:
        kind,arm=phase.split('_');run,report=e.study.paths(arm)
        if kind=='prepare':
            import upstream_full_context_catboost_producer_v2 as producer
            prepared=producer.prepare(arm,guard)
            paths={'plan':e.REPORT/'plan.json','monitor':e.REPORT/'monitor.json','baseline':e.REPORT/'first-scientific-baseline.json',
                'gate':e.REPORT/(phase+'-gate.json'),'ownership':e.REPORT/(phase+'-ownership.json')}
            ownership={k:dict(path=str(v),sha256=e.study.sha(v)) for k,v in paths.items()}
            e.study.prepare_arm(arm,prepared,ownership,guard)
        elif kind in ('fit','replay'):
            def preserve(root,roles,error):
                context.cached_roles.update(roles)
                while True:context.stop(dict(original_source_pipeline_failure=repr(error)))
            kwargs=dict(command=e.producer_command(phase,captured['pid']),guard=guard,
                admit_role=lambda request:e.admit_source(phase,captured,request,guard),preserve_family=preserve)
            if kind=='replay':kwargs['fit_sha256']=e.study.sha(report/'fit-completion.json')
            e.pipeline.run(kind,arm,e.study.sha(report/'manifest.json'),**kwargs)
        elif kind=='canonical':
            e.canonical.verify(arm,e.study.sha(report/'manifest.json'),e.study.sha(report/'fit-completion.json'),
                e.study.sha(report/'independent-verification.json'),guard)
    path,state=e.scientific_path(phase)
    value=dict(state='complete_owned_original_tree_'+phase,execution_plan_sha256=e.study.sha(e.REPORT/'plan.json'),
        scientific_proof_sha256=e.study.sha(path),qualifying_labels_read=False,observed=e.stamp())
    e.validate_result(phase,value,value['execution_plan_sha256']);e.save(e.RESULTS/(phase+'.json'),value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=execution.PHASES,required=True)
    parser.add_argument('--source-root',type=int);args=parser.parse_args()
    if args.source_root:source(args.phase,args.source_root)
    else:main(args.phase)
