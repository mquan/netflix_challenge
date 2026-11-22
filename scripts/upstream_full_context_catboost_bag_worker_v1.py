"""Owned selected-tree source preparation, full query inference/replay and checks."""
import argparse
import sys
import upstream_full_context_catboost_bag_execution_v1 as execution


def source(phase,root_pid):
    e=execution;captured=e.source_bootstrap(phase,root_pid)
    guard=e.supervision.Guard(e.control(phase,captured,'producer'));guard()
    pre=e.read(e.REPORT/'prerequisites.json');stored=e.read(e.SOURCE_REPORT/'metadata.json')
    # The exact full source request has already been validated while Torch-free.
    import upstream_full_context_catboost_query_producer as producer
    prepared=producer.prepare(pre['selected_arm'],guard)
    if prepared['metadata']!=stored:
        raise ValueError('Fresh full qualifying source must reproduce the exact saved metadata')
    producer.send(prepared,sys.stdout.buffer,guard)


def main(phase):
    e=execution;captured=e.authorize(phase);context=e.control(phase,captured)
    guard=e.supervision.Guard(context);guard()
    if phase=='prepare':
        pre=e.read(e.REPORT/'prerequisites.json');known=e.selected.load(pre['selection_sha256'],guard)
        # This dedicated preparation root may import Torch, then exits before
        # the separate Torch-free inference/replay root is authorized.
        import upstream_full_context_catboost_query_producer as producer
        prepared=producer.prepare(known['arm'],guard);metadata=prepared['metadata']
        if (metadata['original_known_source_metadata']!=known['arms'][known['arm']]['metadata']
                or metadata['arm']!=known['arm']):
            raise ValueError('Fresh qualifying preparation must match the completed selected known arm')
        path=e.SOURCE_REPORT/'metadata.json';e.study.save_json(path,metadata,guard)
        e.study.save_json(e.SOURCE_REPORT/'preparation.json',dict(state=e.PREPARED,arm=known['arm'],
            selection_sha256=known['selection_sha256'],metadata_path=str(path),metadata_sha256=e.study.sha(path),
            logical_source_admission_sha256=prepared['contract']['admission_sha256'],
            models_fitted=0,qualifying_labels_read=False,production_data_admitted=False),guard)
    elif phase in ('infer','replay'):
        known,path,digest,metadata,expected=e.query_inputs(guard)
        def preserve(root,roles,error):
            context.cached_roles.update(roles)
            while True:context.stop(dict(original_source_pipeline_failure=repr(error)))
        kwargs=dict(command=e.producer_command(phase,captured['pid']),guard=guard,
            admit_role=lambda request:e.admit_source(phase,captured,request,guard),preserve_family=preserve)
        if phase=='replay':kwargs['completion_sha256']=e.study.sha(e.bag.paths()[1]/'completion.json')
        e.pipeline.run(phase,known['selection_sha256'],path,digest,**kwargs)
    elif phase=='admit':e.admit_pair(guard)
    else:raise ValueError('Exact complete selected-tree phase required')
    path,state=e.scientific_path(phase)
    value=dict(state='complete_owned_original_tree_pair_'+phase,execution_plan_sha256=e.study.sha(e.REPORT/'plan.json'),
        scientific_proof_sha256=e.study.sha(path),qualifying_labels_read=False,observed=e.stamp())
    e.validate_result(phase,value,value['execution_plan_sha256']);e.save(e.RESULTS/(phase+'.json'),value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=execution.PHASES,required=True)
    parser.add_argument('--source-root',type=int);args=parser.parse_args()
    if args.source_root:source(args.phase,args.source_root)
    else:main(args.phase)
