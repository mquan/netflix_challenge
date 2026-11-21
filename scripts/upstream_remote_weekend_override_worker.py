#!/usr/bin/env python3
"""Only the separately declared live supervisor can authorize a full return."""
import argparse
from pathlib import Path
import upstream_remote_weekend_override_runtime as runtime


def bindings(context,queue):
    values=dict(context.bindings)
    for p in [runtime.PROTOCOL,queue/'plan.json',queue/'monitor.json',queue/'prerequisites.json',
              queue/'pinned-remote.json',queue/'first-scientific-baseline.json',Path(__file__),Path(runtime.__file__)]:
        runtime.ops.merge_bindings(values,{str(p.relative_to(runtime.ROOT)):runtime.ops.sha(p)})
    return values


def run(job,phase,lane_fd):
    guard=runtime.Guard(job,phase,lane_fd)  # Before archive reads/NumPy scientific allocation.
    from upstream_remote_weekend_override_context import Context
    context=Context();queue,report,art=runtime.paths(job);decl=bindings(context,queue)
    revision=runtime.read(queue/'pinned-remote.json')['commit'];context.require_revision(revision)
    spec=context.transport.job_spec(job);guard.check()
    if phase=='transport':
        context.transport.transport(spec,revision,report,art,guard.check)
        runtime.check_bindings(decl);guard.check()
        runtime.write(report/'variant-transport-context.json',dict(state='complete_variant_transport_context_bound_no_numerical_admission_yet',
            source_sha256=decl,remote_commit=revision,job=job,
            transport_plan_sha256=runtime.ops.sha(report/'plan.json'),
            transport_verification_sha256=runtime.ops.sha(report/'transport-verification.json'),
            original_remote_resource_guarantees=False,coordinator_local_guards_unchanged=True,
            qualifying_labels_read=False),exclusive=True)
    elif phase=='independent_admission':
        context_path=report/'variant-transport-context.json';prior=runtime.read(context_path)
        if (prior['state']!='complete_variant_transport_context_bound_no_numerical_admission_yet' or
            prior['job']!=job or prior['remote_commit']!=revision or prior['source_sha256']!=decl or
            prior['transport_plan_sha256']!=runtime.ops.sha(report/'plan.json') or
            prior['transport_verification_sha256']!=runtime.ops.sha(report/'transport-verification.json') or
            prior['original_remote_resource_guarantees'] is not False or prior['qualifying_labels_read'] is not False):
            raise ValueError('Exact preceding configured transport context required')
        stage=report/'original-scientific-validator';runtime.durable_directory(stage)
        metadata=runtime.read(report/'plan.json')['metadata_sha256']
        for name in [*metadata,'plan.json','transport-verification.json']:
            guard.check();source=report/name;target=stage/name
            if '/' in name or target.exists():raise ValueError('Exact unique original metadata filenames required')
            with target.open('xb') as f:
                f.write(source.read_bytes());f.flush()
                import os
                os.fsync(f.fileno())
        context.verifier.verify(spec,stage,art,guard.check)
        base_path=stage/'independent-verification.json';proof=runtime.read(base_path)
        if (proof['state']!='complete_independent_weekend_full_source_queries_projection_disjointness_and_all_compact_bits'
            or proof['all_pairs']!=spec['published_columns'] or proof['qualifying_labels_read'] is not False
            or proof['numerical_public_features_read'] is not False or proof['score_claimed'] is not False):
            raise ValueError('Complete unchanged original scientific verification required')
        runtime.ops.merge_bindings(proof['source_sha256'],decl)
        for p in [base_path,context_path]:
            runtime.ops.merge_bindings(proof['source_sha256'],{str(p.relative_to(runtime.ROOT)):runtime.ops.sha(p)})
        proof.update(protocol_variant='pinned_remote_owner_resource_override_original_numerical_blueprint',
            original_remote_resource_guarantees=False,coordinator_local_guards_unchanged=True,
            original_scientific_validator_result_sha256=runtime.ops.sha(base_path),
            variant_context_sha256=runtime.ops.sha(context_path))
        runtime.check_bindings(proof['source_sha256']);guard.check()
        runtime.write(report/'independent-verification.json',proof,exclusive=True)
    else:raise ValueError('Only original complete transport/admission phases permitted')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',choices=runtime.JOBS,required=True)
    p.add_argument('--phase',choices=runtime.PHASES,required=True);p.add_argument('--lane-fd',type=int,required=True)
    a=p.parse_args();run(a.job,a.phase,a.lane_fd)
