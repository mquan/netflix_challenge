"""Connect selected-tree query inference/replay to an actual isolated source.

This component does not launch a supervisor, admit production data, or acquire
lanes. Its caller must provide original family admission and preservation.
Each fresh source waits for byte G before loading data or importing Torch.
"""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import upstream_full_context_catboost_bag_v1 as bag
import verify_upstream_full_context_catboost_bag_v1 as replay
import upstream_full_context_catboost_exit as lifecycle

study=bag.study
consumer=bag.consumer
PRESERVED=[]


def paths():
    return study.ROOT/'experiments/upstream-full-context-catboost-selected-bag-pipeline-v1'


def code_paths():
    return [Path(__file__),Path(lifecycle.__file__)]


def run(mode,selection_sha256,query_metadata_path,query_metadata_sha256,*,command,
        guard,admit_role,preserve_family,completion_sha256=None):
    consumer.no_torch()
    if (mode not in ('infer','replay') or not all(consumer.wire.digest_ok(v) for v in
            (selection_sha256,query_metadata_sha256))
            or any(not callable(f) for f in (guard,admit_role,preserve_family))
            or not isinstance(command,list) or not command or command[0]!=sys.executable
            or any(type(v) is not str or not v for v in command)
            or (mode=='infer' and completion_sha256 is not None)
            or (mode=='replay' and not consumer.wire.digest_ok(completion_sha256))):
        raise ValueError('Exact selected-pair mode, source bindings and externally owned hooks required')
    root=consumer.identity.process(os.getpid())
    if not root or root['pid']!=root['pgid']:
        raise ValueError('Already authorized dedicated scientific root process group required')
    scripts=[Path(token) for token in command[1:] if token.endswith('.py')]
    if not scripts or any(not p.is_absolute() or p.resolve()!=p or not p.is_relative_to(study.ROOT)
            or not p.name.startswith(('upstream_','verify_upstream_')) or not p.is_file() for p in scripts):
        raise ValueError('Explicit original-scope source-role script paths required')
    guard();known=bag.selected.load(selection_sha256,guard)
    metadata,expected=bag.selected.qualifying_metadata(known,Path(query_metadata_path),query_metadata_sha256,guard)
    art,report=bag.paths();control=paths()
    if mode=='infer':
        if art.exists() or report.exists():raise ValueError('Original partial/full pair preserved; no pipeline retry')
    else:
        if (report/'replay-started.json').exists() or (report/'independent-verification.json').exists():
            raise ValueError('Original independent pair replay preserved; no pipeline retry')
        value=consumer.verified_json(report/'completion.json',completion_sha256,study.ROOT,guard)
        if (value.get('selection_sha256')!=selection_sha256
                or value.get('query_metadata_path')!=str(query_metadata_path)
                or value.get('query_metadata_sha256')!=query_metadata_sha256):
            raise ValueError('Replay must use the exact original selection and query metadata')
    receipt_path=control/(mode+'-producer-exit.json')
    admission_path=control/(mode+'-producer-admission.json')
    pipeline_path=control/(mode+'-pipeline-completion.json')
    if any(p.exists() for p in (receipt_path,admission_path,pipeline_path)):
        raise ValueError('Original selected-pair source-role namespace preserved; no pipeline retry')
    bindings=dict(known['source_sha256'])
    bag.selected.merge(bindings,metadata['source_sha256'])
    bag.selected.merge(bindings,{str(Path(query_metadata_path).relative_to(study.ROOT)):query_metadata_sha256})
    bag.selected.merge(bindings,{str(p.relative_to(study.ROOT)):study.sha(p) for p in [*code_paths(),*scripts]})
    if mode=='replay':
        bag.selected.merge(bindings,{str((report/'completion.json').relative_to(study.ROOT)):completion_sha256})
    consumer.verify_files(study.ROOT,bindings,guard)
    study.durable_directory(control,study.ROOT,guard)
    child=None;captured=None
    try:
        child=subprocess.Popen(command,cwd=study.ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        record=lifecycle.capture(child,owner=root,group=root['pid'],expected_contract=expected,
            metadata_sha256=expected['admission_sha256'],source_sha256=bindings,guard=guard)
        captured=record.producer
        request=dict(mode=mode,arm=known['arm'],command=command,scientific_root=root,producer=captured,
            selection_sha256=selection_sha256,completion_sha256=completion_sha256,
            query_metadata_path=str(query_metadata_path),query_metadata_sha256=query_metadata_sha256,
            source_metadata_sha256=expected['admission_sha256'],
            contract_sha256=hashlib.sha256(consumer.wire.encoded(expected)).hexdigest(),
            source_sha256=bindings,qualifying_labels_read=False)
        authorized=admit_role(request)
        if not isinstance(authorized,dict) or set(authorized)!={'path','sha256'}:
            raise ValueError('Immutable external owned query-producer authorization required')
        proof=consumer.verified_json(Path(authorized['path']),authorized['sha256'],study.ROOT,guard)
        if proof.get('request')!=request or proof.get('state')!='authorized_original_full_context_tree_query_producer_role':
            raise ValueError('Exact actual selected-pair source request must match external authorization')
        study.save_json(admission_path,dict(request=request,external_authorization=authorized,
            production_data_admitted=False,qualifying_labels_read=False),guard)
        authorization_bindings=dict(record.source_sha256)
        bag.selected.merge(authorization_bindings,{str(Path(authorized['path']).relative_to(study.ROOT)):authorized['sha256'],
            str(admission_path.relative_to(study.ROOT)):study.sha(admission_path)})
        consumer.verify_files(study.ROOT,authorization_bindings,guard)
        record=replace(record,source_sha256=authorization_bindings)
        child.stdin.write(b'G');child.stdin.flush();child.stdin.close()
        def exit_gate(stream):
            if (stream['contract_sha256']!=request['contract_sha256']
                    or stream['admission_sha256']!=expected['admission_sha256']
                    or stream['exact_footer_and_EOF'] is not True):
                raise ValueError('Complete exact original query stream required before observing source exit')
            while True:
                guard();result=lifecycle.poll(record,receipt_path,guard=guard)
                if result is not None:return result
                time.sleep(.1)
        if mode=='infer':
            result=bag.infer(selection_sha256,query_metadata_path,query_metadata_sha256,child.stdout,
                guard=guard,producer_exit_gate=exit_gate)
        else:
            result=replay.verify(completion_sha256,child.stdout,guard=guard,producer_exit_gate=exit_gate)
        if child.poll()!=0:raise ValueError('Actual original query producer must remain reaped at zero')
        proof_path=report/('completion.json' if mode=='infer' else 'independent-verification.json')
        study.save_json(pipeline_path,dict(state='complete_actual_isolated_selected_tree_query_pipeline_component',
            mode=mode,arm=known['arm'],selection_sha256=selection_sha256,completion_sha256=completion_sha256,
            scientific_proof_sha256=study.sha(proof_path),query_metadata_sha256=query_metadata_sha256,
            source_metadata_sha256=expected['admission_sha256'],actual_producer_exit_sha256=study.sha(receipt_path),
            producer_admission_sha256=study.sha(admission_path),external_authorization=authorized,
            scientific_root=root,producer=captured,source_sha256=authorization_bindings,
            producer_reaped_zero_before_any_pair_publication=True,production_data_admitted=False,
            qualifying_labels_read=False),guard)
        consumer.no_torch();guard();return result
    except BaseException as error:
        if child is not None and child.poll() is None:
            if captured is None:captured=consumer.identity.process(child.pid)
            PRESERVED.append(dict(handle=child,producer=captured,root=root,mode=mode))
            preserve_family(root,{'producer':captured} if captured else {},error)
        raise
    finally:
        if child is not None and child.poll() is not None:
            if child.stdin is not None and not child.stdin.closed:child.stdin.close()
            if child.stdout is not None:child.stdout.close()
