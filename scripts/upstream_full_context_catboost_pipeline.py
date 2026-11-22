"""Connect isolated source roles to complete tree fit/replay components.

No CLI, queue, standalone owner or production admission is provided. The future
owned worker must already hold its original group/lanes/watchdog/resource gate.
Its producer command must wait for byte G on stdin before touching full data.
Actual child authorization and failure preservation belong to that owner.
"""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import upstream_full_context_catboost_study as study
import upstream_full_context_catboost_consumer as consumer
import upstream_full_context_catboost_exit as lifecycle
import verify_upstream_full_context_catboost_study as replay

# Retain pipe/control handles for an original still-live source if the owner
# reports failure. There is no termination/restart/release path in this module.
PRESERVED=[]


def code_paths():
    return [Path(__file__),Path(lifecycle.__file__)]


def run(mode,arm,manifest_sha256,*,command,guard,admit_role,preserve_family,fit_sha256=None):
    consumer.no_torch()
    if (mode not in ('fit','replay') or arm not in study.ARMS
            or any(not callable(f) for f in (guard,admit_role,preserve_family))
            or not isinstance(command,list) or not command or command[0]!=sys.executable
            or any(type(v) is not str or not v for v in command)
            or (mode=='fit' and fit_sha256 is not None)
            or (mode=='replay' and not consumer.wire.digest_ok(fit_sha256))):
        raise ValueError('Exact original mode/arm and externally owned role/guard/preservation hooks required')
    root=consumer.identity.process(os.getpid())
    if not root or root['pid']!=root['pgid']:
        raise ValueError('Already authorized dedicated scientific root process group required')
    # Keep the original observer's exact scope: no hidden arbitrary numerical
    # helper names and no expansion of its selected-process predicate.
    scripts=[Path(token) for token in command[1:] if token.endswith('.py')]
    if not scripts or any(not p.is_absolute() or p.resolve()!=p or not p.is_relative_to(study.ROOT)
            or not p.name.startswith(('upstream_','verify_upstream_')) or not p.is_file() for p in scripts):
        raise ValueError('Explicit original-scope source-role script paths required')
    guard();manifest,metadata,expected=study.load(arm,manifest_sha256,guard)
    run_path,report=study.paths(arm)
    if mode=='fit':
        if (report/'fit-started.json').exists() or (run_path/'folds').exists():
            raise ValueError('Original partial/full learned state preserved; no pipeline retry')
    else:
        if (report/'independent-verification.json').exists():
            raise ValueError('Original independent replay retained; no pipeline retry')
        replay.bindings(arm,manifest_sha256,fit_sha256,guard)
    receipt_path=report/(mode+'-producer-exit.json')
    admission_path=report/(mode+'-producer-admission.json')
    if receipt_path.exists() or admission_path.exists():
        raise ValueError('Existing owned source-role namespace preserved; no pipeline retry')
    bindings=dict(metadata['source_sha256'])
    for name,row in manifest['outputs'].items():
        path=Path(row['path']);relative=str(path.relative_to(study.ROOT))
        if relative in bindings and bindings[relative]!=row['sha256']:raise ValueError('Conflicting original source binding')
        bindings[relative]=row['sha256']
    bindings[str((report/'manifest.json').relative_to(study.ROOT))]=manifest_sha256
    for path in [*code_paths(),*scripts]:
        bindings[str(path.relative_to(study.ROOT))]=study.sha(path)
    child=None;captured=None
    try:
        # The owned root drains this pipe in the same frame. The source cannot
        # write a matrix/header before capture and external role admission.
        child=subprocess.Popen(command,cwd=study.ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        captured=consumer.identity.process(child.pid)
        record=lifecycle.capture(child,owner=root,group=root['pid'],expected_contract=expected,
            metadata_sha256=manifest['source_metadata_sha256'],source_sha256=bindings,guard=guard)
        if captured and not lifecycle.same_lifetime(captured,record.producer):
            raise ValueError('Original producer lifetime changed during source capture')
        captured=record.producer
        request=dict(mode=mode,arm=arm,command=command,scientific_root=root,producer=captured,
            manifest_sha256=manifest_sha256,source_metadata_sha256=manifest['source_metadata_sha256'],
            contract_sha256=hashlib.sha256(consumer.wire.encoded(expected)).hexdigest(),
            source_sha256=bindings,qualifying_labels_read=False)
        authorized=admit_role(request)
        if (not isinstance(authorized,dict) or set(authorized)!={'path','sha256'}):
            raise ValueError('Immutable external owned producer authorization required')
        proof=consumer.verified_json(Path(authorized['path']),authorized['sha256'],study.ROOT,guard)
        if proof.get('request')!=request or proof.get('state')!='authorized_original_full_context_tree_producer_role':
            raise ValueError('Actual source-bound producer/root/mode request must match external authorization')
        study.save_json(admission_path,dict(request=request,external_authorization=authorized,
            production_study_admitted=False,qualifying_labels_read=False),guard)
        authorization_bindings=dict(record.source_sha256)
        for path,digest in [(Path(authorized['path']),authorized['sha256']),(admission_path,study.sha(admission_path))]:
            relative=str(path.relative_to(study.ROOT))
            if relative in authorization_bindings and authorization_bindings[relative]!=digest:
                raise ValueError('External authorization conflicts with original immutable source closure')
            authorization_bindings[relative]=digest
            consumer.verify_files(study.ROOT,{relative:digest},guard)
        record=replace(record,source_sha256=authorization_bindings)
        child.stdin.write(b'G');child.stdin.flush();child.stdin.close()
        def exit_gate(stream):
            if (stream['contract_sha256']!=request['contract_sha256']
                    or stream['admission_sha256']!=request['source_metadata_sha256']
                    or stream['exact_footer_and_EOF'] is not True):
                raise ValueError('Exact complete original stream required before observing producer exit')
            while True:
                guard();result=lifecycle.poll(record,receipt_path,guard=guard)
                if result is not None:return result
                time.sleep(.1)
        if mode=='fit':
            result=study.fit(arm,manifest_sha256,child.stdout,guard=guard,producer_exit_gate=exit_gate)
        else:
            result=replay.verify(arm,manifest_sha256,fit_sha256,child.stdout,guard=guard,producer_exit_gate=exit_gate)
        if child.poll()!=0:raise ValueError('Actual original source producer must remain reaped at successful completion')
        proof_path=report/('fit-completion.json' if mode=='fit' else 'independent-verification.json')
        study.save_json(report/(mode+'-pipeline-completion.json'),dict(
            state='complete_actual_isolated_full_context_tree_pipeline_component',mode=mode,arm=arm,
            manifest_sha256=manifest_sha256,scientific_proof_sha256=study.sha(proof_path),
            actual_producer_exit_sha256=study.sha(receipt_path),producer_admission_sha256=study.sha(admission_path),
            external_authorization=authorized,scientific_root=root,producer=captured,
            source_sha256=authorization_bindings,producer_reaped_zero_before_any_tree_fit_or_saved_model_replay=True,
            production_study_admitted=False,qualifying_labels_read=False),guard)
        consumer.no_torch();guard();return result
    except BaseException as error:
        if child is not None and child.poll() is None:
            PRESERVED.append(dict(handle=child,producer=captured,root=root,mode=mode,arm=arm))
            preserve_family(root,{'producer':captured} if captured else {},error)
        raise
    finally:
        # A live numerical role keeps its channels and lifetime. Its external
        # owner/watchdog must preserve the same group; never kill or retry it.
        if child is not None and child.poll() is not None:
            if child.stdin is not None and not child.stdin.closed:child.stdin.close()
            if child.stdout is not None:child.stdout.close()
