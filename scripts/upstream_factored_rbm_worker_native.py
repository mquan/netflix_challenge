"""Fixed full-budget RBM scientific phases; only the owned controller dispatches.

No device, dimensions, epoch, seed, data path or output overrides are accepted.
Recovered qualifying outcomes are never opened by this program.
"""
import argparse
from contextlib import contextmanager
import os
import sys
import numpy as np
import upstream_factored_rbm_execution_native as execution
import upstream_factored_rbm_admission as admission
import upstream_factored_rbm_inputs as inputs
import upstream_factored_rbm_terminal as terminal
from upstream_factored_rbm_rows import UserRows
from upstream_factored_rbm_training import Config,State,cd_steps


def base(plan_sha):
    return dict(execution_plan_sha256=plan_sha,qualifying_labels_read=False,
                recovered_outcome_paths_read=False,producer_pid=os.getpid(),observed=execution.stamp())


def data_receipt():
    path=execution.STUDY/'data-admission.json';value=execution.read(path)
    execution.validate_result('data',value,inputs.sha(execution.REPORT/'plan.json'))
    declaration=admission.load_declared_plan(execution.ROOT)
    if (value['input_declaration_sha256']!=inputs.sha(execution.ROOT/inputs.REPORT/'plan.json')
            or declaration['exact_config']!=execution.read(execution.REPORT/'plan.json')['configuration']):
        raise ValueError('Original complete data admission and fixed recipe required')
    return value,declaration,inputs.sha(path)


@contextmanager
def reader_for(phase,guard):
    data,declaration,digest=data_receipt();records=data['bound_file_records']
    trpaths,qrpaths=admission.phase_paths(declaration,phase)
    guard()
    with admission.arrays(execution.ROOT,trpaths,guard,records) as tr,admission.arrays(execution.ROOT,qrpaths,guard,records) as qr:
        reader=UserRows(tr,qr,declaration['items']);guard()
        if (reader.n_users!=declaration['users'] or len(tr['user_idxs'])!=declaration['phases'][phase]['training_rows']
                or len(qr['user_idxs'])!=declaration['phases'][phase]['query_rows']
                or admission.statistics(reader,declaration['exact_config'])!=data['phases'][phase]['statistics']):
            raise ValueError('Full original phase reader/statistics required')
        yield reader,data,declaration,digest


def torch_runtime():
    # The separate preflight process must import Torch itself, after checking
    # the environment. Fit/replay processes independently use the same flags.
    import torch
    if not torch.backends.mps.is_available():raise RuntimeError('Actual MPS required; no CPU fallback')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=False)
    return torch


def artifact(name):
    path=execution.ART/name
    if path.parent!=execution.ART or path.name!=name:raise ValueError('Owned flat artifact name required')
    return path


def save_vector(name,values,guard):
    if values.dtype!=np.float32 or values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('Complete finite native f32 vector required')
    return terminal.durable_payload(artifact(name),lambda stream:np.save(stream,values,allow_pickle=False),execution.ROOT,guard)


def project_known(native,data,declaration,guard):
    paths={'mapping':'artifacts/upstream-split3-pilot/probex-row-map.npy'}
    with admission.arrays(execution.ROOT,paths,guard,data['bound_file_records']) as extra:
        mapping=extra['mapping']
        if (mapping.dtype!=np.int64 or mapping.shape!=(declaration['known_rows'],)
                or np.any(mapping<0) or np.any(mapping>=len(native))):
            raise ValueError('Full originally admitted known projection required')
        guard();return native[mapping].copy()


def epoch_path(phase,epoch):return execution.STUDY/('%s-epoch-%02d.json'%(phase,epoch))


def verify_history(phase,reader,data_sha,plan_sha,guard):
    plan=execution.read(execution.REPORT/'plan.json');cfg=Config(**plan['configuration'])
    batches=(len(reader.nonempty_users)+cfg.batch_users-1)//cfg.batch_users
    owner=execution.read(execution.REPORT/('fit_'+phase+'-ownership.json'))
    history={}
    for epoch in range(1,cfg.epochs+1):
        guard();path=epoch_path(phase,epoch);r=execution.read(path)
        if (r['state']!='complete_original_full_nonempty_user_epoch' or r['epoch']!=epoch
                or r['phase']!=phase or r['execution_plan_sha256']!=plan_sha or r['data_admission_sha256']!=data_sha
                or r['configuration']!=plan['configuration'] or r['completed_batches']!=batches
                or r['nonempty_users']!=len(reader.nonempty_users) or r['cd_steps']!=cd_steps(epoch)
                or r['producer_pid']!=owner['pid'] or r['qualifying_labels_read'] is not False
                or r['original_baseline_sha256']!=inputs.sha(execution.REPORT/'first-scientific-baseline.json')):
            raise ValueError('Every original full epoch with original source/worker/baseline required')
        history[str(path.relative_to(execution.ROOT))]=inputs.sha(path)
    return history


def fit(phase,plan,guard,plan_sha):
    torch=torch_runtime();guard()
    with reader_for(phase,guard) as (reader,data,declaration,data_sha):
        state=State(reader,data_sha,Config(**plan['configuration']),'mps');guard()
        for epoch in range(1,state.config.epochs+1):
            state.fit_epoch(epoch,guard)
            execution.exclusive_json(epoch_path(phase,epoch),dict(base(plan_sha),
                state='complete_original_full_nonempty_user_epoch',phase=phase,epoch=epoch,
                configuration=plan['configuration'],data_admission_sha256=data_sha,
                completed_batches=state.completed_batches,nonempty_users=len(reader.nonempty_users),cd_steps=cd_steps(epoch),
                original_baseline_sha256=inputs.sha(execution.REPORT/'first-scientific-baseline.json')),guard)
            print('%s epoch %02d/%02d complete'%(phase,epoch,state.config.epochs),flush=True)
        history=verify_history(phase,reader,data_sha,plan_sha,guard)
        record=terminal.freeze(state,phase,data_sha,plan['configuration']);guard()
        model=terminal.durable_payload(artifact(phase+'.pt'),lambda stream:torch.save(record,stream),execution.ROOT,guard)
        del record
        native=state.predict_queries(guard)
        native_receipt=save_vector(phase+'-native.npy',native,guard)
        projected=save_vector('known.npy',project_known(native,data,declaration,guard),guard) if phase=='known' else None
        return dict(base(plan_sha),state=execution.RESULTS['fit_'+phase][1],phase=phase,
            data_admission_sha256=data_sha,preflight_sha256=inputs.sha(execution.STUDY/'mps-preflight.json'),
            configuration=plan['configuration'],epochs=state.completed_epoch,native_rows=len(native),
            model=model,native_predictions=native_receipt,known_predictions=projected,epoch_sha256=history,
            saved_model_is_complete_predictor=True,training_resume_supported=False)


def owned_bindings(receipt):
    result={}
    for key in ('model','native_predictions','known_predictions'):
        row=receipt.get(key)
        if row is None:continue
        path=execution.ROOT/row['path']
        # Receipts produced by durable_payload use absolute paths; confine them
        # to this actual scientific run, disallow symlink/path substitution.
        if path.parent!=execution.ART or path.is_symlink() or path.resolve()!=path:
            raise ValueError('Only original owned model/vector payloads may be replayed')
        if path.stat().st_size!=row['bytes']:raise ValueError('Complete original payload bytes required')
        result[str(path.relative_to(execution.ROOT))]=row['sha256']
    return result


def replay(phase,plan,guard,plan_sha):
    torch=torch_runtime();guard();fit_path=execution.STUDY/(phase+'-fit.json');receipt=execution.read(fit_path)
    execution.validate_result('fit_'+phase,receipt,plan_sha)
    records=inputs.verify_files(execution.ROOT,owned_bindings(receipt),guard)
    with reader_for(phase,guard) as (reader,data,declaration,data_sha):
        history=verify_history(phase,reader,data_sha,plan_sha,guard)
        if (receipt['epoch_sha256']!=history or receipt['data_admission_sha256']!=data_sha
                or receipt['configuration']!=plan['configuration'] or receipt['epochs']!=Config().epochs):
            raise ValueError('Original complete actual-model fit/history required')
        guard();record=torch.load(artifact(phase+'.pt'),map_location='cpu',weights_only=True)
        native=terminal.replay(record,reader,phase,data_sha,plan['configuration'],guard,'mps')
        with admission.mapped(artifact(phase+'-native.npy'),records[str(artifact(phase+'-native.npy').relative_to(execution.ROOT))]) as original:
            if (original.dtype!=np.float32 or original.shape!=native.shape or original.tobytes()!=native.tobytes()):
                raise ValueError('Independent complete saved-model replay must match every original f32 bit')
        if phase=='known':
            projected=project_known(native,data,declaration,guard)
            with admission.mapped(artifact('known.npy'),records[str(artifact('known.npy').relative_to(execution.ROOT))]) as original:
                if original.dtype!=np.float32 or original.shape!=projected.shape or original.tobytes()!=projected.tobytes():
                    raise ValueError('Every independently mapped known f32 bit must match')
        # Rehash payloads after use as well as authenticating before deserializing.
        if inputs.verify_files(execution.ROOT,owned_bindings(receipt),guard)!=records:
            raise ValueError('Actual own-model payload identity changed during independent replay')
        return dict(base(plan_sha),state=execution.RESULTS['replay_'+phase][1],phase=phase,
            fit_sha256=inputs.sha(fit_path),data_admission_sha256=data_sha,epoch_sha256=history,
            native_rows=len(native),all_native_prediction_bits_match=True,known_projection_bits_match=phase=='known',
            actual_model_payloads=records,device='mps',configuration=plan['configuration'])


def assemble(plan,guard,plan_sha):
    proofs={};payloads={}
    for phase in ('known','qualifying'):
        guard();fit_path=execution.STUDY/(phase+'-fit.json');replay_path=execution.STUDY/(phase+'-replay.json')
        fit_record=execution.read(fit_path);replay_record=execution.read(replay_path)
        execution.validate_result('fit_'+phase,fit_record,plan_sha);execution.validate_result('replay_'+phase,replay_record,plan_sha)
        if (replay_record['fit_sha256']!=inputs.sha(fit_path) or replay_record['all_native_prediction_bits_match'] is not True
                or (phase=='known' and replay_record['known_projection_bits_match'] is not True)):
            raise ValueError('Both complete independently replayed original phases required')
        payloads[phase]=inputs.verify_files(execution.ROOT,owned_bindings(fit_record),guard)
        if payloads[phase]!=replay_record['actual_model_payloads']:raise ValueError('Independent model payloads changed')
        proofs[str(replay_path.relative_to(execution.ROOT))]=inputs.sha(replay_path)
    data,declaration,data_sha=data_receipt()
    return dict(base(plan_sha),state=execution.RESULTS['assemble'][1],configuration=plan['configuration'],
        data_admission_sha256=data_sha,preflight_sha256=inputs.sha(execution.STUDY/'mps-preflight.json'),
        replay_sha256=proofs,actual_model_payloads=payloads,
        known=dict(path=str(artifact('known.npy')),rows=declaration['known_rows'],sha256=inputs.sha(artifact('known.npy'))),
        qualifying=dict(path=str(artifact('qualifying-native.npy')),rows=declaration['phases']['qualifying']['query_rows'],sha256=inputs.sha(artifact('qualifying-native.npy'))),
        models_fitted=2,epochs_each=15,score_computed=False,
        remaining='Known-label-only blend integration and independent frozen recovered-group audit; no score has been claimed.')


def main(phase):
    plan=execution.authorize(phase);guard=execution.WorkerGuard(phase);guard()
    plan_sha=inputs.sha(execution.REPORT/'plan.json')
    terminal.durable_directory(execution.STUDY,execution.ROOT,guard)
    terminal.durable_directory(execution.ART,execution.ROOT,guard)
    execution.exclusive_json(execution.STUDY/(phase+'-started.json'),dict(base(plan_sha),phase=phase),guard)
    if phase=='data':value=dict(admission.admit(execution.ROOT,guard),**base(plan_sha))
    elif phase=='mps_preflight':
        if 'torch' in sys.modules:raise RuntimeError('Fresh actual MPS import gate required')
        from upstream_factored_rbm_preflight import run
        value=dict(run(guard,'mps',stress=True),**base(plan_sha))
    elif phase.startswith('fit_'):value=fit(phase[4:],plan,guard,plan_sha)
    elif phase.startswith('replay_'):value=replay(phase[7:],plan,guard,plan_sha)
    elif phase=='assemble':value=assemble(plan,guard,plan_sha)
    else:raise ValueError('Declared full-budget phase required')
    execution.validate_result(phase,value,plan_sha)
    execution.exclusive_json(execution.STUDY/execution.RESULTS[phase][0],value,guard)
    print(value['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=execution.PHASES,required=True)
    main(parser.parse_args().phase)
