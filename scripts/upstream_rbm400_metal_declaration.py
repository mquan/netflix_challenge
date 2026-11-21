"""Full-budget deterministic-Metal declaration after actual device proof.

No model launch on import. Only a reviewed exact idle consumer handoff is
permitted here. Both metadata predecessors and the failed preflight remain.
"""
import argparse
import ast
from dataclasses import asdict
import fcntl
import importlib.metadata
import os
from pathlib import Path
import signal
import sys
import time
import upstream_rbm400_observer_recovery_v2 as history
import upstream_factored_rbm_execution_native_v2 as preceding
import upstream_local_rbm_blend_queue_native_v2 as old_consumer
import upstream_factored_rbm_metal_probe_execution as probe

ROOT=history.ROOT
REPORT=ROOT/'experiments/upstream-rbm400-metal-production-protocol'
DEVICE_PROOF_SHA='a8ab47ca345a93213de0dd7c9cdb94d0cdeca2b87fb05d8f241a8a1572c40a24'
MODULES=('upstream_factored_rbm_execution_metal','upstream_factored_rbm_worker_metal',
         'upstream_factored_rbm_training_metal','upstream_factored_rbm_terminal_metal',
         'upstream_factored_rbm_torch_metal','upstream_factored_rbm_prediction_metal',
         'upstream_factored_rbm_preflight_metal','upstream_factored_rbm_metal_reduction',
         'upstream_local_rbm_blend_metal','verify_upstream_local_rbm_blend_metal',
         'upstream_local_rbm_blend_queue_metal')


def read(path):return history.original.read(path)
def sha(path):return history.original.inputs.sha(path)
def inherited_baseline():return history.inherited_baseline()
def copy_baseline(path):return history.copy_baseline(path)


def validate_history():
    history.original_plan();history.previous_plan()
    preceding.check_sources(read(preceding.REPORT/'plan.json'))
    failure=read(preceding.REPORT/'failure.json');read(preceding.REPORT/'owner-release.json')
    if 'mps_preflight' not in failure['error'] or failure['actual_child_RAM_preserved'] is not False:
        raise ValueError('Failed preflight before any learned state required')
    for capture in (read(preceding.REPORT/'monitor.json'),failure['child'],read(preceding.REPORT/'mps_preflight-gate.json')['watchdog']):
        actual=preceding.identity.process(capture['pid'])
        if actual and (preceding.same_child(capture,actual) if 'worker_command' in capture else preceding.identity.same_owner(capture,actual)):
            raise ValueError('Previous failed production process must be terminal')
    if any(preceding.ART.iterdir()):raise ValueError('No learned predecessor artifacts may be discarded')
    if sha(probe.STUDY/'mps-preflight.json')!=DEVICE_PROOF_SHA:
        raise ValueError('Exact independently checked full-device proof required')
    proof=read(probe.STUDY/'mps-preflight.json');done=read(probe.REPORT/'completion.json')
    probe.check_sources(read(probe.REPORT/'plan.json'))
    if done['state']!='complete_actual_full_width_deterministic_metal_preflight_only_no_models' or done['verification_sha256']!=DEVICE_PROOF_SHA:
        raise ValueError('Complete owned device-probe receipt required')
    probe.validate_result('mps_preflight',proof,sha(probe.REPORT/'plan.json'))
    from upstream_factored_rbm_preflight_metal import REQUIRED_STRESS,SMALL_ATOL,SMALL_RTOL,STRESS_ATOL,STRESS_RTOL
    if (proof['cases']['stress']['shape']!=REQUIRED_STRESS or proof['deterministic_algorithms_during_checks'] is not True
            or proof['production_data_read'] is not False or proof['qualifying_labels_read'] is not False):
        raise ValueError('Actual strict deterministic full synthetic MPS proof required')
    for name,atol,rtol in [('small',SMALL_ATOL,SMALL_RTOL),('stress',STRESS_ATOL,STRESS_RTOL)]:
        case=proof['cases'][name]
        if (case['same_device_gradient_repeats_bitwise_equal'] is not True or case['same_device_prediction_repeats_bitwise_equal'] is not True
                or case['atol']!=atol or case['rtol']!=rtol):
            raise ValueError('Original tolerances and repeated device arithmetic required')


def equivalent_modules():
    mappings={
        'upstream_factored_rbm_worker':{
            'upstream_factored_rbm_execution':'upstream_factored_rbm_execution_metal',
            'upstream_factored_rbm_training':'upstream_factored_rbm_training_metal',
            'upstream_factored_rbm_terminal':'upstream_factored_rbm_terminal_metal',
            'upstream_factored_rbm_preflight':'upstream_factored_rbm_preflight_metal'},
        'upstream_factored_rbm_training':{'upstream_factored_rbm_torch':'upstream_factored_rbm_torch_metal','upstream_factored_rbm_prediction':'upstream_factored_rbm_prediction_metal'},
        'upstream_factored_rbm_terminal':{'upstream_factored_rbm_training':'upstream_factored_rbm_training_metal','upstream_factored_rbm_prediction':'upstream_factored_rbm_prediction_metal'}}
    for stem,changes in mappings.items():
        expected=(ROOT/'scripts'/f'{stem}.py').read_text()
        for a,b in changes.items():expected=expected.replace(a,b)
        actual=(ROOT/'scripts'/f'{stem}_metal.py').read_text()
        if ast.dump(ast.parse(expected))!=ast.dump(ast.parse(actual)):
            raise ValueError('Original worker/training/terminal changed beyond declared imports: '+stem)


def idle():
    validate_history();old_consumer.check_sources(read(old_consumer.REPORT/'plan.json'))
    if old_consumer.study.REPORT.exists() or old_consumer.study.ART.exists():raise ValueError('Only idle consumer can be retired')
    if any((old_consumer.REPORT/n).exists() for n in ('child-ownership.json','prerequisites.json','completion.json')):
        raise ValueError('No consumer scientific child may exist')
    if (old_consumer.REPORT/'first-scientific-baseline.json').read_bytes()!=(history.original.REPORT/'first-scientific-baseline.json').read_bytes():
        raise ValueError('Original inherited consumer baseline required')


def capture_idle():
    idle();owner=read(old_consumer.REPORT/'monitor.json')
    if not preceding.identity.same_owner(owner,preceding.identity.process(owner['pid'])):raise ValueError('Exact idle owner required')
    command=' '.join([sys.executable,'-u',str(Path(old_consumer.__file__).resolve()),'--watch',str(owner['pid']),'--lane-fd','3'])
    matches=[preceding.identity.process(pid) for pid,row in preceding.resources.native.processes().items() if row['ppid']==owner['pid'] and row['command']==command]
    if len(matches)!=1 or not matches[0]:raise ValueError('Exact idle watchdog required')
    return dict(state='declared_exact_idle_v2_consumer_retirement_before_new_metal_production',owner=owner,watchdog=matches[0],
                original_baseline_sha256=sha(history.original.REPORT/'first-scientific-baseline.json'),device_proof_sha256=DEVICE_PROOF_SHA)


def retirement():
    value=read(REPORT/'idle-retirement.json')
    if value['state']!='complete_exact_idle_consumer_retirement_no_learned_state' or value['device_proof_sha256']!=DEVICE_PROOF_SHA or value['both_lanes_verified_free'] is not True:
        raise ValueError('Complete exact idle consumer retirement required')
    for capture in (value['owner'],value['watchdog']):
        if preceding.identity.same_owner(capture,preceding.identity.process(capture['pid'])):raise ValueError('Retired idle process remains live')
    return value


def realization():
    return dict(original_configuration_unchanged=True,original_baseline_sha256=sha(history.original.REPORT/'first-scientific-baseline.json'),
                device_proof_sha256=DEVICE_PROOF_SHA,idle_retirement_sha256=sha(REPORT/'idle-retirement.json'),
                reduction='stable_ordered_Metal_single_writer_no_atomics_v1',integer_address_ordering='CPU_stable_sort',
                learned_float_computation='MPS_no_fallback',learned_state_discarded=False)


def check_declaration(plan):
    validate_history();equivalent_modules();retirement()
    old=read(preceding.REPORT/'plan.json')
    if plan.get('numerical_implementation')!=realization() or plan['policy']!=old['policy']:
        raise ValueError('Exact deterministic Metal realization/original limits required')
    for name,digest in old['source_sha256'].items():
        if plan['source_sha256'].get(name)!=digest:raise ValueError('Complete prior source closure required')


def frozen_plan(execution):
    validate_history();equivalent_modules();retirement()
    old=read(preceding.REPORT/'plan.json');sources=dict(old['source_sha256'])
    files=[ROOT/'scripts'/f'{n}.py' for n in MODULES]
    files += [Path(__file__),ROOT/'scripts/test_upstream_rbm400_metal_production.py',ROOT/'scripts/test_upstream_factored_rbm_execution_metal.py',
              ROOT/'scripts/test_upstream_local_rbm_blend_metal.py',ROOT/'OWN_RBM400_METAL_PRODUCTION.md',ROOT/'OWN_LOCAL_RBM_METAL_BLEND_PLAN.md',
              REPORT/'cpu-verification.json',REPORT/'review.json',REPORT/'idle-retirement-plan.json',REPORT/'idle-retirement.json',
              probe.REPORT/'plan.json',probe.REPORT/'completion.json',probe.STUDY/'mps-preflight.json',
              preceding.REPORT/'plan.json',preceding.REPORT/'failure.json',preceding.REPORT/'owner-release.json',preceding.STUDY/'data-admission.json']
    files += [ROOT/name for name in read(probe.REPORT/'plan.json')['source_sha256']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files})
    return dict(state='declared_full400_original15_15_deterministic_metal_production',created=preceding.stamp(),configuration=asdict(execution.Config()),
                phases=list(execution.PHASES),environment=execution.ENV,source_sha256=sources,libraries={n:importlib.metadata.version(n) for n in ('numpy','torch')},
                policy=old['policy'],numerical_implementation=realization(),scope='Full fresh data admission, actual full MPS preflight, both15epoch models and independent native query replay; original known-only downstream16-family protocol; no recovered outcomes in fitting/selection.')


def retire_idle():
    review=read(REPORT/'review.json');cpu=read(REPORT/'cpu-verification.json');plan=read(REPORT/'idle-retirement-plan.json')
    if review['state']!='complete_independent_full_metal_production_and_idle_retirement_review' or review['findings'] or review['cpu_verification_sha256']!=sha(REPORT/'cpu-verification.json'):
        raise ValueError('Exact independently reviewed production/retirement required')
    for name,digest in cpu['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Reviewed production source changed')
    actual=capture_idle()
    for key in ('owner','watchdog'):
        if not preceding.identity.same_owner(plan[key],actual[key]):raise ValueError('Prepared idle identity changed')
    owner=plan['owner'];preceding.exclusive_json(REPORT/'idle-retirement-begin.json',dict(plan=plan,observed=preceding.stamp()))
    os.kill(owner['pid'],signal.SIGSTOP)
    for _ in range(100):
        p=preceding.identity.process(owner['pid'])
        if p and preceding.identity.same_owner(owner,p) and 'T' in p['state']:break
        time.sleep(.05)
    else:raise RuntimeError('Idle owner stop not observed')
    idle()
    p=preceding.identity.process(owner['pid'])
    if not preceding.identity.same_owner(owner,p) or p['pgid']!=owner['pgid'] or p['ppid']!=owner['ppid']:
        raise ValueError('Exact frozen idle owner required')
    preceding.exclusive_json(REPORT/'idle-retirement-frozen.json',dict(owner=p,no_scientific_state=True,observed=preceding.stamp()))
    os.kill(owner['pid'],signal.SIGTERM);os.kill(owner['pid'],signal.SIGCONT)
    for capture in (owner,plan['watchdog']):
        for _ in range(150):
            if not preceding.identity.same_owner(capture,preceding.identity.process(capture['pid'])):break
            time.sleep(.1)
        else:raise RuntimeError('Idle owner/watchdog did not become terminal')
    streams=[]
    try:
        for name in ('experiments/full-upstream-reproduction/runner.lock','experiments/upstream-source-extension/run.lock'):
            stream=(ROOT/name).open('a');streams.append(stream);fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preceding.exclusive_json(REPORT/'idle-retirement.json',dict(state='complete_exact_idle_consumer_retirement_no_learned_state',owner=owner,watchdog=plan['watchdog'],
            device_proof_sha256=DEVICE_PROOF_SHA,both_lanes_verified_free=True,learned_state_discarded=False,observed=preceding.stamp()))
    finally:
        for stream in streams:stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-idle-retirement',action='store_true');parser.add_argument('--retire-idle',action='store_true');args=parser.parse_args()
    if args.prepare_idle_retirement and not args.retire_idle:
        REPORT.mkdir(exist_ok=True);preceding.exclusive_json(REPORT/'idle-retirement-plan.json',capture_idle())
    elif args.retire_idle and not args.prepare_idle_retirement:retire_idle()
    else:parser.error('Explicit single mode required; importing never starts work')
